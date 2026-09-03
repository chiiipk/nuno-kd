"""
Recovery-slope layer weighting.

Idea: matrix entropy of hidden states across transformer layers typically shows
a U-shape:
  - early layers:  high entropy (token-level diversity)
  - middle layers: VALLEY (attention compression)
  - late layers:   RECOVERY (entropy rises again as features specialize for output)

We want the structural regularizer (NNM/BNM/...) to focus on the RECOVERY phase
because that's where the teacher's representational structure is most
informative and the student is most likely to under-fit.

Pipeline:
  1. After teacher pre-pass, run the teacher one extra time on a few batches
     to record matrix entropy per layer → list of (n_layers,) values H[ℓ].
  2. Compute slope dH/dℓ via central difference.
  3. Identify the valley index ℓ* = argmin H.
  4. Weight w[ℓ] is non-zero only for ℓ > ℓ* and proportional to max(0, slope[ℓ]),
     normalized so max(w) = 1.
  5. Cache the dict {s_lid → w[s_lid]} and return it; replaces the old
     `layer_weight` Gaussian.

This function is meant to be called ONCE in prepare_nnm, then the resulting
dict is stored in nnm_state["layer_weights"] like before — no other code
change required.
"""

import math
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Matrix entropy primitive
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def matrix_entropy(H: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Matrix entropy of a [N, d] hidden-state matrix:
        sigma = svdvals(H)
        p     = sigma / sum(sigma)
        H_mat = -sum p log p           (in nats)

    This is the LOG of the effective rank. We use log-form because slopes
    are easier to compare in nats and we don't need exp() of erank here.
    """
    H = H.float()
    s = torch.linalg.svdvals(H)
    s = s / (s.sum() + eps)
    return float(-(s * (s + eps).log()).sum().item())


# ═══════════════════════════════════════════════════════════════
#  Build entropy curve from teacher
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def build_teacher_entropy_curve(
    teacher,
    dataloader,
    n_layers: int,
    max_batches: int = 50,
    device=None,
) -> list:
    """
    Run teacher forward on `max_batches` and accumulate matrix entropy per
    transformer block output. Returns a list of length (n_layers + 1) — index 0
    is the embedding output, indices 1..n_layers are block outputs (matches
    HuggingFace's `output_hidden_states` convention).

    Uses only response tokens (`labels != -100`) when available; otherwise
    falls back to all valid positions.
    """
    from tqdm import tqdm

    if device is None:
        device = next(teacher.parameters()).device

    teacher.eval()

    sums   = [0.0] * (n_layers + 1)
    counts = [0]   * (n_layers + 1)

    for i, batch in enumerate(tqdm(dataloader,
                                   desc="NNM entropy curve pre-pass",
                                   total=max_batches)):
        if i >= max_batches:
            break
        ids   = batch.get("input_ids")
        mask  = batch.get("attention_mask")
        labels = batch.get("labels")
        if ids is None or mask is None:
            continue
        ids  = ids.to(device)
        mask = mask.to(device)

        out = teacher(ids, attention_mask=mask,
                      output_hidden_states=True, return_dict=True)

        if labels is not None:
            labels = labels.to(device)
            if labels.shape == ids.shape:
                flat_mask = (labels.reshape(-1) != -100)
            else:
                flat_mask = mask.reshape(-1).bool()
        else:
            flat_mask = mask.reshape(-1).bool()

        if not flat_mask.any():
            continue

        for lid in range(n_layers + 1):
            h = out.hidden_states[lid]
            h = h.reshape(-1, h.shape[-1])[flat_mask]
            if h.shape[0] < 2:
                continue
            sums[lid]   += matrix_entropy(h)
            counts[lid] += 1

    curve = []
    for s, c in zip(sums, counts):
        curve.append(s / c if c > 0 else 0.0)
    return curve


# ═══════════════════════════════════════════════════════════════
#  Slope-based weights
# ═══════════════════════════════════════════════════════════════

def recovery_slope_weights(curve: list, normalize: bool = True) -> list:
    """
    Given entropy curve H[ℓ] (length n_layers + 1), return per-layer weights
    that are HIGH for layers in the recovery slope (after the valley) and
    ZERO before the valley.

    Algorithm:
      1. valley = argmin(curve) (over interior layers, so we skip the very
         first/last to avoid edge artifacts).
      2. slope = central difference of curve.
      3. weight[ℓ] = max(0, slope[ℓ]) if ℓ > valley, else 0.
      4. (optional) normalize so max(weight) = 1.

    Returns a list of length (n_layers + 1). Layer 0 (embedding) is always 0.
    """
    n = len(curve)
    if n < 5:
        # too short to do anything meaningful — fall back to uniform
        return [1.0] * n

    # 1) valley index (search interior only)
    lo, hi = 1, n - 1
    interior = curve[lo:hi]
    valley = lo + min(range(len(interior)), key=lambda i: interior[i])

    # 2) central-difference slope
    slope = [0.0] * n
    for i in range(1, n - 1):
        slope[i] = (curve[i + 1] - curve[i - 1]) / 2.0
    slope[0]    = curve[1]    - curve[0]
    slope[-1]   = curve[-1]   - curve[-2]

    # 3) gate on post-valley positive slope
    w = [0.0] * n
    for i in range(n):
        if i > valley:
            w[i] = max(0.0, slope[i])

    # 4) normalize
    if normalize:
        mx = max(w)
        if mx > 0:
            w = [x / mx for x in w]

    return w


# ═══════════════════════════════════════════════════════════════
#  Public entry point — drop into prepare_nnm
# ═══════════════════════════════════════════════════════════════

def compute_recovery_layer_weights(
    teacher,
    dataloader,
    s_mid: list,
    t_mid: list,
    n_s_layers: int,
    n_t_layers: int,
    max_batches: int = 50,
    device=None,
    debug_print=print,
) -> dict:
    """
    Drop-in replacement for the old Gaussian `layer_weight(l, L)`.

    Returns a dict {s_lid -> weight} suitable for nnm_state["layer_weights"].
    Weights are derived from the teacher's matrix-entropy curve so they
    capture "recovery slope" structure rather than a hand-picked Gaussian.

    Caller is responsible for running this only on rank 0 if multi-GPU; the
    resulting (tiny) dict can be broadcast cheaply.
    """
    debug_print("[NNM] computing teacher entropy curve for recovery-slope weights...")
    curve = build_teacher_entropy_curve(
        teacher, dataloader, n_t_layers,
        max_batches=max_batches, device=device,
    )
    w_full = recovery_slope_weights(curve, normalize=True)

    valley = max(range(1, len(curve) - 1), key=lambda i: -curve[i])
    debug_print(f"[NNM] teacher entropy curve: "
                f"min @ layer {valley} (H={curve[valley]:.3f}), "
                f"max @ layer {max(range(len(curve)), key=lambda i: curve[i])} "
                f"(H={max(curve):.3f})")

    # Map TEACHER-layer weights → STUDENT-layer slots, using the s_mid/t_mid
    # pairing decided earlier. We index w_full by t_lid (teacher position
    # whose recovery slope we measured) and assign that weight to s_lid.
    s_weights = {}
    for s_lid, t_lid in zip(s_mid, t_mid):
        s_weights[s_lid] = w_full[t_lid]

    # If every selected layer turned out to have weight 0 (all sit before
    # valley), fall back to small positive uniform so loss isn't pure zero.
    if max(s_weights.values()) == 0.0:
        debug_print("[NNM] WARNING: all selected layers are pre-valley; "
                    "falling back to uniform weights = 1.0")
        s_weights = {k: 1.0 for k in s_weights}

    debug_print(f"[NNM] recovery weights: {s_weights}")
    return s_weights