"""Small model-structure helpers without training-framework dependencies."""


def _get_nested(obj, path):
    for name in path:
        if not hasattr(obj, name):
            return None
        obj = getattr(obj, name)
    return obj


def resolve_transformer_layers(model):
    """Return decoder layers across raw HF, PEFT, and wrapped model layouts."""
    while hasattr(model, "module"):
        model = model.module

    candidate_paths = (
        ("model", "layers"),                  # Qwen/Llama causal LM
        ("base_model", "model", "layers"),  # HF base_model delegation
        ("base_model", "model", "model", "layers"),  # PEFT wrapper
        ("transformer", "h"),                # GPT-2 style
    )
    for path in candidate_paths:
        layers = _get_nested(model, path)
        if layers is not None:
            return layers

    layouts = [".".join(path) for path in candidate_paths]
    raise AttributeError(
        f"Cannot locate transformer layers in {type(model).__name__}; "
        f"tried {layouts}"
    )
