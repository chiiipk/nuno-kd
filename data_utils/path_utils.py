"""Path helpers shared by indexed dataset readers."""

import os


def dataset_file_prefix(path, name, state):
    """Build a shard prefix without requiring a trailing slash in ``path``."""
    return os.path.join(os.fspath(path), f"{name}_{state}")
