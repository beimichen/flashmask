"""flashmask — detect and mask text regions in diagram images.

The heavy ML stacks (torch, ultralytics, doctr) are imported lazily inside the
functions that need them, so ``import flashmask`` stays cheap and the inference
path works without the optional ``[train]`` extras installed.
"""

from flashmask.config import PROJECT_ROOT, paths, set_seed

__version__ = "0.1.0"

__all__ = ["PROJECT_ROOT", "paths", "set_seed", "__version__"]
