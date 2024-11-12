import torch
import logging

# torch/nested/_internal/nested_tensor.py:417: UserWarning: Failed to initialize NumPy: No module named 'numpy'
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy: No module named 'numpy'")


# We use this "hack" to set torchao.__version__ correctly
# the version of ao is dependent on environment variables for multiple architectures
# For local development this will default to whatever is version.txt
# For release builds this will be set the version+architecture_postfix
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("torchao")
except PackageNotFoundError:
    __version__ = 'unknown'  # In case this logic breaks don't break the build

_IS_FBCODE = (
    hasattr(torch._utils_internal, "IS_FBSOURCE") and
    torch._utils_internal.IS_FBSOURCE
)
if not _IS_FBCODE:
    try:
        from importlib.util import find_spec
        from pathlib import Path
        spec = find_spec("torchao")
        assert spec is not None, "torchao python module spec is unexpectedly None"
        SO_PATH = Path(spec.origin).parent / "_C.abi3.so"
        torch.ops.load_library(SO_PATH)
        from . import ops
    except:
        logging.info("Skipping import of cpp extensions")

from torchao.quantization import (
    autoquant,
    quantize_,
)
from . import dtypes
from . import testing

__all__ = [
    "dtypes",
    "autoquant",
    "quantize_",
    "testing",
]

# test-pytorchbot
# test-codev
