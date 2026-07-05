"""Array-backend switch: numpy (CPU) or cupy (GPU).

The whole batched env is written against ``xp`` (an alias for the active array
module) using only the numpy/cupy common subset, so identical code runs on CPU
for parity testing and on GPU for throughput. cupy implements the same API as
numpy, so porting is a module swap, not a rewrite.

Selection order:
  1. ``set_backend("cupy"|"numpy")`` at runtime, or
  2. the ``TTRL_BACKEND`` environment variable, or
  3. auto: cupy if importable, else numpy.

``to_numpy`` / ``asarray`` bridge host<->device so tests and the training loop can
move data without caring which backend is active.
"""

from __future__ import annotations

import os

import numpy as _np

_BACKEND = None
_XP = None


def _load(name: str):
    if name == "cupy":
        import cupy as cp  # noqa: F401  (raises ImportError if unavailable)
        return "cupy", cp
    return "numpy", _np


def _auto() -> str:
    if os.environ.get("TTRL_BACKEND"):
        return os.environ["TTRL_BACKEND"].lower()
    try:
        import cupy  # noqa: F401
        return "cupy"
    except Exception:
        return "numpy"


def set_backend(name: str):
    """Force the array backend ('numpy' or 'cupy'). Returns the array module."""
    global _BACKEND, _XP
    _BACKEND, _XP = _load(name.lower())
    return _XP


def _ensure():
    global _BACKEND, _XP
    if _XP is None:
        try:
            _BACKEND, _XP = _load(_auto())
        except Exception:
            _BACKEND, _XP = _load("numpy")
    return _XP


def get_backend() -> str:
    _ensure()
    return _BACKEND


def to_numpy(a):
    """Return a host numpy array regardless of backend."""
    _ensure()
    if _BACKEND == "cupy":
        import cupy as cp
        return cp.asnumpy(a)
    return _np.asarray(a)


def asarray(a, dtype=None):
    """Move/convert an array onto the active backend."""
    xp_ = _ensure()
    return xp_.asarray(a, dtype=dtype)


class _XPProxy:
    """Lazy attribute proxy so ``from ...backend import xp`` binds to whatever
    backend is active at call time (not import time)."""

    def __getattr__(self, item):
        return getattr(_ensure(), item)


xp = _XPProxy()
