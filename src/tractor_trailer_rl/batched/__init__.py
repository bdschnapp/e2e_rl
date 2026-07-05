"""GPU-parallel (batched) tractor-trailer environment.

This subpackage is the CuPy/GPU port of the scalar `tractor_trailer_rl` env. It
runs N environments at once with a leading (N,) batch axis on every state array,
so a whole fleet steps in a single vectorised call and can live on the GPU.

Design:
  * `backend.xp` selects numpy (CPU, for parity testing) or cupy (GPU) — the same
    code runs on either. Pick with the ``TTRL_BACKEND`` env var or
    ``backend.set_backend("cupy")``.
  * The scalar env in ``tractor_trailer_rl.envs.base`` remains the parity oracle;
    ``tests/`` assert the batched env matches it to <=1e-4.
  * The five scalar blockers are replaced with vectorised equivalents:
      1. per-step scipy ``cont2discrete`` ZOH  -> batched matrix-exponential (expm.py)
      2. scipy ``cKDTree`` occupancy build      -> batched point-to-polyline (occupancy.py)
      3. sequential lidar ray-march             -> fixed-length vectorised march (lidar.py)
      4. proximity/collision Python loops       -> batched gather + any (env.py)
      5. scalar termination/stop/spawn branches -> boolean masks + internal auto-reset (env.py)
"""

from .backend import xp, get_backend, set_backend, to_numpy, asarray  # noqa: F401
