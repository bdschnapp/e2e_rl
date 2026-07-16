"""Ego angular obstacle-clearance profile (Phase-3; NEW, additive).

A coarse "obstacle lidar distilled from vision": for each of `nbins` bearing bins over the
forward FOV, the nearest obstacle EDGE-clearance among obstacles whose angular extent covers
that bearing (far sentinel = R when the bin is clear), normalised by R -> [0,1] (1 = clear).

Chosen over a per-obstacle list because it is order-invariant (no range-swap flicker),
count-agnostic (handles any number of obstacles, not a fixed 2), continuous, and directly
comparable to the real-lidar baseline. Used both as the TRUE obstacle state for the
privileged teacher policy and as the LABEL the distillation encoder predicts from the image.
"""

from __future__ import annotations

from .backend import xp


def obstacle_range_profile(ox, oy, orad, ovalid, lx, ly, lyaw, R, fov_deg=180.0, nbins=15):
    """(N, nbins) obstacle-only clearance profile in [0,1] (1 = clear), forward-facing."""
    N, Kobs = ox.shape
    f32 = xp.float32
    c = xp.cos(lyaw)[:, None]; s = xp.sin(lyaw)[:, None]
    dx = ox - lx[:, None]; dy = oy - ly[:, None]
    fwd = dx * c + dy * s                                   # forward in ego frame
    lat = -dx * s + dy * c                                  # lateral (left +)
    rng = xp.sqrt(fwd * fwd + lat * lat)                    # (N,K)
    bearing = xp.arctan2(lat, fwd)                          # (N,K) 0 = ahead
    half = xp.arcsin(xp.clip(orad / xp.maximum(rng, 1e-6), 0.0, 1.0))   # obstacle angular half-width
    valid = ovalid & (rng <= R)
    fov = float(fov_deg) * 3.14159265 / 180.0
    edges = xp.linspace(-fov / 2, fov / 2, nbins).astype(f32)           # (nbins,) bin bearings
    covered = valid[:, :, None] & (xp.abs(bearing[:, :, None] - edges[None, None, :]) <= half[:, :, None])
    edge_clr = xp.maximum(rng - orad, 0.0)[:, :, None]                  # (N,K,1)
    big = xp.asarray(1e9, dtype=f32)
    prof = xp.where(covered, edge_clr.astype(f32), big).min(axis=1)     # (N,nbins)
    prof = xp.where(prof >= big, xp.asarray(float(R), dtype=f32), prof) # far sentinel
    return (prof / float(R)).astype(f32)                               # (N,nbins) in [0,1]


def obstacle_profile_names(nbins):
    return [f"obs_bin{b}" for b in range(nbins)]
