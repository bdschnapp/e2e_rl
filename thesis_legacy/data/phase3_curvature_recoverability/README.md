# Why is obstacle curvature recoverability low? (2026-07-23)

Probe of the obstacle image->state encoder to localise the soft curvature recoverability in
Table 5.12. Script: `scripts/curvature_recoverability_probe.py`. Raw:
`probe_{reverse_dynamic,forward_static}.{txt,json}`. 307k samples, 25 epochs, test-split R^2.

**Note on geometry:** obstacle path spacing is **0.25 m**, so k1/k2 are curvature at **2.5 m / 5.0 m**
look-ahead (10 / 20 samples) — not 10/20 m.

## Headline: the deficit is curvature ONLY, and only in obstacle-diverted sections

**Per-component R^2 (full):**

| dim | forward static | reverse dynamic |
|---|---|---|
| e_y | 0.952 | 0.970 |
| e_psi | 0.896 | 0.876 |
| e_y_t | 0.975 | 0.981 |
| e_psi_t | 0.986 | 0.883 |
| **k1** | **0.644** | **0.451** |
| **k2** | **0.593** | **0.397** |
| d_win | 0.891 | 0.950 |

Only **k1/k2 are soft**; the errors (0.88-0.99) and difficulty (0.89-0.95) are fine. (So "e_psi is
bad" is overstated — 0.88-0.90 is decent; it is just below the 0.998 of the smooth non-obstacle case.)

**Curvature R^2 stratified by obstacle-divergence** (how far the hidden avoidance path bends off the
lane centreline over the look-ahead window):

| divergence (m) | forward k1 R^2 | reverse k1 R^2 |
|---|---|---|
| [0.0, 0.2) clear (~centreline) | **0.965** | **0.766** |
| [0.2, 1.0) | 0.244 | 0.320 |
| [1.0, 3.0) | -- | 0.087 |
| [3.0, +) strong divert | -- | 0.136 |

**On clear path the encoder recovers curvature well (forward 0.965 -> approaching the non-obstacle
0.997); it collapses only where the avoidance path diverts around obstacles.** So the deficit is
localised to the obstacle-diverted bends, exactly as hypothesised — not a general perception failure.

## Why those bends are hard (and why the obvious fixes don't work)
- **The diverted curvature is intrinsically high-frequency.** A smoothing cubic fit of the *true*
  path offsets fails to reproduce the true curvature: smooth-fraction R^2 = **-0.58 (reverse)**, +0.65
  (forward). So the reverse potential-field bends are sharp/local; forward's are gentler -> reverse
  curvature (0.45) < forward (0.64).
- **Predicting path offsets and differentiating does NOT help** (the "path-point" idea): offsets
  recover at ~0.98, but exact finite-difference of the predictions amplifies noise catastrophically
  (R^2 ~ -88), and a smoothing cubic of the predictions gives ~direct or worse (rev -0.42, fwd +0.57).
  Sanity: exact FD from *true* offsets reproduces k1/k2 at R^2 = 1.000 (derivation validated).
- **e_psi is NOT recovered better from proprioception** (an idea we tested and rejected): predicting
  e_psi from `[predicted e_psi_t, hitch, steer]` is *worse* than the direct image prediction (rev
  0.81 vs 0.88; fwd 0.34 vs 0.90), and the e_psi residual is uncorrelated with hitch/steer (~0.01).
  The e_psi_t >= e_psi pattern is the tracked-body (trailer, path-determined heading) vs
  actively-steering-body (tractor) effect, not a leading-body or hitch effect.
- k1 error IS concentrated on high curvature (|resid| vs |true curv| corr = 0.73 rev / 0.66 fwd) --
  consistent with the sharp-bend localisation.

## Implication for the fix
The bottleneck is not encoder capacity or image resolution -- it is the **sharp potential-field
avoidance path**. Its curvature at the diverted bends is high-frequency, so it is both hard to read
from the image AND (arguably) not the right thing to feed-forward. The productive fix is a **smoother
avoidance reference** (spline-smoothed or optimisation-based planning): gentler curvature that is both
image-recoverable and easier to track, addressing the random-failure issue at its source (planning),
not perception. Curvature is a feed-forward term; the stabiliser is the *measured* hitch feedback, so
this is an anticipation improvement, not a stability prerequisite.
