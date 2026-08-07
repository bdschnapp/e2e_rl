# Review checklist — `0-Ben-main-amir-review.pdf`

Source: 73 PDF annotations (56 `bdschnap`, 17 `amirk`) extracted from
`~/Downloads/0-Ben-main-amir-review.pdf` (annotated 2026-08-02, built from the
2026-07-26 PDF). Raw dump with page numbers, anchors, and comment text:
`thesis/amir_review_annotations.json`.

**28 annotations resolved** by the mechanical pass (see bottom).
**45 remain**: 9 from Amir, 36 own notes.

Structural work (chapter merge, section moves, missing content, and Amir's email
comments) is planned separately in `REVISION_PLAN_DRAFT2.md`, which references
the `A*`/`B*` item numbers below.

---

## A. Amir's comments (9 open)

| # | p. | Location | What he asks | Notes |
|---|---|---|---|---|
| A1 | 21 | `02_literature_review.tex:28` | "The appeal of this pipeline is that it comes with guarantees" is a strong statement, add a reference | Candidates already in `references.bib`: `anderson1990optimal` (LQR optimality), `rawlings2017mpc` / `mayne2000constrained` (MPC stability + feasibility) |
| A2 | 22 | `02_literature_review.tex:34` onward | Subsections are very small: drop the sub-subsection numbers, keep the title in bold | Scope decision. Anchored on §2.3.1; the same pattern runs through ch. 2. Cheapest route: `\subsection*` + `\addcontentsline`, or bump those levels to the new display-`\paragraph` style |
| A3 | 30 | `03_modeling_simulation_problem.tex:15-19`, `figures/tikz/tractor_trailer_geometry` | Make the tiers larger; rename **hitch → fifth wheel**; add a justification for placing the fifth wheel at the tractor rear axle | Terminology change is thesis-wide (γ is "hitch articulation angle" throughout, plus `$L_h$` in the symbol list and `P_jack`/`no-hitch` reward labels). Decide once, then it is a global rename |
| A4 | 36 | end of ch. 3 (`:93`) and each chapter | Add a summary at the end of every chapter | Also asked again at p. 57 (A7). Ch. 6 already has a `Summary` section; ch. 2–5 do not |
| A5 | 46 | `04_learning_framework.tex:88` | If "dimension" means the number of independent inputs, say so | Interacts with own note B6 (8-dim vs 2-dim narrative) — resolve together |
| A6 | 50 | `04_learning_framework.tex:174` | "cheap **direct measurements**" → "measurements or estimates" | Own note on the same phrase wants "calculations based on on-board sensor measurements". Pick one wording; ch. 5:89 has the parallel phrase "cheaply available from onboard sensing and state estimation" |
| A7 | 57 | `04_learning_framework.tex:278` (end of ch. 4) | Add a summary here, **and discuss transferability of the trained network to a different tractor or trailer** — he flags this as the main question readers will have | The substantive one. Nothing in the thesis currently addresses cross-vehicle transfer; needs either an experiment or an explicit limitations/future-work treatment (ch. 7) |
| A8 | 70 | `05_results_discussion.tex:140` | Reverse motion is inherently unstable, so the forward-motion approach may not carry over; suggests a form of feedback in the network for stability | Reads as a design suggestion (recurrence / error feedback in the policy). Ch. 5 already argues the forward recipe does *not* transfer and fixes it with state-dependent difficulty — worth answering that explicitly, and noting recurrent policies as future work |
| A9 | 86 | `06_sim_to_real.tex:144` | Questions the actuator-delay premise: such steering delay is not considered in higher-speed vehicle control, and the training data contains no delay to compensate | Either justify the Smith predictor with measured hardware delay from the platform, or reframe/cut the section |

## B. Own notes (36 open)

### B.1 Data / results correctness — blocking for draft 2
| # | p. | Location | Issue |
|---|---|---|---|
| B1 | 62 | `05_results_discussion.tex:50` | Table cells import as "— —"; debug `scripts/generate_phase_tables.py` |
| B2 | 73 | `tables/`, geometry recoverability | κ₁/κ₂ R² values (0.699/0.425, 0.558/0.277) are outdated; current runs are much higher. Determine whether it is an import bug or missing data |
| B3 | 63 | `05_results_discussion.tex:71` | "the reverse image run was set aside for the dedicated study of the next section" is no longer true — may be the same import failure |
| B4 | 56 | `04_learning_framework.tex:244` (`tab:sim_throughput`) | Report exact throughput numbers instead of `~`, and add a BEV observation row |
| B5 | 60 | `05_results_discussion.tex:28` | Report forward tracking results, and state that multiplicative TD3 was best in **both** directions |

### B.2 Claims that no longer match the implementation
| # | p. | Location | Issue |
|---|---|---|---|
| B6 | 46 | `04_learning_framework.tex:88` | Keep the 8-dim framing here (stable 8-dim state → add perception → destabilizes → motivates the minimal 2-dim state + perception later). Highlight says "2 dimensional"; sticky note says keep 8-dim — the arc needs writing out. Pairs with A5 |
| B7 | 45, 49 | `04_learning_framework.tex:81`, `:174` | **Checked against `tractor_trailer_rl_cupy` on 2026-08-05: the thesis is correct, the note was not.** `batched/bev_sdf.py::render_bev_sdf` renders a signed-distance field (+1 at centerline, 0 at the corridor edge, negative outside, clipped to [-1,1], normalized by corridor half-width) plus 2 CoordConv channels, and the geometry-encoder pipeline (`geom_obs_env` → `SdfBevEnv`, `geom_pretrain.py`/`geom_frozen_rl.py` via `SdfVecEnv`, final campaigns run with `--coord`) uses exactly that. There is no soft occupancy grid anywhere in the code ("soft", "graded", "GSD" all appear 0 times; `sdf` appears 117 times). Only change needed: terminology consistency — drop "graded"/"GSD" and call it the SDF-BEV throughout, matching the code. Optional accuracy note: obstacle interiors are set to a hard $-1$ rather than a graded distance, so the field is graded only with respect to the corridor edge |
| B8 | 54 | `04_learning_framework.tex:213`, and `05_results_discussion.tex:14` | "the stop action is disabled during the lane-following and drivability evaluations" is false — it is always enabled so every policy can choose to stop. Both the ch. 4 protocol sentence and the ch. 5 restatement need rewriting, and the "so that completion measures tracking rather than stop-timing" justification has to be replaced |
| B9 | 68 | `05_results_discussion.tex:116` | Calling obstacle slots appended to the engineered state a "different high-dimensional input" is not fair; find another explanation for the collapse |
| B10 | 70 | `05_results_discussion.tex:140` | Explicitly define **dynamic difficulty** = static difficulty inflated by the current error from the obstacle-aware path |

### B.3 Prose rewrites (author-supplied direction, needs new sentences)
| # | p. | Location | Issue |
|---|---|---|---|
| B11 | 25 | `02_literature_review.tex:63` | "compared on an equal footing" is not accurate: vision gets an encoder, LiDAR does not |
| B12 | 46 | `04_learning_framework.tex:112` | "The first and most direct decouples at the level of the representation." — reword |
| B13 | 50 | `04_learning_framework.tex:174` | "direct measurements" → "calculations based on on-board sensor measurements" (reconcile with A6) |
| B14 | 55 | `04_learning_framework.tex:227` | The vectorized LiDAR ray-march with first-hit reduction needs a better explanation |
| B15 | 63 | `05_results_discussion.tex:76` | "the obstacle task requires the image" → it does not *require* the image; the obstacle task needs information the engineered state does not contain |
| B16 | 64 | `05_results_discussion.tex:85` | "If the trouble were that the image is a poor learning target, one of these would have improved completion." — claim is too aggressive |
| B17 | 65 | `05_results_discussion.tex:85` | Record the actual procedure: freeze actor → train critic → unfreeze actor to try to improve. Unfreezing did not improve CTE and dropped one seed's completion |
| B18 | 65 | `05_results_discussion.tex:89` | Is "distilling" the right term for the frozen-encoder recipe? |
| B19 | 67 | `05_results_discussion.tex:112` | "a compact tracking summary is blind to obstacles that are not on the reference path" → the engineered state does not include information about obstacles a lane-center-tracking vehicle could collide with |
| B20 | 70 | `05_results_discussion.tex:134` | The systematic 0.8 m reverse off-path error argument needs a cleaner explanation |

### B.4 Structure — cuts and moves
| # | p. | Location | Issue |
|---|---|---|---|
| B21 | 43 | `04_learning_framework.tex:64` (§4.4 HPO) | Candidate cut; a note in ch. 5 that different algorithms used different hyperparameters may suffice |
| B22 | 51 | `04_learning_framework.tex:180` (§4.6.5 Safe Adaptation) | Candidate cut: helped behavior cloning, not useful for the geometry encoder |
| B23 | 52 | `04_learning_framework.tex:188` (Failure Replay Curriculum) | Candidate cut: helps only strategies that already struggle to learn |
| B24 | 53, 58 | `04_learning_framework.tex:202` → `05_results_discussion.tex:14` | Move §4.8 Evaluation Methodology into §5.1 so the protocol is discussed once |
| B25 | 57 | `04_learning_framework.tex:278` | Cut the hand-written GPU-resident TD3 — it was no different from SB3 |
| B26 | 59 | `05_results_discussion.tex:18` | Cut the "word on the evaluation pool" paragraph from the final version once the fair-pool audit is verified |
| B27 | 76 | `tables/benchmark_results.tex:23` | Shorten captions — move analysis out of captions into body text (applies to other tables too) |
| B28 | 89 | `06_sim_to_real.tex:202` | Cut or substantially shorten the high-fidelity geographic simulation section: a real contribution, but lab maps with collected LiDAR were used for most testing |
| B29 | 34 | `03_modeling_simulation_problem.tex:76` | Naming: "lane driving" vs "line following". Amir's answer is "lane following", and the heading is now **Lane Following**. If you still want "lane driving" thesis-wide (cf. the branch name), that is a separate global rename |

### B.5 Figures
| # | p. | Location | Issue |
|---|---|---|---|
| B30 | 30 | `figures/tikz/tractor_trailer_geometry` | Too compact / unclear (same figure as A3) |
| B31 | 68 | `05_results_discussion.tex:122` or wherever the hidden path is first introduced | Add a figure showing the hidden avoidance path |
| B32 | 83 | `06_sim_to_real.tex:86` (fig. 6.4) | Clean up for readability |
| B33 | 87 | `06_sim_to_real.tex:166` (fig. 6.6) | Fix arrow overlap |

### B.6 Already answered by Amir — no action
| # | p. | Issue |
|---|---|---|
| B34 | 20 | "How do I talk about myself?" → Amir: use passive voice generally to avoid first person, but this instance is fine as-is |

*(B35, B36 = the two "potential to cut" duplicates folded into B22/B23 above; every annotation is accounted for in `amir_review_annotations.json`.)*

---

## Applied in the mechanical pass (28 annotations, all verified to compile)

**Run-in heading style** — resolves 6 annotations (p. 40 ×2, 41 ×3, 42 "fix all such cases") plus your two "new line" notes: `\paragraph` redefined in `main.tex` as a display heading (bold, own line), and the trailing period stripped from all 12 `\paragraph{...}` headings in `equations/reward_function.tex`, `chapters/04_learning_framework.tex`, `chapters/06_sim_to_real.tex`. Amir wrote ":" on two of the five, but with the heading on its own line neither a period nor a colon is needed — say if you would rather have colons on run-in headings instead.

| p. | Change | File |
|---|---|---|
| 25 | "latent code" → "latent space" | `02_literature_review.tex` |
| 25 | "map of obstacle probability" → "map of drivable space" | `02_literature_review.tex` |
| 34 | `\subsection{Line Following}` → `{Lane Following}` | `03_modeling_simulation_problem.tex` |
| 34 | "Reverse line-following" → "Reverse lane-following" | `03_modeling_simulation_problem.tex` |
| 41 | "difficulty-gated" → "difficulty-scaled" | `equations/reward_function.tex` |
| 47 | struck "however," | `04_learning_framework.tex` |
| 49 | "`state` agent" → "`state` observation agent" | `04_learning_framework.tex` |
| 53 | removed "or timeout" from the outcome categories | `04_learning_framework.tex` |
| 53 | struck "continuous" | `04_learning_framework.tex` |
| 54 | "matrix of agents" → "matrix of policies" | `04_learning_framework.tex` |
| 55 | "environments restart" → "environments reset, drawing a new path and setting a new vehicle pose, and restart" | `04_learning_framework.tex` |
| 56 | struck "(no LiDAR)" from the throughput table | `04_learning_framework.tex` |
| 58 | "the same path the design took" → "the same order the design took" | `05_results_discussion.tex` |
| 64 | "none of them works" → "none of them work" | `05_results_discussion.tex` |
| 65 | "the vehicle's geometry" → "the vehicle and path geometry" | `05_results_discussion.tex` |
| 67 | "a learned controller" → "a learned controller and perception system" | `05_results_discussion.tex` |
| 70 | struck both "now" | `05_results_discussion.tex` |
| 73 | `\subsection{A Deterministic Safety Gate}` → `{Deterministic Safety Gate}` | `05_results_discussion.tex` |
