# Review checklist — `0-Ben-main-amir-review.pdf`

Source: 73 PDF annotations (56 `bdschnap`, 17 `amirk`) from
`~/Downloads/0-Ben-main-amir-review.pdf` (annotated 2026-08-02). Raw dump with
page numbers, anchors, and comment text: `thesis/amir_review_annotations.json`.

Structural work is planned in `REVISION_PLAN_DRAFT2.md`, which references the
`A*`/`B*` ids below.

**Status as of 2026-08-07.** Chapter numbering below is the current one: the old
ch4+ch5 are the merged `chapters/04_learning_and_results.tex`, sim-to-real is
ch5, the conclusion is ch6, and the geographic simulator is Appendix B.

| | count |
|---|---|
| Closed | 28 mechanical annotations + 16 checklist items |
| Open | **3 from Amir, 16 own** |

---

## A. Amir's comments

### Open (3)

| # | Where | What he asks | Notes |
|---|---|---|---|
| **A9** | ch5 §Command Latency and Actuation Dynamics, ch6 | Rejects the actuator-delay premise: such delay is not considered even in high-speed vehicle control, and the training data contains none to compensate. Suggests modeling max acceleration/deceleration instead | **Answered but not yet accepted by him.** 2026-08-07: reframed from "actuator delay" to measured command latency, split into a $0.09$\,s digital transport dead time (CAN plus the discrete loop, about one $10$\,Hz policy tick) and a $0.11$\,s first-order actuator lag, from a bidirectional step-response test on the stationary vehicle. The text now concedes his point for production vehicles and bounds the claim to a ROS2 research platform, and reports the measured $1.18$\,rad/s steering rate limit. Remaining: the effect of the compensator on closed-loop tracking is still unmeasured |
| **A7** | ch5 §Vehicle Scale, ch6 | Discuss transferability of the trained network to a different tractor/trailer, especially **different trailer sizes** | Partially answered: §Vehicle Scale now states the model is re-parameterized and a fresh policy trained rather than transferring across scale. That says what was done, not how transferable the network is. No trailer-size sensitivity anywhere |
| **A4** | end of ch2, ch3 | A summary at the end of each chapter | ch4, ch5, and ch6 have one; **ch2 and ch3 do not** |

### Closed

| # | Resolution |
|---|---|
| A1 | Reference added for "comes with guarantees" (`mayne2000constrained`, `samson1995chained`, `lavalle2006planning`) |
| A2 | 2026-08-07: 12 single-paragraph subsections in ch2 and 3 in ch5 converted to unnumbered bold `\paragraph` headings. ch2 now has zero numbered subsections |
| A3 | Figure 3.1 redrawn larger with labeled axles and visible tires. Justification added: the reference tractor's hitch sits 0.25 m ahead of the rear axle, under a tenth of the 2.95 m wheelbase, and $L_h$ stays a free parameter. **Hitch → fifth wheel rename declined**: the thesis keeps "hitch" |
| E6 (email) | 2026-08-07: figure 3.2 shows the simulated scene the BEV crop comes from, and figure 5.4 shows the RViz deployment view with the occupancy image the policy consumes. With the AgileX photo and the architecture diagram, all three of Amir's requested figures exist |
| A5 | 2026-08-07: dimension phrasing changed to input counts ("8 input vector", "2 input vector", "8 input state") |
| A8 | 2026-08-07: ch4 now states that reverse instability means no forward-tuned component can be assumed to carry over, and that feedback enters through the observation (the articulation angle) and the reward rather than through memory in the network. Future Work gained a history-stacked / recurrent policy entry |
| A6 | 2026-08-07: "cheap direct measurements" → "available on the vehicle as measurements or estimates from on-board sensing" |
| p20, p34 ×2, p40, p41 ×3, p42 | Closed in the 2026-08-05 mechanical pass (passive voice needed no action, lane-following title, run-in heading periods) |

---

## B. Own notes

### Open (16)

All are marked in place with `% REVIEW:<id>`:
`grep -n "% REVIEW" chapters/04_learning_and_results.tex chapters/sections/scalable_simulation.tex`

**Prose and claims — ch4** (14): **B3** stale "reverse image run set aside" clause · **B5** report forward tracking detail and state multiplicative TD3 won both directions · **B8** stop action is always enabled now, so the protocol sentence and its justification need rewriting · **B9** obstacle slots are not fairly called a "high-dimensional input" · **B10** define dynamic difficulty explicitly · **B12** reword "The first and most direct decouples…" · **B15** the obstacle task does not "require" the image · **B16** softened claim about representation fixes · **B17** record the freeze-then-unfreeze procedure · **B18** is "distillation" the right term · **B19** rewrite the obstacle-blindness sentence · **B20** cleaner explanation of the reverse off-path error · **B27** shorten the benchmark table captions (still 156 and 233 characters) · **B31** add a figure of the hidden avoidance path

**Scalable simulation section** (3): **B4** exact throughput numbers and a BEV row · **B14** clearer LiDAR ray-march explanation · **B25** cut the hand-written GPU-resident TD3 sentence

**Also open:** **B6** the 8-input narrative arc (stable 8-input state → perception destabilizes → minimal 2-input state plus perception) is still not written out · the "32 parallel environments" figure needs confirming against the training script's 256 default · **B33** figure 5.3 (RL bridge dataflow) still has overlapping labels and arrows in the top-left

### Closed

| # | Resolution |
|---|---|
| B1 | Results-table import fixed; no placeholder cells remain |
| B2 | Curvature $R^2$ now 0.997 for both samples (was 0.699/0.425 and 0.558/0.277) |
| B7 | Verified against `tractor_trailer_rl_cupy`: the code renders a signed-distance field, the thesis was right and the note was wrong. Terminology unified to SDF-BEV |
| B11 | ch2 "equal footing" wording |
| B13 | Closed with A6 |
| B21 | HPO section replaced by a real training-configuration table (no HPO framework exists in the code) |
| B22, B23 | Safe adaptation and failure replay cut outright; the dependent `\cref` removed cleanly |
| B24 | Evaluation protocol stated once, in the merged chapter |
| B26 | Fair-pool paragraph cut |
| B28 | Geographic simulation moved to Appendix B |
| B29 | Section renamed to Lane Following, per Amir's answer |
| B30 | Figure 3.1 redrawn |
| B34 | No action needed; Amir answered it |
| `\pending` ×2 | Both resolved; markers remain only in two unused table skeletons |
