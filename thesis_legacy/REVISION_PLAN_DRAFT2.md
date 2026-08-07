# Draft-2 revision plan

Consolidates Amir's email (2026-08), the five structural decisions from the
2026-08-05 discussion, and items found while auditing the sources. Line-level
comments from the annotated PDF live separately in
`AMIR_REVIEW_CHECKLIST.md` (45 open) and are cross-referenced as `[B*]`/`[A*]`.

Status legend: **must** = draft is incomplete without it, **should** = Amir asked
for it, **propose** = my recommendation, needs your call.

---

## 0. What the audit found

| Measure | Current state |
|---|---|
| ch4 ↔ ch5 duplication | **92 shared 7-word sequences**, 12 near-verbatim claim pairs. Every other chapter pair is ≤ 6, so this is the only significant duplication in the thesis |
| One-paragraph sections | ch2: **15 of 21** sections/subsections are a single paragraph. ch6: 13, six of them in §6.4 alone |
| Figures | **18 total**: ch1 = 0, ch2 = 0, ch3 = 1, ch4 = 2, ch5 = 4, ch6 = 11 |
| Missing image files | **5** (`figures/hardware/` does not exist at all: `platform.jpg`, `sensor_suite.jpg`, `hitch_lidar_fit.png`, `deployment_trial.jpg`; plus `figures/gazebo/gazebo_dc_world.png`) |
| Empty figure dirs | `figures/ch1 … ch5`, `reward_design`, `system_architecture`, `trailer_kinematics` (planned figures never made) |
| Unwritten sections | ch6 §6.7 Deployment Results (TODO block only), ch7 Limitations (8 words), ch7 Future Work (7 words) |
| Unwritten appendices | `appendix_hyperparameters.tex` (10 words), `appendix_additional_plots.tex` (11 words), `appendix_ros2_stack.tex` (16 words, **not included in main.tex at all**) |
| `main.tex` includes | **ch1 and ch7 are the placeholder stubs**, as are two appendices. Amir reviewed a build where the introduction and conclusion read "[Section content omitted for review]" |
| Unresolved data | 5 `\pending{}` markers, 2 all-placeholder tables (`obstacle_results.tex`, `failure_replay_results.tex`), table-import failures `[B1–B3]` |

---

## 1. Amir's email comments

| # | Ask | Action | Status |
|---|---|---|---|
| E1 | Summary at the end of each chapter | Only ch6 has one. Add to ch1(?), ch2, ch3, merged ch4, sim-to-real, and keep ch6's. Write them last, after the restructure settles | must |
| E2 | Discuss transferability across tractor/trailer configurations, especially trailer size; you already mention retraining for AgileX in ch6 | See **N1** for a concrete proposal. Anchor is ch6 §6.4.1 Vehicle Scale and Parameterization | must |
| E3 | Actuator delay is not a real concern; if you mean throttle/brake, model max accel/decel instead; and there is no delay in the training data either | Structural decision **S3** below. Note his alternative is a *substantive* suggestion, not just a cut: adding accel/decel limits to the vehicle model would be a legitimate fidelity improvement in ch3 | should |
| E4 | Many one-paragraph short sections: unnumber and bold, or combine | Hits ch2 hardest (15 of 21) and ch6 §6.4 (6 of 6). See **N2**. The `\paragraph` style is already redefined as a bold display heading, so the mechanism exists | should |
| E5 | Repeated material across sections; remove to shorten | The measured duplication is almost entirely ch4 ↔ ch5, which **S1** resolves. Two smaller ones found: ch6 Overview ↔ Summary, and ch6 §6.2.2 Trailer ↔ §6.5.1 Articulated Vehicle Model (the trailer geometry is described twice) | should |
| E6 | Add diagrams/photos: AgileX photo, the road fed to the network, the network structure | Three specific figures. `figures/pygame/sim_bev.png` and `tikz_recreation.png` already exist and are **unused** and may serve the "road fed to the network" ask. `figures/tikz/learning_architecture.tex` exists in ch4 and may already be the network diagram, but check it shows the observation split, encoder, and actor/critic heads clearly. AgileX photo must be taken/found | should |
| E7 | "Apply the changes and add the remaining sections to complete your thesis" | This is partly an artifact of the review build: ch1/ch7 exist but were stubbed. Flip `main.tex` to the real files, then finish ch7 Limitations + Future Work, ch6 Deployment Results, and the two empty appendices | must |

---

## 2. Structural decisions from the 2026-08-05 discussion

> **Status 2026-08-05:** S1 and S2 are drafted and building. `chapters/04_learning_and_results.tex`
> is the merged chapter (drop-in replacement for both includes) and
> `chapters/sections/scalable_simulation.tex` is the extracted §4.9 ready to `\input` at the
> end of ch3. The originals are untouched. Verified via `main_merged_test.tex`: 0 undefined
> and 0 multiply-defined references, all 18 figures and every table input preserved, and the
> moved section numbers as §3.6. Folded in along the way: B24 (protocol merged), B26 (fair-pool
> paragraph cut), B21 (HPO replaced by a real configuration table), B7 (SDF terminology).
> B22 and B23 are now discussion-only paragraphs marked `% REVIEW:CUT-CANDIDATE`.
> A simplification pass then removed 547 words (9117 → 8570 prose words, about 6%) and 4
> unreferenced display equations, concentrated in the framing and design prose: the failure-replay
> walkthrough, the Routes A/B caveat repeated in §7.4, the three restatements of the
> not-a-superiority hedge in §9, and the §10 description of a failure-mode analysis the chapter
> never presents. **Page count is 117, the same as the current thesis**, so neither the merge nor
> the simplification is where length comes from; see N11 and the structural cuts S3/S4.

### S1. Merge ch4 and ch5 to minimize duplication
Evidence: 92 shared 7-grams; the evaluation protocol, the instability hypothesis,
the graded-SDF rationale, the distillation recipe, and the windowed-difficulty
argument are each stated twice in near-identical words.

Proposed structure for the merged chapter (design immediately followed by its
evidence, so nothing is set up twice):

1. Learning framework overview; TD3; action and observation interface
2. **Evaluation protocol** (old §4.8 + §5.1 merged; kills the worst duplication) `[B24]`
3. Reward formulation → algorithm and reward selection results
4. Observation spaces (state / LiDAR / BEV) → observation-space ablation results
5. Vision-based policies: instability hypothesis → routes A/B/C → causal ladder → geometry distillation results
6. Obstacle navigation: reward design, hidden path, stop gate → forward and reverse results
7. Comparison with classical controllers (+ information parity)
8. **Transferability** (new, E2/N1)
9. Failure modes and deployment relevance
10. Summary (E1)

Cuts folded in: HPO → appendix `[B21]`, safe adaptation shortened `[B22]`,
failure replay shortened or appendix `[B23]`, hand-written TD3 removed `[B25]`,
fair-pool paragraph removed `[B26]`.

Cross-reference cost, measured: **20** `\cref{ch:results}` calls inside ch4 and
1 `\cref{ch:learning_framework}` in ch5 become self-references and need prose
edits ("as reported below", or a section-level `\cref`). Keep **both** chapter
labels on the merged chapter so no other file breaks; 37 section/table labels in
the two files stay valid if the `\label` lines travel with their content.

Risk: the merged chapter will be large (~35 pages before cuts). If that reads
badly, the fallback is a theme split rather than a design/results split:
"Learning Framework and Perception" + "Obstacle Navigation and Benchmarking",
which preserves the no-duplication property.

### S2. Move Scalable Simulation to the end of ch3
Old §4.9 (`sec:batched_sim`, 3 subsections: batched design, numerical parity,
throughput) becomes §3.6. Keep the `\label` so the 4 inbound `\cref`s survive.
Follow-ups: retitle ch3 (it is currently "Modeling, Simulation, and Problem
Formulation", so the existing title still fits, but consider "…and Scalable
Simulation" to advertise the contribution); ch3 then needs its own summary (E1);
the throughput discussion's forward reference to the results chapter becomes a
forward reference to the merged chapter, which is still valid; and `[B4]`
(exact throughput numbers + a BEV row) should be done at the same time.

### S3. Smith predictor → appendix or removal
Amir does not accept the premise (E3), so the framing has to change either way.
Everything that must move together, since the claim is threaded through the thesis:

| Location | Current content | Action |
|---|---|---|
| ch6 §6.3.7 | Smith predictor implementation, 5 paragraphs + `figures/tikz/smith_predictor.tex` | → appendix, or delete |
| ch6 §6.4.2 Actuation Dynamics | Names delay as a primary sim-to-real divergence | Rewrite; this is where Amir's accel/decel-limit alternative belongs |
| ch6 Summary | Calls the Smith predictor one of three "principal engineering contributions" | Must be rewritten regardless of choice |
| ch2 §2.3.7 Learning under Actuation Delay | Literature subsection justifying the whole thread | Shrink to a sentence or fold into §2.3, per E4 |
| ch1 Contributions / abstract | Check for delay claims | Sweep |

Recommendation: keep a short honest paragraph in the sim-to-real gap analysis
("a first-order actuator lag was observed on the platform and compensated with a
Smith predictor; details in appendix X"), move the implementation to an appendix,
and add acceleration/deceleration limits to the ch3 vehicle model as the
fidelity improvement Amir actually suggested. That answers him without
discarding working code. `[A9]`

### S4. Shorten High-Fidelity Geographic Simulation
Your vote: shorten, not defer, because the tractor-trailer photo is still needed
for the RVIZ/simulation discussion. Current: §6.5 + §6.5.1, ~710 words, 4
figures, one of which (`gazebo_dc_world.png`) is **missing**. Note §6.5.1
Articulated Vehicle Model duplicates the §6.2.2 Trailer description (E5), so
cutting it removes duplication and length in one move. Target: one section of
2–3 paragraphs keeping the Blender/Esri reconstruction figure and the
`electrans_model.png` figure, with the workflow detail dropped or appended.
`[B28]`

### S5. Import missing images; import real-robot results
Two distinct jobs:
- **Images**: 5 missing files above. `\safegraphic` degrades to a "Figure unavailable" box, so these are silently shipping as empty frames in the review PDF. `[B30–B33]` cover the *broken* ones (fig 3.1 clarity, fig 6.4 readability, fig 6.6 arrow overlap).
- **Real-robot results**: ch6 §6.7 is an empty TODO block that also asks for the metric-capture methodology (which rosbag topics, window selection, how statistics are computed) before any numbers. Without this section the deployment chapter has no evidence, so treat it as the highest-value writing task after the data fixes.

---

## 3. Additional items I propose

### N1. Make transferability an experiment, not just a discussion  *(propose, high value)*
E2 is Amir's most substantive ask and currently has no answer anywhere in the
thesis. The cheapest credible answer uses infrastructure you already have: the
batched GPU simulator makes a **trailer-length sweep** nearly free. Evaluate a
policy trained at the nominal $L_t$ across a range of trailer lengths (and
optionally wheelbase/width), report completion and CTE degradation, and state the
retraining threshold. That yields one figure, one table, a real answer to "what
happens with a different trailer", and it connects directly to the AgileX
retraining already described in ch6 §6.4.1. Fallback if time is short: a
limitations subsection that states the policy is configuration-specific, explains
why (the observation is in vehicle-relative error coordinates, so geometry enters
through the dynamics rather than the observation), and cites the AgileX
retraining as evidence that retraining is cheap.

### N2. Consolidate ch2 rather than only unnumbering it  *(propose)*
E4 read literally would leave 15 bold run-in headings in ch2, which is not
shorter. Recommended: merge §2.3's six subsections into two (foundations +
applications), merge §2.4's three into one, and convert only what is left to bold
run-ins. `[A2]`

### N3. Regenerate or retire the `chapters/placeholder/` mirrors  *(must)*
These stubs exist to preserve numbering for external-review builds. After S1/S2
they will silently drift out of sync with the real chapters, and `main.tex`
currently points at four of them. Decide now: either regenerate them as the last
step before each review build, or delete the mechanism and share the full PDF.
Either way, **flip `main.tex` to the real ch1/ch7 before the next send** (E7).

### N4. Front-matter sweep after the content changes  *(must)*
`SDF` (Signed-Distance Field) in the abbreviation list is **correct and stays** —
verified against the code, see `[B7]`. The symbol list carries `$L_h$` and "Hitch articulation angle",
both of which change if you accept Amir's hitch → fifth wheel rename `[A3]`. Do
one pass over abbreviations, symbols, and the abstract after the content is
final.

### N5. Do the data fixes before the restructure  *(must, sequencing)*
`[B1–B3]` are table-import failures and `[B2]` may be a missing-data problem. If
the restructure happens first, every moved table has to be re-verified against a
still-broken importer. Fix `scripts/generate_phase_tables.py`, clear the 5
`\pending{}` markers and the 2 placeholder tables, then move things.

### N6. Decide the fate of the two empty appendices  *(must)*
`appendix_hyperparameters.tex` and `appendix_additional_plots.tex` are 10 and 11
words. Under S1 the HPO section `[B21]` needs somewhere to go, which makes the
hyperparameter appendix worth completing rather than deleting. Also
`appendix_ros2_stack.tex` exists, is empty, and is not included anywhere: either
write it (it would be the natural home for the Smith predictor detail under S3)
or delete the file.

### N7. Figure inventory as an explicit task list  *(propose)*
ch1 and ch2 have zero figures and ch5 has four for a 21-page results chapter,
which is what prompted E6. Concrete candidates: a thesis-structure or
contributions diagram in ch1; the hidden avoidance path `[B31]`; the
already-existing but unused `figures/pygame/sim_bev.png` for "the road fed to
the network"; the trailer-length transferability plot (N1). Delete the seven
empty `figures/*` directories or fill them.

### N8. ch7 Conclusion is effectively unwritten  *(must)*
Limitations (8 words) and Future Work (7 words) are empty, and Summary of
Findings is 198 words. This is the other half of E7. Several open items are
natural Future Work entries: reverse pull-forward recovery, recurrent policies
for reverse stability `[A8]`, cross-configuration transfer if N1 stays a
discussion, and the learned dynamics model already flagged as in-progress in ch6.

### N9. Answer A8 in the text, not just in the reply  *(propose)*
Amir suggests network feedback for reverse stability. Ch5 already argues the
forward recipe does not transfer to reverse and fixes it with state-dependent
difficulty. Say that explicitly where he raised it, and add recurrent/feedback
policies to Future Work. Cheap, and it shows the comment was addressed.

### N10. ch6 Overview ↔ Summary overlap  *(propose)*
Five shared 7-grams. With a Summary now required in every chapter (E1), rewrite
the ch6 Overview as scope-setting only and let the Summary carry the claims.

### N11. Recheck length after the net of cuts and additions  *(propose)*
Cuts (HPO, safe adaptation, failure replay, hand-written TD3, geographic sim,
Smith predictor, fair-pool paragraph, ch4/ch5 duplication, ch2 consolidation)
plausibly recover 12–18 pages. Additions (7 chapter summaries, transferability,
deployment results, ch7, new figures) plausibly add 10–15. Expect the thesis to
end up about the same length but with a much higher fraction of it being
content Amir has not already read twice. Worth saying to him explicitly.

### N12. Keep a reviewer-facing response note  *(propose)*
Amir made 17 line comments plus 6 email comments. A short "how each comment was
addressed" note with the next draft makes the second review much faster, and the
two checklists here already contain the raw material.

---

## 4. Suggested order

1. **Data integrity** (N5): fix the importer, clear `\pending`, resolve `[B1–B5]`. `[B7]` is settled: the code uses the SDF-BEV as described, so only the word "graded" needs dropping in ch4 and ch5.
2. **Content corrections** in place: `[B8]` stop-action claim, `[B6]` dimension narrative, `[B9]`, `[B10]`, and the prose rewrites `[B11–B20]`. Doing these before the merge means editing each claim once.
3. **Structure**: S2 (move scalable sim), then S1 (merge), then S3 and S4. One commit each, rebuild between them, and check the LaTeX log for undefined references after each.
4. **Missing content**: ch6 Deployment Results (S5), ch7 (N8), appendices (N6), transferability (N1).
5. **Figures** (E6, N7, S5) and the terminology decisions (`[A3]` fifth wheel, `[B29]` lane driving).
6. **Last**: chapter summaries (E1), front-matter sweep (N4), placeholder/`main.tex` flip (N3), response note (N12).

---

## 5. Decisions I need from you

1. **Merged chapter shape** (S1): one large interleaved chapter, or the two-theme split?
2. **Smith predictor** (S3): appendix + short acknowledgment (my recommendation), or delete outright? And do you want accel/decel limits added to the ch3 model as Amir suggests?
3. **Transferability** (N1): run the trailer-length sweep, or discussion-only?
4. **Terminology** (`[A3]`, `[B29]`): hitch → fifth wheel thesis-wide? "lane following" (Amir's answer) or "lane driving" (your branch name)?
5. **Placeholder mechanism** (N3): keep and regenerate, or retire?
6. **Appendices** (N6): complete the hyperparameter and additional-plots appendices, or drop them and inline what matters?
