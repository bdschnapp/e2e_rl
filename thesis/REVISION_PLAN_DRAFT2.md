# Draft-2 revision plan — status

Consolidates Amir's email (2026-08), the five structural decisions from the
2026-08-05 discussion, and items found while auditing the sources. Line-level
comments live in `AMIR_REVIEW_CHECKLIST.md` and are referenced as `[A*]`/`[B*]`.

**Updated 2026-08-07 against the refactored thesis.** Current structure: ch1
Introduction, ch2 Literature Review, ch3 Modeling and Simulation (now including
§3.6 Scalable Simulation), ch4 Learning Framework and Experimental Results
(merged), ch5 Sim-to-Real Deployment, ch6 Conclusion, Appendix A classical
controllers, Appendix B geographic simulator. 120 pages, building clean with
zero undefined references.

---

## Still open

### Substantive

| # | Item | State |
|---|---|---|
| **E3 / A9** | **Actuator delay.** Amir rejects the premise and proposes acceleration/deceleration limits instead | **Answered 2026-08-07** by reframing to measured command latency: a $0.09$\,s digital transport dead time plus a $0.11$\,s actuator lag, measured by step response on the platform, with the production-vehicle case conceded explicitly. Remaining risk: he has not seen the reframing, and the compensator's effect on closed-loop tracking is still unmeasured, which ch6 Limitations admits |
| **E2 / A7 / N1** | **Transferability** across tractor/trailer configurations, especially trailer size | **Answered as future work 2026-08-07**: a ch6 Limitations entry states that the policies are trained against a single wheelbase and trailer length and that sensitivity to a different configuration is uncharacterized, and a Future Work entry proposes the parameter sweep. No experiment run, by decision |
| **E1 / A4** | **Chapter summaries** | ch4, ch5, ch6 have them; **ch2 and ch3 do not** |


### Editorial

- **16 open `% REVIEW` items** in ch4 and the scalable-simulation section, listed in `AMIR_REVIEW_CHECKLIST.md`. The heaviest are B8 (the stop action is always enabled now, so the evaluation-protocol sentence is wrong), B17 (record the freeze-then-unfreeze procedure), and B5 (report forward tracking detail).
- **B33**: figure 5.3, the RL bridge dataflow, still has overlapping labels and arrows in the top-left.
- **B31**: no figure of the hidden avoidance path.
- **E5 residual duplication**: ch5 Overview ↔ ch5 Summary (5 shared 7-grams), and the ch5 Trailer description ↔ the geographic-simulator appendix (6).
- **N12**: a short "how each comment was addressed" note to send with the next draft. The two checklists here are the raw material.

---

## Done

### Structural (the 2026-08-05 decisions)

- **S1 Merge ch4 + ch5** — adopted as `chapters/04_learning_and_results.tex`. The 92 shared 7-grams between the old chapters are gone; a later simplification pass removed a further 547 words and 4 unreferenced display equations.
- **S2 Scalable simulation to ch3** — now §3.6, `\input` from `chapters/03_modeling_simulation_problem.tex`.
- **S3 Smith predictor** — implementation section removed from ch5. **The claim itself is still open, see E3/A9 above.**
- **S4 Geographic simulation** — moved to Appendix B.
- **S5 Images and real-robot results** — every referenced image resolves, including the AgileX platform photo. Deployment Results is written, with Trial Protocol, Tracking Performance, and Articulation Limit Cycle subsections and two new figures.

### Amir's email

- **E4** short sections — 12 subsections in ch2 and 3 in ch5 converted to unnumbered bold headings (2026-08-07); ch2 now has zero numbered subsections.
- **E5** repeated material — resolved by the merge, apart from the two residuals noted above.
- **E6** figures — AgileX photo (fig 5.1), network architecture (fig 4.1), the simulated scene fed to the network (fig 3.2), and the RViz deployment view showing the policy's image input (fig 5.4).
- **E7** remaining sections — ch1 and ch6 are real chapters in the build (they were placeholder stubs in the PDF Amir read), ch6 Limitations and Future Work are written, and Deployment Results is populated.

### Audit items

- **N2** ch2 consolidation — done via E4, following Amir's literal instruction (unnumber and bold) rather than merging subsections.
- **N4** front-matter sweep — `SDF` in the abbreviation list is correct and still used; `$L_h$` in the symbol list is correct and now explained in ch3.
- **N5** data integrity first — table-import failure fixed, curvature $R^2$ corrected, both `\pending` markers resolved.
- **N6** empty appendices — dropped; only the two written appendices remain.
- **N8** ch7 conclusion — written.
- **N3** placeholder mechanism — the `\reviewplaceholder` macro and the commented-out placeholder appendix includes are removed from `main.tex`; the `chapters/placeholder/` files are yours to delete.
- **N11** length — 117 pages. The merge and the simplification pass were roughly length-neutral; the reductions came from the structural cuts.

---

## Decisions still yours

None outstanding. Settled 2026-08-07: no further latency justification will be added (the
measured numbers plus the compensator are enough), and the placeholder mechanism is retired.

Settled: the actuator-delay framing (measured command latency), transferability (future work, no
experiment), the merged-chapter shape (one interleaved chapter),
the cut candidates (safe adaptation and failure replay both cut), the hyperparameter table
(inline in ch4, no appendix), and terminology (**hitch** stays hitch; "lane following" over
"lane driving").
