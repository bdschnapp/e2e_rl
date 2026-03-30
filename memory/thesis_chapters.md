---
name: Thesis Chapter References
description: Three-level escalation for thesis context. Always use summaries before raw .tex files to reduce token usage. Read THESIS_CONTEXT.md first for any thesis question.
type: reference
---

## Token-Efficient Escalation

**Level 1** — `THESIS_CONTEXT.md` (root): scope, notation, vehicle params, baselines, chapter map. Read for any thesis question.

**Level 2** — `thesis/summaries/ch0N.md`: per-chapter bullet summaries (~20 lines each). Read before editing a chapter.

**Level 3** — `thesis/chapters/0N_*.tex`: raw LaTeX. Read ONLY when about to edit that specific file.

## Chapter Summary Paths
| Chapter | Summary |
|---|---|
| Ch1 Introduction | `thesis/summaries/ch01.md` |
| Ch2 Literature Review | `thesis/summaries/ch02.md` |
| Ch3 Modeling | `thesis/summaries/ch03.md` |
| Ch4 Perception | `thesis/summaries/ch04.md` |
| Ch5 RL Framework | `thesis/summaries/ch05.md` |
| Ch6 Simulation | `thesis/summaries/ch06.md` |
| Ch7 Results | `thesis/summaries/ch07.md` |
| Ch8 Conclusion | `thesis/summaries/ch08.md` |

**Pending code changes needing thesis updates**: `thesis/pending_updates.md`
