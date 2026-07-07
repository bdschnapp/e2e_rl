# Removed from thesis LaTeX: Waterloo MASc theses rigor-calibration table

This content was in `chapters/02_literature_review.tex` (§Benchmarking Against Recent
Waterloo Theses) as scaffolding to calibrate the expected level of rigor. Per the
author's decision it was **removed from the manuscript** (the rigor of other Waterloo
theses should not be reported in the draft) and preserved here only for reference.

Original framing: "To contextualise the expected level of rigor, ... summarises recent
MASc theses from the Mechatronics domain at the University of Waterloo."

| Author (Year) | Topic | Methodology | Rigor Highlights |
|---|---|---|---|
| Joseph (2025) | Gaze-enabled grasping assistance | ROS2, Kinova Arm, Meta Quest Pro | User study with 30 participants; workload and performance metrics |
| Thibault (2022) | Humanoid robotic manipulation | EUROBENCH, REEM-C robot | Defined "loco-manipulation"; new manipulability–stability metrics |
| Hart (2023) | Gait detection via machine learning | Wearable sensors, ternary classification | 6 classical + 1 Transformer model; grid-search hyperparameter optimisation |
| Wei (2021) | Path following for manipulators | Transpose Jacobian control | Free of inverse transformations; validated on 2-DoF and 4-DoF robots |
| Bickford (2019) | Autonomous bicycle stabilisation | Linear control, Whipple model | Multi-loop architecture; "virtual paths" for high-error recovery |

Closing sentiment (also removed): common themes were strong mathematical modelling
foundations and comprehensive validation datasets; a simple "it works" result is
insufficient — the work must quantify stability limits, evaluate performance across
varied paths, and compare against classical benchmarks. (This rigor expectation is now
expressed in the thesis without naming other theses.)
