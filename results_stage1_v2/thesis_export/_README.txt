Thesis import — Stage-1 GPU ablation study
============================================

Source: /home/ben/Ben/Thesis/tractor_trailer_rl_cupy/results_stage1_v2  (12 completed runs)

[forward] reward table @ algo=ppo   |   algorithm table @ reward=guided

To import into the thesis (from thesis/ dir), \input the tables, e.g.:
    \input{tables/reward_ablation_forward.tex}
    \input{tables/algorithm_comparison_forward.tex}

Copy the generated files in with (run yourself when ready):
    cp results_stage1_v2/thesis_export/*.tex  <thesis>/tables/
    cp results_stage1_v2/thesis_export/*.pdf  <thesis>/figures/
