#!/bin/bash
source ~/Ben/Thesis/e2e_rl/venv/bin/activate
cd ..
python run_model.py --scenario forward_obs --obs lidar --model ./models/ObstacleAvoidance/Forward/Best/feb3.zip --episodes 5 --render --lidar_beams 24
