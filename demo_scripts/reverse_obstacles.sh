#!/bin/bash
source ~/Ben/Thesis/e2e_rl/venv/bin/activate
cd ..
python run_model.py --scenario reverse_obs --obs lidar --lidar_beams 24 --model ./models/ObstacleAvoidance/Reverse/Best/feb3.zip --episodes 5 --render
