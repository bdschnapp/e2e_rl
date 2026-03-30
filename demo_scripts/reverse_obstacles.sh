#!/bin/bash
source ~/Ben/Thesis/e2e_rl/venv/bin/activate
cd ..
python run_model.py --model ./models/ObstacleAvoidance/Reverse/Best/feb3.zip --episodes 5 --render --obstacles --reverse
