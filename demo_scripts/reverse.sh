#!/bin/bash
source ~/Ben/Thesis/e2e_rl/venv/bin/activate
cd ..
python run_model.py --model ./models/LineFollowing/Reverse/normalized/norm7.zip --episodes 5 --render --reverse
