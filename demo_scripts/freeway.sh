#!/bin/bash
source ~/Ben/Thesis/e2e_rl/venv/bin/activate
cd ..
python run_model.py --model ./models/LineFollowing/Forward/normalized/norm5.zip --episodes 5 --render
