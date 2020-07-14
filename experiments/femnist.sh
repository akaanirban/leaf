#!/bin/bash

# Does not work with --num-epochs i.e. local epochs too low
python main_pytorch.py -dataset 'femnist' -model 'cnn_pytorch' --num-epochs 10 -t small --clients-per-round 50 --eval-every 2 -lr 0.1 --batch-size 50
