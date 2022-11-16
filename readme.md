Hello!

Below you can find an outline of how to reproduce my (unclean version) solution for the <ai4code> competition.
If you run into any trouble with the setup/code or have any question please contact me at <hydantess@163.com>

## ARCHIVE CONTENTS
input : data of competitions 

src: code to rebuild models from scratch 

## HARDWARE: (The following specs were used to create the original solution)
Inference: Kaggle notebook(turn on gpu) 

Training:
GPU: 1 * A100 80G 
CPU: 32G 64core

## SOFTWARE:
Inference: Kaggle notebook (turn on gpu)

Training: requirements.txt

## Configuration files & SETTINGS.json
see src/parameter.py

## MODEL TRAIN&PREDICT
1.Training:

cd src

python train_mlm.py --model_name deberta-v3-large --base_epoch 15 --batch_size 7 --learning_rate 5e-6 --max_length 1024

python train.py --model deberta-v3-large --base_epoch 10 --batch_size 5 --lr 5e-6 --seq_length 2048 --max_grad_norm 1.0 --folds 0  

2.Inference:

Please run notebook: https://www.kaggle.com/code/hydantess/ai4code0729?scriptVersionId=103060429

