#!/bin/bash

cd ..
# python train.py --target tri_plane --experiment tri_plane_recon_plus --lambda_loss_recon 10 --epoch 3 --encoder_path ./checkpoints/tri_plane_recon/tri_plane_encoder_epoch_9\|10_minibatch_9200\|10000.pt
python train.py --target ws --experiment ws_recon --lambda_loss_recon 10
python train.py --target z --experiment z_recon --lambda_loss_recon 10