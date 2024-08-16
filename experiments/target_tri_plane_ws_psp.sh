cd ..
# python train.py --target ws \
#                 --experiment ws_psp \
#                 --batch 16 \
#                 --save_minibatch_freq 200
python train.py --target tri_plane \
                --epoch 10 \
                --batch 16 \
                --experiment tri_plane_psp_plus_v4 \
                --save_minibatch_freq 300 \
                --encoder_path ./checkpoints/tri_plane_psp_plus_v3/tri_plane_encoder_epoch_30\|30_minibatch_625\|625.pt

python train.py --target ws \
                --epoch 10 \
                --batch 16 \
                --experiment ws_psp_v2 \
                --save_minibatch_freq 300 \
                --encoder_path ./checkpoints/ws_psp/ws_encoder_epoch_10\|10_minibatch_625\|625.pt