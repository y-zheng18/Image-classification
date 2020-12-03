python train.py --lr 0.1 --epoch 200 --data_type coarse --lr_policy multi-step --lr_steps 60 120 160 --dropout_rate 0 --gpu 3 \
--use_triplet --lambda_triplet 0.1