python train.py --lr 0.1 --epoch 200 --data_type fine --lr_policy multi-step --lr_steps 60 120 160 --dropout_rate 0 --gpu 1 \
--use_triplet --lambda_triplet 10 --triplet_warm_up -1 --bs 128