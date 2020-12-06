python train.py --model wide_resnet --gpu 1 --bs 64 --lr_policy multi-step --lr_steps 60 120 160 --data_type fine \
--layers 6 6 6 --use_triplet --lambda_triplet 10 --triplet_warm_up -1 --lr 0.1