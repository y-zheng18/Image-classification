python train.py --model wide_resnet --gpu 0 --bs 64 --lr_policy multi-step --lr_steps 60 120 160 --data_type coarse \
--layers 4 4 4 --wide_factor 20 \
--lr 0.1 --dropout_rate 0.3