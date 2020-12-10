python train.py --model wide_resnet --gpu 0 --bs 128 --lr_policy multi-step --lr_steps 60 120 160 --data_type coarse \
--layers 6 6 6 --wide_factor 10 \
--lr 0.1 --dropout_rate 0.3