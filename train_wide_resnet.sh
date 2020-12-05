python train.py --model wide_resnet --gpu 0 --bs 128 --lr_policy multi-step --lr_steps 60 120 160 --data_type fine \
--layers 6 6 6