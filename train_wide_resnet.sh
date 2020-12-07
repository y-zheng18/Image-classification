python train.py --model wide_resnet --gpu 1 --bs 256 --lr_policy multi-step --lr_steps 60 120 160 --data_type fine \
--layers 6 6 6 --wide_factor 10 --gpu_ids 0 1