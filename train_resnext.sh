python train.py --lr 0.1 --epoch 200 --lr_policy multi-step --lr_steps 60 120 160 --dropout_rate 0 --gpu 0 --bs 128 \
--data_type fine --model wide_resnext --layers 6 6 6 --wide_factor 10