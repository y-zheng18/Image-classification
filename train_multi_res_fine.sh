python train.py --lr 0.1 --epoch 240 --lr_policy multi-step --lr_steps 80 160 200 --dropout_rate 0.3 --gpu 0 --bs 128 --model multi-res \
--data_type fine --layers 6 6 6 --wide_factor 10