python train.py --gpu 0 --bs 64 --lr_policy multi-step --lr_steps 40 60 80 --epoch 100 --data_type fine \
--use_all_data \
--lr 0.001 --model resnet_pretrained