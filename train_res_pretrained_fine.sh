python train.py --gpu 2 --bs 32 --lr_policy multi-step --lr_steps 10 20 --epoch 30 --data_type fine \
--lr 0.001 --model resnet_pretrained --pretrained # --use_all_data