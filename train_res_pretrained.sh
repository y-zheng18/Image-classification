python train.py --gpu 0 --bs 128 --lr_policy multi-step --lr_steps 60 120 160 --epoch 200 --data_type coarse \
--lr 0.001 --model resnet_pretrained # --use_all_data