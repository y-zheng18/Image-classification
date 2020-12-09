python train.py --gpu 0 --bs 128 --lr_policy multi-step --lr_steps 15 30 50 --epoch 100 --data_type coarse \
--lr 0.01 --model wideresnet_pretrained # --use_all_data