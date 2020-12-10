python train.py --gpu 0 --bs 32 --lr_policy multi-step --lr_steps 15 30 50 --epoch 100 --data_type coarse \
--lr 0.001 --model wideresnet_pretrained --pretrained # --use_all_data