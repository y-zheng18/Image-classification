python train.py --model wide_resnet --gpu 1 --bs 128 --lr_policy multi-step --lr_steps 60 120 160 --epoch 200 --data_type fine \
--layers 6 6 6 --wide_factor 10 \
--lr 0.1 --dropout_rate 0.3 # --use_all_data --epoch_resume 140 --load_model_dir chkpoints/wide_resnet_fine_28_20.pth