python train.py --model wide_resnet --gpu 1 --bs 64 --lr_policy multi-step --lr_steps 60 120 160 --data_type fine \
--layers 4 4 4 --wide_factor 20 --load_model_dir chkpoints/wide_resnet_fine_28_20.pth --load_optim_dir chkpoints/optim_wide_resnet_SGD_fine.pth --use_all_data \
--lr 0.004 --epoch_resume 151