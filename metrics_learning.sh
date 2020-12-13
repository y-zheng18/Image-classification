python metrics_learning.py --data_type coarse --gpu 1 \
--model wide_resnet --layers 6 6 6 --wide_factor 10 \
--lambda_triplet 1 --lambda_rec 0 --lambda_cls 1 \
--bs 64 --lr_policy multi-step --lr_steps 50 80 --epoch 100 --embedding_size 32 --use_triplet --lr 0.01 \
--epoch_resume 0 --load_model_dir chkpoints/wide_resnet_SGD_coarse.pth --fix_backbone #--use_all_data # --fix_backbone
#--load_autoencoder_dir chkpoints/autoencoder_512.pth \
#--load_optim_dir chkpoints/optim_autoencoder_512.pth --use_all_data