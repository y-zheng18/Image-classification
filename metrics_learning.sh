python metrics_learning.py --data_type fine --gpu 0 \
--model wide_resnet --layers 4 4 4 --wide_factor 20 \
--lambda_triplet 1 --lambda_rec 1 --lambda_cls 1 \
--bs 128 --lr_policy multi-step --lr_steps 2 4 6 --epoch 10 --embedding_size 512 --use_triplet --lr 0.005 \
--epoch_resume 0 --load_model_dir chkpoints/wide_resnet_SGD_fine.pth --use_all_data --fix_backbone
#--load_autoencoder_dir chkpoints/autoencoder_512.pth \
#--load_optim_dir chkpoints/optim_autoencoder_512.pth --use_all_data