python visualize.py --data_type coarse --gpu 1 \
--model wide_resnet --layers 6 6 6 --wide_factor 10 \
--bs 128 --embedding_size 256  --lr 0.01 \
--load_model_dir chkpoints/wide_resnet_SGD_coarse.pth # --use_triplet --use_all_data # --fix_backbone
#--load_autoencoder_dir chkpoints/autoencoder_512.pth \
#--load_optim_dir chkpoints/optim_autoencoder_512.pth --use_all_data