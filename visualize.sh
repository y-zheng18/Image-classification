python visualize.py --data_type coarse --gpu 0 \
--model wideresnet_pretrained --layers 2 2 2 2 --wide_factor 10 \
--bs 128 \
--load_model_dir chkpoints/wideresnet_pretrained_SGD_coarse.pth --pretrained # --use_triplet --use_all_data # --fix_backbone
#--load_autoencoder_dir chkpoints/autoencoder_512.pth \
#--load_optim_dir chkpoints/optim_autoencoder_512.pth --use_all_data