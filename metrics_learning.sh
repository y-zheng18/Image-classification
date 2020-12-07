python metrics_learning.py --data_type fine --gpu 0 \
--load_model_dir chkpoints/wide_resnet_fine_40_10.pth --model wide_resnet --layers 6 6 6 \
--lambda_triplet 10 --lambda_rec 10 --lambda_cls 1 \
--bs 128 --lr_policy multi-step --lr_steps 3 8 15 --epoch 20 --embedding_size 256 --use_tripletL2