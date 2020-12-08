python metrics_learning.py --data_type fine --gpu 0 \
--load_model_dir chkpoints/wide_resnet_fine_40_10.pth --model wide_resnet --layers 6 6 6 \
--lambda_triplet 1 --lambda_rec 1 --lambda_cls 1 \
--bs 64 --lr_policy multi-step --lr_steps 20 40 60 --epoch 80 --embedding_size 512 --use_triplet --lr 0.005