python metrics_learning.py --data_type fine --gpu 0 \
--model wide_resnet --layers 6 6 6 \
--lambda_triplet 1 --lambda_rec 1 --lambda_cls 1 \
--bs 64 --lr_policy multi-step --lr_steps 20 40 60 --epoch 100 --embedding_size 512 --use_triplet --lr 0.0001 \
--epoch_resume 80 --load_autoencoder_dir chkpoints/autoencoder_512.pth \
--load_optim_dir chkpoints/optim_autoencoder_512.pth