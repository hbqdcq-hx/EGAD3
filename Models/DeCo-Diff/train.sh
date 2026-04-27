torchrun train_DeCo_Diff.py \
            --dataset mvtec \ 
            --data-dir /mnt/T38/bioinf/xjj/Datasets/MVTec-AD \
            --model-size UNet_L \
            --object-category all  \
            --image-size 288 \
            --center-size 256 \
            --center-crop True