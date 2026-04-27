CUDA_VISIBLE_DEVICES=3 python evaluation_DeCo_Diff_heatmap.py \
            --dataset mvtec \
            --data-dir /mnt/T4_2/xjj/FirstWorkData1 \
            --model-size UNet_L \
            --object-category all  \
            --anomaly-class all  \
            --image-size 288 \
            --center-size 256 \
            --center-crop True \
            --model-path /mnt/T38/bioinf/xjj/CheckPoints/DeCo-Diff_CP/MVTEC-AD-model.pt

            #/mnt/T4_2/xjj/FirstWorkData1
            #/mnt/T38/bioinf/xjj/Datasets/MVTec-AD