#多类设置训练
#CUDA_VISIBLE_DEVICES=1 python dinomaly_mvtec_uni.py --data_path /mnt/T38/bioinf/xjj/Datasets/MVTec-AD  
#多类设置测试
CUDA_VISIBLE_DEVICES=1 python test_mvtec_uni_heatmap.py --data_path /mnt/T4_2/xjj/FirstWorkData1  --model_path ./ckpts/mvtec_uni/model.pth  

#/mnt/T4_2/xjj/FirstWorkData1
#单类训练
#python dinomaly_mvtec_sep.py --data_path /mnt/T38/bioinf/xjj/Datasets/MVTec-AD  