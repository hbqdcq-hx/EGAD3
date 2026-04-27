#BASE_PATH -- the base directory of mvtec
#i -- the gpu id used for evaluation
#python test_dsr.py $i $BASE_PATH DSR
#MVTec-AD/good_dataset
#python test_dsr_numpy.py 2 /mnt/T4_2/xjj/FirstWorkData1/ DSR        
python test_dsr_heatmap.py 0 /mnt/T38/bioinf/xjj/Datasets/MVTec-AD/ DSR   #/mnt/T38/bioinf/xjj/Datasets/MVTec-AD/