
export PYTHONPATH=src
#datapath=/mnt/T38/bioinf/xjj/Datasets/MVTec-AD
datapath=/mnt/T4_2/xjj/FirstWorkData1
loadpath=./models
#scorepath=/mnt/T4_2/xjj/patchcore/FirstWorkData  # 添加score_path变量，用于保存CSV文件
score_path=/mnt/T4_1/xjj/1/patchcore/FirstWorkData

modelfolder=IM320_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
# modelfolder=IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
savefolder=evaluated_results'/'$modelfolder

datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

CUDA_VISIBLE_DEVICES=2 python bin/load_and_evaluate_patchcore_heatmap.py --gpu 0 --seed 0 --score_path $score_path $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
