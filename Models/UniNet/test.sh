#单类
#CUDA_VISIBLE_DEVICES=7 python main.py --setting oc --dataset "MVTec AD"     #train
#CUDA_VISIBLE_DEVICES=2 python main.py --setting oc --dataset "MVTec AD" --load_ckpts     #test

#多类（没有复现结果）
#CUDA_VISIBLE_DEVICES=1 python main.py --setting mc --dataset "MVTec AD"     #train
#CUDA_VISIBLE_DEVICES=2 python main.py --setting mc --dataset "MVTec AD" --load_ckpts     #test

#CUDA_VISIBLE_DEVICES=3 python test_mvtec_oc.py 
CUDA_VISIBLE_DEVICES=3 python test_mvtec_oc_heatmap.py