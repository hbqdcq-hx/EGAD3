"""
MVTec AD单类测试脚本 - 简单版本
根据类别列表测试相应数量的类别
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tabulate import tabulate

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import parsing_args
from test import test
from datasets import MVTecDataset
from utils import setup_seed

def test_mvtec_by_list(class_list, gpu_id=2, save_dir="./saved_results"):
    """
    根据类别列表测试MVTec AD单类模型
    
    参数:
        class_list: 类别列表，有几个类就测试几个类
        gpu_id: CUDA设备ID
        save_dir: 结果保存目录

     注意:
        测试数据路径由 datasets.py 中的 MVTecDataset 类决定，
        默认路径为 '/mnt/T38/bioinf/xjj/Datasets/' + dataset。
        如需修改数据路径，请修改 datasets.py 中的 MVTecDataset 类。
    """    
    # 设置参数
    test_args = [
        "--setting", "oc",
        "--dataset", "MVTec AD",
        "--load_ckpts",
        "--save_dir", save_dir,
    ]
    
    print("=" * 60)
    print("MVTec AD单类测试脚本")
    print("=" * 60)
    
    # 解析参数
    sys.argv = ['test_mvtec_oc.py'] + test_args
    c = parsing_args()
    
    # 设置随机种子
    setup_seed(1203)
    
    # 设置CUDA设备
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"测试 {len(class_list)} 个类别: {class_list}")
    print(f"权重路径: ./ckpts/MVTec AD/{{类别名}}/BEST_P_PRO.pth")
    print("-" * 60)
    
    # 测试类别
    image_auroc_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    table_ls = []
    
    for idx, class_name in enumerate(class_list):
        c._class_ = class_name
        print(f"测试类别 [{idx+1}/{len(class_list)}]: {class_name}")
        
        
        # 调用测试函数，现在返回五个值
        #auroc_sp, auroc_px, pro, anomaly_score, anomaly_map = test(c, suffix='BEST_P_PRO')
        auroc_sp, auroc_px, pro, anomaly_score, anomaly_map = test(c, suffix='BEST_P_PRO')

        
        image_auroc_list.append(auroc_sp)
        pixel_auroc_list.append(auroc_px)
        pixel_aupro_list.append(pro)

        print(f"  I_AUROC: {auroc_sp:.1f}, P_AUROC: {auroc_px:.1f}, AUPRO: {pro:.1f}")
        
        # 获取测试数据集的图像路径
        test_dataset = MVTecDataset(c, is_train=False)
        img_paths = test_dataset.x  # 图像路径列表
        
    
    return {
        'test_classes': class_list,
        'image_auroc_mean': np.mean(image_auroc_list) if image_auroc_list else 0,
        'pixel_auroc_mean': np.mean(pixel_auroc_list) if pixel_auroc_list else 0,
        'pixel_aupro_mean': np.mean(pixel_aupro_list) if pixel_aupro_list else 0
    }


if __name__ == '__main__':

    class_list = ['carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']  
    test_mvtec_by_list(class_list, gpu_id=1, save_dir=" ")      #/mnt/T4_2/xjj/UniNet/FirstWorkData1/
