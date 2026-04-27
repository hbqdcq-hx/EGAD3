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

def get_filename_from_path(img_path):
    """
    从图像路径中提取文件名，使用路径的最后4层作为文件名
    
    Args:
        img_path: 图像路径
        
    Returns:
        str: 使用路径最后4层组成的文件名
    """
    # 标准化路径并分割
    normalized_path = os.path.normpath(img_path)
    path_parts = normalized_path.split(os.sep)
    
    # 获取最后4层路径，如果路径少于4层则使用所有层
    if len(path_parts) >= 4:
        last_four_levels = path_parts[-4:]
    else:
        last_four_levels = path_parts
    
    # 将最后4层路径用下划线连接作为文件名
    filename = '_'.join(last_four_levels)
    
    # 移除文件扩展名
    filename = os.path.splitext(filename)[0]
    
    return filename

def save_prediction_scores(img_paths, prediction_scores, class_name, save_dir):
    """
    保存预测分数到CSV文件
    
    Args:
        img_paths: 图像路径列表
        prediction_scores: 预测分数列表 (numpy array)
        class_name: 类别名称
        save_dir: 保存目录
        
    Returns:
        str: 保存的CSV文件路径
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 使用自定义函数生成文件名
    file_names = [get_filename_from_path(path) for path in img_paths]
    
    # 归一化预测分数到 [0, 1] 范围
    scores_norm = (prediction_scores - np.min(prediction_scores)) / (np.max(prediction_scores) - np.min(prediction_scores))

    
    # 保存预测分数到CSV，只包含图片名字和预测分数的一一对应
    csv_path = os.path.join(save_dir, f'anomaly_scores_{class_name}.csv')
    
    with open(csv_path, 'w') as f:
        # 写入表头
        f.write("File_Name,Anomaly_Score\n")
        # 归一化后添加这行代码
        #scores_norm = [float(score.item()) if isinstance(score, (np.ndarray, np.generic)) else float(score) for score in scores_norm]
        # 使用zip确保一一对应
        for name, score in zip(file_names, scores_norm):
            f.write(f"{name},{score.item():.10f}\n")  # 使用逗号分隔的表格格式
    
    print(f"异常分数已保存到: {csv_path}")
    return csv_path


def save_heatmaps(anomaly_maps, img_paths, class_name, save_dir):
    """
    保存热力图到指定目录
    
    Args:
        anomaly_maps: 异常图列表 (numpy arrays)
        img_paths: 图像路径列表
        class_name: 类别名称
        save_dir: 保存目录
        
    Returns:
        str: 保存热力图的目录路径
        int: 保存的热力图数量
    """
    # 创建类别文件夹
    class_dir = os.path.join(save_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    # 获取文件名列表
    file_names = [get_filename_from_path(path) for path in img_paths]
    
    count = 0
    for batch_idx, anomaly_map in enumerate(anomaly_maps):
        # 获取原始图像大小
        # 从数据集的img_paths中获取原始图像路径
        if batch_idx < len(img_paths):
            img_path = img_paths[batch_idx]
        else:
            img_path = None
        
       # 读取原始图像获取大小
        if img_path is not None:
            original_img = cv2.imread(img_path)
            if original_img is not None:
                orig_height, orig_width = original_img.shape[:2]
            else:
                # 如果无法读取原始图像，抛出异常
                raise FileNotFoundError(f"无法读取图像文件: {img_path}")
        else:
            # 如果没有图像路径，抛出异常
            raise ValueError("图像路径为None，无法获取原始图像大小")
        
        # 将异常图上采样到原始图像大小
        # Convert anomaly_map to tensor for interpolation
        anomaly_map_tensor = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        
        target_size = (orig_height, orig_width)
        
        # Upsample to original size using bilinear interpolation
        upsampled = F.interpolate(
            anomaly_map_tensor,
            size=target_size,
            mode='bilinear',
            align_corners=True
        ).squeeze()  # [H_original, W_original]
        
        # Convert back to numpy and normalize
        upsampled_np = upsampled.cpu().numpy()
        
        # 直接实现 min_max_norm 功能
        # 归一化到 [0, 1] 范围
        a_min, a_max = upsampled_np.min(), upsampled_np.max()
        if a_max - a_min > 0:
            ano_map = (upsampled_np - a_min) / (a_max - a_min)
        else:
            ano_map = upsampled_np * 0  # 全部设为0
        
        # 直接实现 cvt2heatmap 功能
        # 将归一化的异常图转换为热力图
        # 首先将值缩放到 0-255 范围
        heatmap_gray = np.uint8(ano_map * 255)
        # 使用 OpenCV 的 COLORMAP_JET 颜色映射
        heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        
        # 获取对应的文件名
        if batch_idx < len(file_names):
            filename = file_names[batch_idx]
        else:
            filename = f"heatmap_{count}"
        
        # 保存热力图
        heatmap_path = os.path.join(class_dir, f"{filename}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        
        count += 1
    
    print(f"热力图保存完成，共保存 {count} 张热力图到 {class_dir}")
    return class_dir, count

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
        
        
        # 保存预测分数
        csv_path = save_prediction_scores(img_paths, anomaly_score, class_name, save_dir)
        # 保存热力图
        heatmap_dir, heatmap_count = save_heatmaps(anomaly_map, img_paths, class_name, save_dir)
    
    return {
        'test_classes': class_list,
        'image_auroc_mean': np.mean(image_auroc_list) if image_auroc_list else 0,
        'pixel_auroc_mean': np.mean(pixel_auroc_list) if pixel_auroc_list else 0,
        'pixel_aupro_mean': np.mean(pixel_aupro_list) if pixel_aupro_list else 0
    }


if __name__ == '__main__':

    class_list = ['carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']  

    test_mvtec_by_list(class_list, gpu_id=1, save_dir="/mnt/T4_1/xjj/2/UniNet/FirstWorkData/")      #/mnt/T4_2/xjj/UniNet/FirstWorkData1/   /mnt/T38/bioinf/xjj/Datasets/MVTec-AD/
