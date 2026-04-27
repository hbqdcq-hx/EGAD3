#!/usr/bin/env python3
"""
独立测试脚本 for Dinomaly (MVTec-AD 多类统一训练模型)
从dinomaly_mvtec_uni.py中提取的测试逻辑
仅保留I-AUROC结果输出
"""

import torch
import torch.nn as nn
from dataset import get_data_transforms, MVTecDataset
from models.uad import ViTill
from models import vit_encoder
import numpy as np
import os
import argparse
import logging
from torch.utils.data import DataLoader
from utils_heatmap import evaluation_batch, cal_anomaly_maps, get_gaussian_kernel
import warnings

warnings.filterwarnings("ignore")


import torch.nn.functional as F
import cv2

def get_logger(name, save_path=None, level='INFO'):
    """创建日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    return logger


def load_model(model_path, encoder_name='dinov2reg_vit_base_14', device='cuda'):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重文件路径
        encoder_name: 编码器名称，必须与训练时一致
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        model: 加载好的模型
    """
    print(f"加载编码器: {encoder_name}")
    encoder = vit_encoder.load(encoder_name)
    
    # 模型配置（必须与训练时一致）
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    
    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")
    
    # 创建bottleneck和decoder（结构与训练时一致）
    from models.vision_transformer import bMlp, Block as VitBlock, LinearAttention2
    from functools import partial
    
    bottleneck = []
    decoder = []
    
    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)
    
    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)
    
    # 创建模型
    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        mask_neighbor_size=0
    )
    
    # 加载权重
    print(f"加载模型权重: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    return model

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
        # 使用zip确保一一对应
        for name, score in zip(file_names, scores_norm):
            f.write(f"{name},{score:.10f}\n")  # 使用逗号分隔的表格格式
    
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

def test_model(model, data_path, item_list, batch_size=16, device='cuda', logger=None, save_dir=None):
    """
    测试模型在所有类别上的性能，并保存预测分数和热力图
    
    Args:
        model: 训练好的模型
        data_path: 数据集路径
        item_list: 类别列表
        batch_size: 批大小
        device: 设备
        logger: 日志记录器或打印函数
        save_dir: 结果保存目录（如果为None，则不保存）
    
    Returns:
        result_list: 每个类别的I-AUROC结果列表
        mean_auroc_sp: 平均I-AUROC结果
    """
    # 处理logger参数：如果是logging.Logger对象，使用info方法；否则直接调用
    if logger is None:
        log_func = print
    elif hasattr(logger, 'info'):
        log_func = logger.info
    else:
        log_func = logger
    
    # 数据转换（与训练时一致）
    image_size = 448
    crop_size = 392
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    
    # 准备测试数据
    test_data_list = []
    for item in item_list:
        test_path = os.path.join(data_path, item)
        test_data = MVTecDataset(root=test_path, transform=data_transform, 
                                gt_transform=gt_transform, phase="test")
        test_data_list.append(test_data)
    
    # 测试每个类别
    result_list = []  # 存储(类别, I-AUROC)
    auroc_sp_list = []
    
    log_func("=" * 80)
    log_func("开始测试")
    log_func("=" * 80)
    
    for idx, (item, test_data) in enumerate(zip(item_list, test_data_list)):
        test_dataloader = DataLoader(test_data, batch_size=batch_size, 
                                    shuffle=False, num_workers=4)
        
        log_func(f"测试类别 [{idx+1}/{len(item_list)}]: {item}")
        
        # 创建类别保存目录
        class_save_dir = os.path.join(save_dir, item)
        os.makedirs(class_save_dir, exist_ok=True)
        
         # 调用 evaluation_batch 获取所有结果（包括指标、预测分数、异常图和图像路径）
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, prediction_scores, anomaly_maps, img_paths = evaluation_batch(
            model, test_dataloader, device, _class_=item, max_ratio=0.01, resize_mask=256
        )
        
        # 保存预测分数
        csv_path = save_prediction_scores(
            img_paths, prediction_scores, item, class_save_dir
        )
        
        # 保存热力图
        heatmap_dir, heatmap_count = save_heatmaps(
            anomaly_maps, img_paths, item, class_save_dir
        )
        
        # 记录结果
        result_list.append((item, auroc_sp))
        auroc_sp_list.append(auroc_sp)
        
        # 仅输出I-AUROC结果
        log_func(f"{item}: I-AUROC = {auroc_sp:.4f}")
    
    # 计算平均I-AUROC
    mean_auroc_sp = np.mean(auroc_sp_list)
    return result_list, mean_auroc_sp

def main():
    parser = argparse.ArgumentParser(description='Dinomaly MVTec-AD 多类统一模型测试脚本（仅保留I-AUROC）')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型权重文件路径')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection',
                       help='MVTec-AD数据集路径 (默认: ../mvtec_anomaly_detection)')
    parser.add_argument('--encoder_name', type=str, default='dinov2reg_vit_base_14',
                       help='编码器名称 (默认: dinov2reg_vit_base_14)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批大小 (默认: 16)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备: cuda 或 cpu (默认: cuda)')
    parser.add_argument('--save_dir', type=str, default='/mnt/T4_1/xjj/2/Dinomaly/FirstWorkData',
                       help='结果保存目录 (默认: ./test_results)')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    logger = get_logger(args.save_dir)
    
    # MVTec-AD类别列表
    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 
                'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
                'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    

    
    # 1. 加载模型
    logger.info(f"设备: {args.device}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"数据集路径: {args.data_path}")
    logger.info(f"编码器: {args.encoder_name}")
    logger.info(f"测试类别数: {len(item_list)}")
    
    model = load_model(args.model_path, args.encoder_name, args.device)
    
    # 2. 测试模型（仅返回I-AUROC相关结果）
    result_list, mean_auroc_sp = test_model(
        model=model,
        data_path=args.data_path,
        item_list=item_list,
        batch_size=args.batch_size,
        device=args.device,
        logger=logger,
        save_dir=args.save_dir
    )
    return 0


if __name__ == '__main__':
    exit(main())