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
from utils import evaluation_batch, cal_anomaly_maps, get_gaussian_kernel
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
    parser.add_argument('--save_dir', type=str, default='./test_results',
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