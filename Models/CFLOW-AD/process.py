import os
import numpy as np
from typing import Dict, Any, List, Tuple, Union
import glob
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import math

def get_filename_from_path(img_path: str) -> str:
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


def extract_class_from_path(filepath):
    """
    从文件路径中提取MVTec-AD类别信息
    
    Args:
        filepath: 文件路径
        
    Returns:
        类别名称或None
    """
    # 标准化路径
    normalized_path = os.path.normpath(filepath)
    parts = normalized_path.split(os.sep)
    
    # MVTec-AD的15个类别
    mvtec_classes = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    
    # 在路径中查找类别
    for part in parts:
        if part in mvtec_classes:
            return part
    
    # 如果没有找到，返回None
    return None


def extract_class_from_path(filepath):
    """
    从文件路径中提取MVTec-AD类别信息
    
    Args:
        filepath: 文件路径
        
    Returns:
        类别名称或None
    """
    # 标准化路径
    normalized_path = os.path.normpath(filepath)
    parts = normalized_path.split(os.sep)
    
    # MVTec-AD的15个类别
    mvtec_classes = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    
    # 在路径中查找类别
    for part in parts:
        if part in mvtec_classes:
            return part
    
    # 如果没有找到，返回None
    return None

def process_cflow_ad(model_output_path: str, save_dir: str, global_max_values: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Args:
        model_output_path: 模型输出文件路径（6个文件中的任意一个）
        save_dir: 保存目录
        global_max_values: 可选的全局最大值字典，如果提供则使用全局归一化
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理CFLOW-AD模型: {model_output_path}")
    
    if global_max_values:
        print("  使用全局归一化模式")
    else:
        print("  使用单样本归一化模式（临时方案）")
    
    # 常量定义（根据cflow-ad论文）
    _GCONST_ = -0.9189385332046727  # -log(sqrt(2*pi))
    
    # 获取目录
    base_dir = os.path.dirname(model_output_path)
    
    # 查找所有z文件（z_reshaped_layer*.npy）
    z_pattern = os.path.join(base_dir, "z_reshaped_layer*.npy")
    z_files = sorted(glob.glob(z_pattern))
    
    # 查找所有log_jac_det文件（log_jac_det_reshaped_layer*.npy）
    log_pattern = os.path.join(base_dir, "log_jac_det_reshaped_layer*.npy")
    log_jac_det_files = sorted(glob.glob(log_pattern))
    
    if not z_files or not log_jac_det_files:
        print(f"  警告: 在目录 {base_dir} 中未找到CFLOW-AD输出文件")
        return {
            "model": "CFLOW-AD",
            "input_shape": "未找到文件",
            "output_shape": (1,),
            "prediction_score": 0.0,
            "file_path": model_output_path,
            "filename": get_filename_from_path(base_dir)
        }
    
    print(f"  找到 {len(z_files)} 个尺度")
    
    # 加载所有文件
    z_tensors = []
    log_jac_det_tensors = []
    layer_heights = []
    layer_widths = []
    
    for i, (z_file, log_file) in enumerate(zip(z_files, log_jac_det_files)):
        # 从文件名中提取尺寸信息
        filename = os.path.basename(z_file)
        # 文件名格式: z_reshaped_layer{l}_{H}x{W}.npy
        import re
        match = re.search(r'(\d+)x(\d+)\.npy$', filename)
        if match:
            H, W = int(match.group(1)), int(match.group(2))
            layer_heights.append(H)
            layer_widths.append(W)
            print(f"  尺度 {i}: {H}x{W}")
        else:
            # 如果无法从文件名提取，尝试从数据形状推断
            print(f"  警告: 无法从文件名 {filename} 提取尺寸信息")
        
        # 加载z文件
        z_data = np.load(z_file)
        z_tensor = torch.from_numpy(z_data)
        
        # 加载log_jac_det文件
        log_data = np.load(log_file)
        log_tensor = torch.from_numpy(log_data)
        
        # 统一处理：确保所有张量都是4D（批次维度为1）
        if len(z_tensor.shape) == 3:
            # [C, H, W] -> [1, C, H, W]
            z_tensor = z_tensor.unsqueeze(0)
        
        if len(log_tensor.shape) == 2:
            # [H, W] -> [1, H, W]
            log_tensor = log_tensor.unsqueeze(0)
        
        # 现在处理4D张量
        B, C, H, W = z_tensor.shape
        S = H * W
        E = B * S
        
        # 将z变回原始形状
        z_original = z_tensor.reshape(B, C, S).transpose(1, 2).reshape(E, C)
        
        # 将log_jac_det变回原始形状
        log_original = log_tensor.reshape(E)
        
        z_tensors.append(z_original)
        log_jac_det_tensors.append(log_original)
        
        # 如果之前没有从文件名提取到尺寸，现在从数据形状获取
        if i >= len(layer_heights):
            layer_heights.append(H)
            layer_widths.append(W)
    
    # 按照cflow-ad.py中的后处理逻辑:
    # 1. 计算每个尺度的log_prob
    # 2. 对每个尺度的log_prob进行归一化：减去该尺度的最大值
    # 3. 转换为概率：exp(log_prob) 得到[0:1]范围的概率
    # 4. 重塑为特征图
    # 5. 上采样到统一尺寸
    # 6. 聚合所有尺度
    # 7. 转换为异常分数：score_map.max() - score_map
    
    # 确定上采样尺寸
    # 根据用户提供的命令参数，不同类别使用不同的-inp参数：
    # -inp 128: transistor
    # -inp 256: cable, capsule, pill, hazelnut, metal_nut  
    # -inp 512: bottle, carpet, grid, leather, screw, tile, toothbrush, wood, zipper
    # 这个-inp参数很可能就是c.crp_size
    
    # 提取类别
    class_name = extract_class_from_path(model_output_path)
    
    # 根据类别确定上采样尺寸（匹配用户命令中的-inp参数）
    target_size_map = {
        "transistor": 128,
        "cable": 256,
        "capsule": 256,
        "pill": 256,
        "hazelnut": 256,
        "metal_nut": 256,
        "bottle": 512,
        "carpet": 512,
        "grid": 512,
        "leather": 512,
        "screw": 512,
        "tile": 512,
        "toothbrush": 512,
        "wood": 512,
        "zipper": 512
    }
    
    if class_name and class_name in target_size_map:
        target_size = target_size_map[class_name]
        print(f"  检测到类别: {class_name}, 使用上采样尺寸: {target_size}x{target_size}")
    else:
        # 默认使用256x256
        target_size = 256
        print(f"  未检测到类别或类别未知, 使用默认上采样尺寸: {target_size}x{target_size}")
    
    test_maps = []
    
    for i, (z, log_jac_det) in enumerate(zip(z_tensors, log_jac_det_tensors)):
        # 计算log概率（每个维度的似然）
        C = z.shape[1] 
        logp = C * _GCONST_ - 0.5 * torch.sum(z**2, dim=1) + log_jac_det
        log_prob = logp / C  # 每个维度的似然
        
        # 转换为torch.double类型，与cflow-ad.py中的逻辑一致
        log_prob = log_prob.double()
        
        # 获取当前尺度的空间维度
        H = layer_heights[i] if i < len(layer_heights) else 64
        W = layer_widths[i] if i < len(layer_widths) else 64
        
        # 将log概率重塑为特征图 [B, H, W]
        B = 1  # 单张图片
        E = log_prob.shape[0]
        S = H * W
        
        # 确保E = B * S
        if E == S:  # B=1
            # 归一化逻辑：优先使用全局最大值，否则使用当前样本最大值
            if global_max_values and f"layer{i}" in global_max_values:
                # 使用全局最大值进行归一化（正确的方法）
                norm_max = global_max_values[f"layer{i}"]
                log_prob_norm = log_prob - norm_max
                print(f"  尺度 {i}: 使用全局最大值 {norm_max:.6f} 进行归一化")
            else:
                # 使用当前样本最大值（临时方案）
                norm_max = torch.max(log_prob)
                log_prob_norm = log_prob - norm_max
                if not global_max_values:
                    print(f"  尺度 {i}: 使用单样本最大值 {norm_max.item():.6f} 进行归一化（临时方案）")
                else:
                    print(f"  尺度 {i}: 未找到全局最大值，使用单样本最大值 {norm_max.item():.6f}")
            
            # 转换为概率：exp(log_prob) 得到[0:1]范围的概率
            test_prob = torch.exp(log_prob_norm)
            
            # reshape为特征图
            test_prob = test_prob.reshape(B, H, W)
        else:
            # 如果形状不匹配，尝试调整
            # 归一化逻辑：优先使用全局最大值，否则使用当前样本最大值
            if global_max_values and f"layer{i}" in global_max_values:
                norm_max = global_max_values[f"layer{i}"]
                log_prob_norm = log_prob - norm_max
            else:
                norm_max = torch.max(log_prob)
                log_prob_norm = log_prob - norm_max
            
            # 转换为概率
            test_prob = torch.exp(log_prob_norm)
            
            # reshape
            test_prob = test_prob.reshape(-1, H, W)
            B = test_prob.shape[0]
        
        # 上采样到统一尺寸
        upsampled = F.interpolate(test_prob.unsqueeze(1), 
                                 size=(target_size, target_size), 
                                 mode='bilinear', 
                                 align_corners=True).squeeze()
        
        test_maps.append(upsampled)
    
    # 聚合所有尺度的特征图
    if test_maps:
        # 将所有尺度的特征图堆叠并求和
        score_map = torch.stack(test_maps).sum(dim=0)
        
        # 将概率转换为异常分数（概率越高表示越正常，所以需要反转）
        super_mask = score_map.max() - score_map
        
        # 计算检测分数：取每个空间位置的最大值
        prediction_score = float(super_mask.max().item())
    else:
        prediction_score = 0.0
    
    print(f"  预测分数: {prediction_score:.10f}")
    
    return {
        "model": "CFLOW-AD",
        "input_shape": f"多尺度特征 ({len(z_tensors)}个尺度)",
        "output_shape": (1,),  # 标量分数
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(base_dir),
        "num_scales": len(z_tensors),
        "target_size": target_size,
        "normalization_mode": "global" if global_max_values else "single_sample"
    }

def collect_cflow_global_max_values(base_dir: str, class_name: str = None) -> Dict[str, float]:
    """
    收集CFLOW-AD测试集的全局最大值
    
    Args:
        base_dir: 基础目录
        class_name: 可选，指定类别
        
    Returns:
        全局最大值字典：{"layer0": max_value, "layer1": max_value, ...}
    """
    print(f"开始收集CFLOW-AD全局最大值...")
    if class_name:
        print(f"类别: {class_name}")
    
    # 导入批量处理器的函数
    try:
        from cflow_batch_processor import find_all_cflow_files, collect_global_max_values
        
        # 查找所有文件
        file_paths = find_all_cflow_files(base_dir, class_name)
        if not file_paths:
            print("未找到CFLOW-AD文件")
            return {}
        
        print(f"找到 {len(file_paths)} 张图片")
        
        # 收集全局最大值
        global_max_values = collect_global_max_values(file_paths)
        
        print("全局最大值收集完成")
        return global_max_values
        
    except ImportError:
        print("警告: 无法导入cflow_batch_processor，请确保文件存在")
        return {}
    except Exception as e:
        print(f"收集全局最大值时出错: {e}")
        return {}


def get_logp(C, z, logdet_J):
    """
    计算log概率，与源代码cflow-ad.py中的get_logp函数一致
    
    Args:
        C: 通道数
        z: 潜在变量
        logdet_J: Jacobian行列式的对数
        
    Returns:
        log概率
    """
    _GCONST_ = -0.9189385332046727  # -log(sqrt(2*pi))
    logp = C * _GCONST_ - 0.5 * torch.sum(z**2, dim=1) + logdet_J
    return logp


def process_cflow_ad_global(model_output_path: str, save_dir: str, global_max_cache: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
    """
    使用全局归一化处理CFLOW-AD模型输出
    
    与源代码cflow-ad.py中的逻辑完全对应：
    1. 读取一个类别的所有三个层数据
    2. 分别计算每层的最大值（使用test_dist逻辑）
    3. 使用每个层的最大值进行归一化：test_norm -= torch.max(test_norm)
    
    Args:
        model_output_path: 模型输出文件路径
        save_dir: 保存目录
        global_max_cache: 可选的全局最大值缓存，避免重复计算
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理CFLOW-AD模型（全局归一化模式）: {model_output_path}")
    
    # 提取类别信息
    class_name = extract_class_from_path(model_output_path)
    
    # 获取全局最大值
    global_max_values = None
    
    if global_max_cache and class_name in global_max_cache:
        # 使用缓存
        global_max_values = global_max_cache[class_name]
        print(f"  使用缓存的全局最大值（类别: {class_name}）")
    else:
        # 计算全局最大值 - 与源代码test_meta_epoch函数逻辑一致
        print(f"  开始计算类别 {class_name} 的全局最大值...")
        
        # 获取基础目录
        base_dir = "/mnt/T4_1/xjj/cflow-ad/MVTec-AD_numpy"
        class_base_dir = os.path.join(base_dir, class_name)
        
        # 查找该类别的所有测试图片目录
        test_dirs = []
        for root, dirs, files in os.walk(class_base_dir):
            if "test" in root:
                test_dirs.append(root)
        
        # 初始化每层的log_prob列表（模拟test_dist）
        global_max_values = {}
        test_dist = {"layer0": [], "layer1": [], "layer2": []}  # 与源代码test_dist结构一致
        
        # 遍历所有测试图片目录，收集每个层的log_prob值
        for test_dir in test_dirs:
            # 查找该目录下的所有z文件
            z_pattern = os.path.join(test_dir, "*", "z_reshaped_layer*.npy")
            z_files = sorted(glob.glob(z_pattern))
            
            # 查找该目录下的所有log_jac_det文件
            log_pattern = os.path.join(test_dir, "*", "log_jac_det_reshaped_layer*.npy")
            log_jac_det_files = sorted(glob.glob(log_pattern))
            
            if not z_files or not log_jac_det_files:
                continue
            
            # 处理每个图片的每个层（模拟test_meta_epoch中的循环）
            for i, (z_file, log_file) in enumerate(zip(z_files, log_jac_det_files)):
                if i >= 3:  # 只处理前3个层
                    break
                    
                try:
                    # 加载z文件
                    z_data = np.load(z_file)
                    z_tensor = torch.from_numpy(z_data)
                    
                    # 加载log_jac_det文件
                    log_data = np.load(log_file)
                    log_tensor = torch.from_numpy(log_data)
                    
                    # 统一处理：确保所有张量都是4D（批次维度为1）
                    if len(z_tensor.shape) == 3:
                        z_tensor = z_tensor.unsqueeze(0)
                    
                    if len(log_tensor.shape) == 2:
                        log_tensor = log_tensor.unsqueeze(0)
                    
                    # 现在处理4D张量
                    B, C, H, W = z_tensor.shape
                    S = H * W
                    E = B * S
                    
                    # 将z变回原始形状
                    z_original = z_tensor.reshape(B, C, S).transpose(1, 2).reshape(E, C)
                    
                    # 将log_jac_det变回原始形状
                    log_original = log_tensor.reshape(E)
                    
                    # 计算log概率（使用与源代码一致的get_logp函数）
                    decoder_log_prob = get_logp(C, z_original, log_original)
                    log_prob = decoder_log_prob / C  # likelihood per dim（与源代码一致）
                    
                    # 转换为torch.double类型
                    log_prob = log_prob.double()
                    
                    # 将log_prob值添加到对应层的test_dist中（模拟源代码逻辑）
                    layer_key = f"layer{i}"
                    test_dist[layer_key].extend(log_prob.detach().cpu().tolist())
                    
                except Exception as e:
                    print(f"  处理文件失败: {z_file} - {e}")
                    continue
        
        # 计算每个层的全局最大值（与源代码test_norm -= torch.max(test_norm)逻辑一致）
        for layer_key, log_probs in test_dist.items():
            if log_probs:
                # 转换为tensor并计算最大值
                test_norm = torch.tensor(log_probs, dtype=torch.double)
                global_max_values[layer_key] = torch.max(test_norm).item()
                print(f"  {layer_key}: 最大值 = {global_max_values[layer_key]:.6f} (基于{len(log_probs)}个log_prob值)")
            else:
                global_max_values[layer_key] = 0.0
                print(f"  {layer_key}: 未找到数据，使用默认值0.0")
        
        # 更新缓存
        if global_max_cache is not None:
            global_max_cache[class_name] = global_max_values
        
        print(f"  类别 {class_name} 的全局最大值计算完成")
    
    # 使用全局最大值处理（保持原来的process_cflow_ad函数逻辑）
    return process_cflow_ad(model_output_path, save_dir, global_max_values)


def process_dsr(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理DSR模型输出
    
    根据DSR.py中的evaluate_model函数逻辑：
    1. DSR模型保存的是out_mask特征，形状为(1, 2, 256, 256)
    2. 需要先应用softmax：out_mask_sm = torch.softmax(out_mask, dim=1)
    3. 然后对out_mask_sm[:,1:,:,:]进行21x21平均池化
    4. 取最大值作为预测分数：image_score = np.max(out_mask_averaged)
    
    Args:
        model_output_path: 模型输出文件路径（npy文件）
        save_dir: 保存目录（可选）
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理DSR模型: {model_output_path}")
    
    # 获取文件所在目录
    base_dir = os.path.dirname(model_output_path)
    
    # 加载数据 - out_mask特征，形状为(1, 2, 256, 256)
    data = np.load(model_output_path)
    print(f"  输入形状: {data.shape}")
    
    # 将numpy数组转换为torch张量
    data_tensor = torch.from_numpy(data)
    
    # 1. 应用softmax：out_mask_sm = torch.softmax(out_mask, dim=1)
    out_mask_sm = torch.softmax(data_tensor, dim=1)
    
    # 2. 提取异常通道（通道1）：out_mask_sm[:,1:,:,:]
    anomaly_channel = out_mask_sm[:, 1:, :, :]  # 形状: (1, 1, 256, 256)
    
    # 3. 21x21平均池化，stride=1，padding=10（21//2）
    kernel_size = 21
    stride = 1
    padding = kernel_size // 2
    
    # 应用平均池化
    out_mask_averaged = F.avg_pool2d(anomaly_channel, kernel_size=kernel_size, 
                                     stride=stride, padding=padding)
    
    # 4. 取最大值作为预测分数：image_score = np.max(out_mask_averaged)
    prediction_score = float(torch.max(out_mask_averaged).item())
    
    return {
        "model": "DSR",
        "input_shape": data.shape,
        "output_shape": (1,),  # 标量分数
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(model_output_path),
        "kernel_size": kernel_size,
        "method_used": "softmax_channel1_avgpool"  # 使用的方法：softmax + 通道1 + 平均池化
    }

def process_dinomaly(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理DINOMALY模型输出
    
    根据dinomaly.py中的evaluation_batch函数逻辑：
    1. DINOMALY模型保存了en_f、en_a、de_f、de_a特征图（28x28分辨率）
    2. 需要计算余弦相似度得到异常图
    3. 应用高斯滤波（sigma=4）
    4. 根据max_ratio=0.01计算预测分数（使用前1%最大值的平均值）
    
    Args:
        model_output_path: 模型输出文件路径（任意一个npy文件）
        save_dir: 保存目录（可选）
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理DINOMALY模型: {model_output_path}")
    
    # 设置max_ratio参数（根据用户反馈，使用0.01）
    max_ratio = 0.01
    resize_mask = 256
    
    # 获取文件所在目录
    base_dir = os.path.dirname(model_output_path)
    
    # 检查目录中是否有DINOMALY的4个输出文件
    required_files = [
        "en_f_28x28.npy",
        "en_a_28x28.npy", 
        "de_f_28x28.npy",
        "de_a_28x28.npy"
    ]
    
    # 验证所有必需文件都存在
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"  警告: 缺少文件: {missing_files}")
        # 返回默认结果
        return {
            "model": "DINOMALY",
            "input_shape": "文件不完整",
            "output_shape": (1,),
            "prediction_score": 0.0,
            "file_path": model_output_path,
            "filename": get_filename_from_path(base_dir),
            "missing_files": missing_files,
            "max_ratio": max_ratio
        }
    
    # 加载所有特征图文件
    en_f = np.load(os.path.join(base_dir, "en_f_28x28.npy"))
    en_a = np.load(os.path.join(base_dir, "en_a_28x28.npy"))
    de_f = np.load(os.path.join(base_dir, "de_f_28x28.npy"))
    de_a = np.load(os.path.join(base_dir, "de_a_28x28.npy"))
    
    print(f"  加载成功: 4个文件，形状: {en_f.shape}, {en_a.shape}, {de_f.shape}, {de_a.shape}")
    
    # 设置设备与源代码一致（CUDA）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    # 转换为torch张量并移动到设备
    en_f_t = torch.from_numpy(en_f).to(device)
    en_a_t = torch.from_numpy(en_a).to(device)
    de_f_t = torch.from_numpy(de_f).to(device)
    de_a_t = torch.from_numpy(de_a).to(device)
    
    # 在计算异常图之前添加批次维度
    # 特征图形状可能是(C, H, W)，需要变成(1, C, H, W)
    def add_batch_dim_if_needed(tensor):
        if len(tensor.shape) == 3:
            return tensor.unsqueeze(0)
        return tensor
    
    en_f_t = add_batch_dim_if_needed(en_f_t)
    en_a_t = add_batch_dim_if_needed(en_a_t)
    de_f_t = add_batch_dim_if_needed(de_f_t)
    de_a_t = add_batch_dim_if_needed(de_a_t)
    
    # 根据dinomaly.py中的cal_anomaly_maps函数实现
    # 该函数计算1 - F.cosine_similarity(fs, ft)并上采样
    def compute_anomaly_map_single(fs, ft, out_size=392):
        """计算单个特征图对的异常图"""
        # 计算余弦相似度
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        return a_map
    
    # 计算两个特征图对的异常图
    # 根据dinomaly.py中的evaluation_batch函数，en和de都是包含两个元素的列表
    # en = [en_f, en_a], de = [de_f, de_a]
    anomaly_map_f = compute_anomaly_map_single(en_f_t, de_f_t, out_size=392)
    anomaly_map_a = compute_anomaly_map_single(en_a_t, de_a_t, out_size=392)
    
    # 聚合两个异常图（根据cal_anomaly_maps函数，使用mean聚合）
    anomaly_map = torch.cat([anomaly_map_f, anomaly_map_a], dim=1).mean(dim=1, keepdim=True)

    # 插值（感觉是多余的，不加指标也不会变化）
    anomaly_map_resized = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
    
    # 应用高斯滤波（sigma=4），与evaluation_batch函数中的gaussian_kernel一致
    # 使用与dinomaly.py中相同的get_gaussian_kernel函数
    def get_gaussian_kernel(kernel_size=5, sigma=4, channels=1):
        # 创建坐标网格
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        # 计算2D高斯核
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        
        # 确保高斯核的和为1
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # 重塑为深度可分离卷积权重
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        
        gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, 
                                         kernel_size=kernel_size, groups=channels,
                                         bias=False, padding=kernel_size // 2)
        
        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        
        return gaussian_filter

    # 应用高斯滤波
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4, channels=1).to(device)
    anomaly_map_smoothed = gaussian_kernel(anomaly_map_resized)
    
    # 根据max_ratio参数计算预测分数
    # 与evaluation_batch函数中的逻辑一致
    if max_ratio == 0:
        # 使用最大值
        sp_score = torch.max(anomaly_map_smoothed.flatten(1), dim=1)[0]
    else:
        # 使用前max_ratio比例的最大值的平均值
        anomaly_map_flat = anomaly_map_smoothed.flatten(1)
        sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :int(anomaly_map_flat.shape[1] * max_ratio)]
        sp_score = sp_score.mean(dim=1)
    
    prediction_score = float(sp_score[0].item())
    
    print(f"  预测分数: {prediction_score:.10f} (max_ratio={max_ratio})")
    
    # 将异常图转换为numpy数组（用于保存）
    anomaly_map_np = anomaly_map_smoothed[0, 0, :, :].cpu().detach().numpy()
    
    # 如果需要保存处理结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.basename(model_output_path)
        save_path = os.path.join(save_dir, f"dinomaly_{basename}")
        np.save(save_path, anomaly_map_np)
        print(f"  保存处理结果到: {save_path}")
    
    return {
        "model": "DINOMALY",
        "input_shape": f"4个特征图: {en_f.shape}, {en_a.shape}, {de_f.shape}, {de_a.shape}",
        "output_shape": anomaly_map_np.shape,
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(base_dir),
        "anomaly_map_shape": anomaly_map_np.shape,
        "max_ratio": max_ratio
    }

def process_deco_diff(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理DeCo-Diff模型输出
    
    根据DeCo_Diff.py中的evaluation函数逻辑：
    1. DeCo-Diff模型保存了4个npy文件：
       - x0_s.npy: 原始输入图像经过VAE解码后的结果 (1, 3, 256, 256)
       - encoded_s.npy: 输入图像在潜在空间的编码 (1, 4, 32, 32)
       - image_samples_s.npy: 扩散模型生成的图像 (1, 3, 256, 256)
       - latent_samples_s.npy: 扩散模型生成的潜在表示 (1, 4, 32, 32)
    2. 需要按照calculate_anomaly_maps函数的逻辑计算异常图
    3. 注意：calculate_anomaly_maps处理的是列表，但这里我们处理的是单个文件
    
    Args:
        model_output_path: 模型输出文件路径（任意一个npy文件）
        save_dir: 保存目录（可选）
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理DeCo-Diff模型: {model_output_path}")
    
    # 获取文件所在目录
    base_dir = os.path.dirname(model_output_path)
    
    # 检查目录中是否有DeCo-Diff的4个输出文件
    required_files = [
        "x0_s.npy",
        "encoded_s.npy", 
        "image_samples_s.npy",
        "latent_samples_s.npy"
    ]
    
    # 验证所有必需文件都存在
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"  警告: 缺少文件: {missing_files}")
        # 返回默认结果
        return {
            "model": "DeCo-Diff",
            "input_shape": "文件不完整",
            "output_shape": (1,),
            "prediction_score": 0.0,
            "file_path": model_output_path,
            "filename": get_filename_from_path(base_dir),
            "missing_files": missing_files
        }
    
    # 加载所有文件
    x0_s_data = np.load(os.path.join(base_dir, "x0_s.npy"))
    encoded_s_data = np.load(os.path.join(base_dir, "encoded_s.npy"))
    image_samples_s_data = np.load(os.path.join(base_dir, "image_samples_s.npy"))
    latent_samples_s_data = np.load(os.path.join(base_dir, "latent_samples_s.npy"))
    
    print(f"  加载成功: 4个文件，形状: {x0_s_data.shape}, {encoded_s_data.shape}, {image_samples_s_data.shape}, {latent_samples_s_data.shape}")
    
    # 转换为torch张量
    x0_s_t = torch.from_numpy(x0_s_data)
    encoded_s_t = torch.from_numpy(encoded_s_data)
    image_samples_s_t = torch.from_numpy(image_samples_s_data)
    latent_samples_s_t = torch.from_numpy(latent_samples_s_data)
    
    # 确保所有张量都是4D [B, C, H, W]
    # 注意：npy文件保存的是(1, C, H, W)形状，但calculate_anomaly_maps期望列表
    # 所以我们需要将张量包装在列表中
    if len(x0_s_t.shape) == 3:
        x0_s_t = x0_s_t.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    if len(encoded_s_t.shape) == 3:
        encoded_s_t = encoded_s_t.unsqueeze(0)
    if len(image_samples_s_t.shape) == 3:
        image_samples_s_t = image_samples_s_t.unsqueeze(0)
    if len(latent_samples_s_t.shape) == 3:
        latent_samples_s_t = latent_samples_s_t.unsqueeze(0)
    
    # 根据DeCo_Diff.py中的calculate_anomaly_maps函数实现
    # 注意：这里使用默认的center_size=256
    center_size = 256
    
    # 实现smooth_mask函数（从DeCo_Diff.py复制）
    def smooth_mask(mask, sigma=1.0):
        smoothed_mask = gaussian_filter(mask, sigma=sigma)
        return smoothed_mask
    
    # 模拟calculate_anomaly_maps中的循环
    # 由于我们只有单个图像，所以创建单元素列表
    x0_s_list = [x0_s_t]
    encoded_s_list = [encoded_s_t]
    image_samples_s_list = [image_samples_s_t]
    latent_samples_s_list = [latent_samples_s_t]
    
    pred_geometric = []
    
    for x, encoded, image_samples, latent_samples in zip(x0_s_list, encoded_s_list, image_samples_s_list, latent_samples_s_list):
        # 计算image_difference
        # 注意：image_samples和x的形状都是(1, C, H, W)
        image_difference = (((((torch.abs(image_samples-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))

        # 应用clip和缩放
        image_difference = np.clip(image_difference, 0.0, 0.4) * 2.5
        
        # 应用高斯滤波
        image_difference = smooth_mask(image_difference, sigma=3)
        
        # 计算latent_difference
        latent_difference = (((((torch.abs(latent_samples-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
        
        # 应用clip和缩放
        latent_difference = np.clip(latent_difference, 0.0, 0.2) * 5
        
        # 应用高斯滤波
        latent_difference = smooth_mask(latent_difference, sigma=1)
        
        # 调整大小
        latent_difference = resize(latent_difference, (center_size, center_size))
        
        # 计算最终异常图
        final_anomaly = image_difference * latent_difference
        
        # 取平方根
        final_anomaly = np.sqrt(final_anomaly)
        
        pred_geometric.append(final_anomaly)
    
    # 将列表转换为numpy数组
    pred_geometric_array = np.stack(pred_geometric, axis=0)  # 形状: (1, H, W)
    
    # 提取最大分数作为预测分数
    # 注意：calculate_anomaly_maps返回的是字典，其中'anomaly_geometric'是(1, H, W)数组
    # 我们需要取第一个（也是唯一一个）元素的最大值
    prediction_score = float(np.max(pred_geometric_array[0]))
    
    print(f"  预测分数: {prediction_score:.10f}")
    
    # 如果需要保存处理结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.basename(model_output_path)
        save_path = os.path.join(save_dir, f"deco_diff_{basename}")
        np.save(save_path, pred_geometric_array[0])
        print(f"  保存处理结果到: {save_path}")
    
    return {
        "model": "DeCo-Diff",
        "input_shape": f"4个特征图: {x0_s_data.shape}, {encoded_s_data.shape}, {image_samples_s_data.shape}, {latent_samples_s_data.shape}",
        "output_shape": pred_geometric_array[0].shape,
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(base_dir),
        "anomaly_map_shape": pred_geometric_array[0].shape,
        "method_used": "geometric_mean_sqrt"  # 使用的方法：几何平均取平方根
    }

def process_rd4ad(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理RD4AD模型输出：

    1. RD4AD模型输出包含6个npy文件（input_64x64.npy, input_32x32.npy, input_16x16.npy,
       output_64x64.npy, output_32x32.npy, output_16x16.npy）
    2. 需要计算余弦相似度得到异常图
    3. 从异常图中提取最大分数作为预测分数
    
    Args:
        model_output_path: 模型输出文件路径（任意一个npy文件）
        save_dir: 保存目录（可选）
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理RD4AD模型: {model_output_path}")
    
    # 获取文件所在目录
    base_dir = os.path.dirname(model_output_path)
    
    # 检查目录中是否有RD4AD的6个输出文件
    required_files = [
        "input_64x64.npy",
        "input_32x32.npy", 
        "input_16x16.npy",
        "output_64x64.npy",
        "output_32x32.npy",
        "output_16x16.npy"
    ]
    
    # 验证所有必需文件都存在
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            print("缺少文件")
    
    # 加载inputs
    input_64 = np.load(os.path.join(base_dir, "input_64x64.npy"))
    input_32 = np.load(os.path.join(base_dir, "input_32x32.npy"))
    input_16 = np.load(os.path.join(base_dir, "input_16x16.npy"))
    
    # 加载outputs
    output_64 = np.load(os.path.join(base_dir, "output_64x64.npy"))
    output_32 = np.load(os.path.join(base_dir, "output_32x32.npy"))
    output_16 = np.load(os.path.join(base_dir, "output_16x16.npy"))
    
    print(f"  加载成功: 6个文件，形状: {input_64.shape}, {input_32.shape}, {input_16.shape}")
    
    # 转换为torch张量
    input_64_t = torch.from_numpy(input_64)
    input_32_t = torch.from_numpy(input_32)
    input_16_t = torch.from_numpy(input_16)
    
    output_64_t = torch.from_numpy(output_64)
    output_32_t = torch.from_numpy(output_32)
    output_16_t = torch.from_numpy(output_16)

    # 在计算异常图之前添加批次维度
    # 输入形状是(256,64,64)，需要变成(1,256,64,64)
    input_64_t = input_64_t.unsqueeze(0)
    input_32_t = input_32_t.unsqueeze(0)
    input_16_t = input_16_t.unsqueeze(0)
    
    output_64_t = output_64_t.unsqueeze(0)
    output_32_t = output_32_t.unsqueeze(0)
    output_16_t = output_16_t.unsqueeze(0)
    
    # 该函数计算1 - F.cosine_similarity(fs, ft)
    def compute_anomaly_map_single(fs, ft, out_size=256):
        """计算单个尺度的异常图"""
        # 计算余弦相似度
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].cpu().detach().numpy()
        return a_map
    
    # 计算每个尺度的异常图
    anomaly_map_64 = compute_anomaly_map_single(input_64_t, output_64_t, out_size=256)
    anomaly_map_32 = compute_anomaly_map_single(input_32_t, output_32_t, out_size=256)
    anomaly_map_16 = compute_anomaly_map_single(input_16_t, output_16_t, out_size=256)
    
    # 根据rd4ad.py中的逻辑，使用'mul'模式相乘或'a'模式相加
    # 这里使用'a'模式（相加），与evaluation函数中的amap_mode='a'一致
    anomaly_map = anomaly_map_64 + anomaly_map_32 + anomaly_map_16
    
    # 应用高斯滤波（sigma=4），与evaluation函数中的gaussian_filter一致
    
    anomaly_map_smoothed = gaussian_filter(anomaly_map, sigma=4)
    
    # 提取最大分数作为预测分数
    prediction_score = float(np.max(anomaly_map_smoothed))
    
    print(f"  预测分数: {prediction_score:.10f}")
    
    # 如果需要保存处理结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.basename(model_output_path)
        save_path = os.path.join(save_dir, f"rd4ad_{basename}")
        np.save(save_path, anomaly_map_smoothed)
        print(f"  保存处理结果到: {save_path}")
    
    return {
        "model": "RD4AD",
        "input_shape": f"多尺度特征: {input_64.shape}, {input_32.shape}, {input_16.shape}",
        "output_shape": anomaly_map_smoothed.shape,
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(base_dir),
        "anomaly_map_shape": anomaly_map_smoothed.shape
    }

def process_rrd(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理RRD（Revisiting Reverse Distillation）模型输出：
    
    1. RRD模型输出包含6个npy文件（inputs_64x64.npy, inputs_32x32.npy, inputs_16x16.npy,
       outputs_64x64.npy, outputs_32x32.npy, outputs_16x16.npy）
    2. 需要计算余弦相似度得到异常图
    3. 从异常图中提取最大分数作为预测分数
    
    Args:
        model_output_path: 模型输出文件路径（任意一个npy文件）
        save_dir: 保存目录（可选）
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理RRD模型: {model_output_path}")
    
    # 获取文件所在目录
    base_dir = os.path.dirname(model_output_path)
    
    # 检查目录中是否有RRD的6个输出文件
    required_files = [
        "inputs_64x64.npy",
        "inputs_32x32.npy", 
        "inputs_16x16.npy",
        "outputs_64x64.npy",
        "outputs_32x32.npy",
        "outputs_16x16.npy"
    ]
    
    # 验证所有必需文件都存在
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            print(f"  警告: 缺少文件 {file_name}")
    
    # 加载inputs
    input_64 = np.load(os.path.join(base_dir, "inputs_64x64.npy"))
    input_32 = np.load(os.path.join(base_dir, "inputs_32x32.npy"))
    input_16 = np.load(os.path.join(base_dir, "inputs_16x16.npy"))
    
    # 加载outputs
    output_64 = np.load(os.path.join(base_dir, "outputs_64x64.npy"))
    output_32 = np.load(os.path.join(base_dir, "outputs_32x32.npy"))
    output_16 = np.load(os.path.join(base_dir, "outputs_16x16.npy"))
    
    print(f"  加载成功: 6个文件，形状: {input_64.shape}, {input_32.shape}, {input_16.shape}")
    
    # 转换为torch张量
    input_64_t = torch.from_numpy(input_64)
    input_32_t = torch.from_numpy(input_32)
    input_16_t = torch.from_numpy(input_16)
    
    output_64_t = torch.from_numpy(output_64)
    output_32_t = torch.from_numpy(output_32)
    output_16_t = torch.from_numpy(output_16)

    # 在计算异常图之前添加批次维度
    # 输入形状可能是(256,64,64)，需要变成(1,256,64,64)
    if len(input_64_t.shape) == 3:
        input_64_t = input_64_t.unsqueeze(0)
        input_32_t = input_32_t.unsqueeze(0)
        input_16_t = input_16_t.unsqueeze(0)
        
        output_64_t = output_64_t.unsqueeze(0)
        output_32_t = output_32_t.unsqueeze(0)
        output_16_t = output_16_t.unsqueeze(0)
    
    # 该函数计算1 - F.cosine_similarity(fs, ft)
    def compute_anomaly_map_single(fs, ft, out_size=256):
        """计算单个尺度的异常图"""
        # 计算余弦相似度
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].cpu().detach().numpy()
        return a_map
    
    # 计算每个尺度的异常图
    anomaly_map_64 = compute_anomaly_map_single(input_64_t, output_64_t, out_size=256)
    anomaly_map_32 = compute_anomaly_map_single(input_32_t, output_32_t, out_size=256)
    anomaly_map_16 = compute_anomaly_map_single(input_16_t, output_16_t, out_size=256)
    
    # 在evaluation_multi_proj函数中，amap_mode='a'表示相加
    anomaly_map = anomaly_map_64 + anomaly_map_32 + anomaly_map_16
    
    # 应用高斯滤波（sigma=4），与evaluation_multi_proj函数中的gaussian_filter一致
    anomaly_map_smoothed = gaussian_filter(anomaly_map, sigma=4)
    
    # 提取最大分数作为预测分数
    prediction_score = float(np.max(anomaly_map_smoothed))
    
    return {
        "model": "RRD",
        "input_shape": f"多尺度特征: {input_64.shape}, {input_32.shape}, {input_16.shape}",
        "output_shape": anomaly_map_smoothed.shape,
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(base_dir),
        "anomaly_map_shape": anomaly_map_smoothed.shape
    }

def process_simplenet(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理SimpleNet模型输出
    
    根据simplenet.py中的源代码后处理逻辑实现：
    1. SimpleNet保存的是每个图像的patch分数数组（npy文件）
    2. patch分数是负的discriminator输出：patch_scores = image_scores = -self.discriminator(features)
    3. 需要实现源代码中的后处理代码：
       image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
       image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
       image_scores = self.patch_maker.score(image_scores)
    
    Args:
        model_output_path: 模型输出文件路径（npy文件）
        save_dir: 保存目录（可选）
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理SimpleNet模型: {model_output_path}")
    
    # 加载数据 - patch分数数组
    data = np.load(model_output_path)
    print(f"  输入形状: {data.shape}")
    
    # 根据simplenet.py中的逻辑，patch分数是负的discriminator输出
    # 在simplenet.py的_predict方法中：
    # patch_scores = image_scores = -self.discriminator(features)
    
    # 将数据转换为numpy数组以便处理
    # 数据形状可能是(1296, 1)或(1296,)
    if len(data.shape) == 2 and data.shape[1] == 1:
        # 如果是(1296, 1)形状，展平为(1296,)
        data = data.flatten()
    
    print(f"  处理后的形状: {data.shape}")
    print(f"  patch分数原始统计: min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}, std={data.std():.6f}")
    
    # 实现源代码中的后处理代码
    # 对于单张图片，batchsize=1
    batchsize = 1
    num_patches = data.shape[0]
    
    # 根据simplenet.py中的PatchMaker类实现
    class PatchMaker:
        def __init__(self, patchsize=3, top_k=0, stride=None):
            self.patchsize = patchsize
            self.stride = stride
            self.top_k = top_k
        
        def unpatch_scores(self, x, batchsize):
            """模拟PatchMaker.unpatch_scores方法"""
            # 源代码：return x.reshape(batchsize, -1, *x.shape[1:])
            return x.reshape(batchsize, -1, *x.shape[1:])
        
        def score(self, x):
            """模拟PatchMaker.score方法"""
            # 转换为torch张量以模拟源代码
            was_numpy = False
            if isinstance(x, np.ndarray):
                was_numpy = True
                x_tensor = torch.from_numpy(x)
            else:
                x_tensor = x
            
            # 源代码逻辑
            while x_tensor.ndim > 2:
                x_tensor = torch.max(x_tensor, dim=-1).values
            if x_tensor.ndim == 2:
                if self.top_k > 1:
                    x_tensor = torch.topk(x_tensor, self.top_k, dim=1).values.mean(1)
                else:
                    x_tensor = torch.max(x_tensor, dim=1).values
            
            if was_numpy:
                return x_tensor.numpy()
            return x_tensor
    
    # 创建PatchMaker实例（使用默认参数）
    patch_maker = PatchMaker(patchsize=3, top_k=0, stride=1)
    
    # 1. unpatch_scores: 将一维数组重塑为(batchsize, num_patches, 1)
    # 原始代码：image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
    # 对于一维数组(1296,)，需要先添加一个维度变成(1296, 1)
    if len(data.shape) == 1:
        data_with_channel = data.reshape(-1, 1)  # (1296, 1)
    else:
        data_with_channel = data
    
    # 应用unpatch_scores：reshape(batchsize, -1, *x.shape[1:])
    # 对于(1296, 1) -> (1, 1296, 1)
    unpatch_scores = patch_maker.unpatch_scores(data_with_channel, batchsize=batchsize)
    
    # 2. reshape: image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
    # 对于(1, 1296, 1) -> (1, 1296)
    reshaped_scores = unpatch_scores.reshape(batchsize, num_patches)
    
    # 3. score: image_scores = self.patch_maker.score(image_scores)
    image_score_array = patch_maker.score(reshaped_scores)  # 形状: (1,)
    
    prediction_score = float(image_score_array[0])
    
    print(f"  预测分数（源代码后处理）: {prediction_score:.10f}")
    
    # 调试：显示不同的聚合方式
    print("  调试信息 - 不同聚合方式:")
    print(f"    源代码后处理结果: {prediction_score:.10f}")
    print(f"    简单最大值: {float(data.max()):.10f}")
    print(f"    原始平均值: {float(data.mean()):.10f}")
    
    # 显示归一化后的值以供参考
    img_min_score = data.min()
    img_max_score = data.max()
    if img_max_score - img_min_score != 0:
        normalized_scores = (data - img_min_score) / (img_max_score - img_min_score)
        print(f"    归一化后最大值: {float(normalized_scores.max()):.10f}")
        print(f"    归一化后平均值: {float(normalized_scores.mean()):.10f}")
    
    # 如果需要保存处理结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.basename(model_output_path)
        save_path = os.path.join(save_dir, f"simplenet_{basename}")
        np.save(save_path, data)  # 保存原始数据
        print(f"  保存处理结果到: {save_path}")
    
    return {
        "model": "SimpleNet",
        "input_shape": data.shape,
        "output_shape": (1,),  # 标量分数
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(model_output_path),
        "method_used": "source_code_postprocessing"  # 使用的方法：源代码后处理
    }

def process_comet(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    和处理SimpleNet模型一样
    """
    print(f"处理CoMet模型: {model_output_path}")
    
    # 加载数据 - patch分数数组
    data = np.load(model_output_path)
    print(f"  输入形状: {data.shape}")
    
    # 根据comet.py中的逻辑，patch分数是负的discriminator输出
    # 在comet.py的_predict方法中：
    # patch_scores = image_scores = -self.discriminator(features)
    
    # 将数据转换为numpy数组以便处理
    # 数据形状可能是(1296, 1)或(1296,)
    if len(data.shape) == 2 and data.shape[1] == 1:
        # 如果是(1296, 1)形状，展平为(1296,)
        data = data.flatten()
    
    print(f"  处理后的形状: {data.shape}")
    print(f"  patch分数原始统计: min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}, std={data.std():.6f}")
    
    # 实现源代码中的后处理代码
    # 对于单张图片，batchsize=1
    batchsize = 1
    num_patches = data.shape[0]
    
    # 根据comet.py中的PatchMaker类实现
    class PatchMaker:
        def __init__(self, patchsize=3, top_k=0, stride=None):
            self.patchsize = patchsize
            self.stride = stride
            self.top_k = top_k
        
        def unpatch_scores(self, x, batchsize):
            """模拟PatchMaker.unpatch_scores方法"""
            # 源代码：return x.reshape(batchsize, -1, *x.shape[1:])
            return x.reshape(batchsize, -1, *x.shape[1:])
        
        def score(self, x):
            """模拟PatchMaker.score方法"""
            # 转换为torch张量以模拟源代码
            was_numpy = False
            if isinstance(x, np.ndarray):
                was_numpy = True
                x_tensor = torch.from_numpy(x)
            else:
                x_tensor = x
            
            # 源代码逻辑
            while x_tensor.ndim > 2:
                x_tensor = torch.max(x_tensor, dim=-1).values
            if x_tensor.ndim == 2:
                if self.top_k > 1:
                    x_tensor = torch.topk(x_tensor, self.top_k, dim=1).values.mean(1)
                else:
                    x_tensor = torch.max(x_tensor, dim=1).values
            
            if was_numpy:
                return x_tensor.numpy()
            return x_tensor
    
    # 创建PatchMaker实例（使用默认参数）
    patch_maker = PatchMaker(patchsize=3, top_k=0, stride=1)
    
    # 1. unpatch_scores: 将一维数组重塑为(batchsize, num_patches, 1)
    # 原始代码：image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
    # 对于一维数组(1296,)，需要先添加一个维度变成(1296, 1)
    if len(data.shape) == 1:
        data_with_channel = data.reshape(-1, 1)  # (1296, 1)
    else:
        data_with_channel = data
    
    # 应用unpatch_scores：reshape(batchsize, -1, *x.shape[1:])
    # 对于(1296, 1) -> (1, 1296, 1)
    unpatch_scores = patch_maker.unpatch_scores(data_with_channel, batchsize=batchsize)
    
    # 2. reshape: image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
    # 对于(1, 1296, 1) -> (1, 1296)
    reshaped_scores = unpatch_scores.reshape(batchsize, num_patches)
    
    # 3. score: image_scores = self.patch_maker.score(image_scores)
    image_score_array = patch_maker.score(reshaped_scores)  # 形状: (1,)
    
    prediction_score = float(image_score_array[0])
    
    print(f"  预测分数（源代码后处理）: {prediction_score:.10f}")
    
    # 调试：显示不同的聚合方式
    print("  调试信息 - 不同聚合方式:")
    print(f"    源代码后处理结果: {prediction_score:.10f}")
    print(f"    简单最大值: {float(data.max()):.10f}")
    print(f"    原始平均值: {float(data.mean()):.10f}")
    
    # 显示归一化后的值以供参考
    img_min_score = data.min()
    img_max_score = data.max()
    if img_max_score - img_min_score != 0:
        normalized_scores = (data - img_min_score) / (img_max_score - img_min_score)
        print(f"    归一化后最大值: {float(normalized_scores.max()):.10f}")
        print(f"    归一化后平均值: {float(normalized_scores.mean()):.10f}")
    
    # 如果需要保存处理结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.basename(model_output_path)
        save_path = os.path.join(save_dir, f"comet_{basename}")
        np.save(save_path, data)  # 保存原始数据
        print(f"  保存处理结果到: {save_path}")
    
    return {
        "model": "CoMet",
        "input_shape": data.shape,
        "output_shape": (1,),  # 标量分数
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(model_output_path),
        "method_used": "source_code_postprocessing"  # 使用的方法：源代码后处理
    }

def process_uninet(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理UniNet模型输出：
    
    1. UniNet模型输出包含12个npy文件：
       - source_t_64x64.npy, source_t_32x32.npy, source_t_16x16.npy
       - target_t_64x64.npy, target_t_32x32.npy, target_t_16x16.npy
       - target_s_64x64.npy, target_s_32x32.npy, target_s_16x16.npy
       - source_s_64x64.npy, source_s_32x32.npy, source_s_16x16.npy
    2. 根据uninet.py中的逻辑，需要计算1 - F.cosine_similarity(t, s)得到异常图
    3. 使用weighted_decision_mechanism聚合多尺度异常图（根据mechanism.py实现）
    4. 从异常图中提取最大分数作为预测分数
    
    Args:
        model_output_path: 模型输出文件路径（任意一个npy文件）
        save_dir: 保存目录（可选）
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理UniNet模型: {model_output_path}")
    
    # 获取文件所在目录
    base_dir = os.path.dirname(model_output_path)
    
    # 检查目录中是否有UniNet的12个输出文件
    required_files = [
        "source_t_64x64.npy", "source_t_32x32.npy", "source_t_16x16.npy",
        "target_t_64x64.npy", "target_t_32x32.npy", "target_t_16x16.npy",
        "target_s_64x64.npy", "target_s_32x32.npy", "target_s_16x16.npy",
        "source_s_64x64.npy", "source_s_32x32.npy", "source_s_16x16.npy"
    ]
    
    # 验证所有必需文件都存在
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"  警告: 缺少文件: {missing_files}")
        # 返回默认结果
        return {
            "model": "UniNet",
            "input_shape": "文件不完整",
            "output_shape": (1,),
            "prediction_score": 0.0,
            "file_path": model_output_path,
            "filename": get_filename_from_path(base_dir),
            "missing_files": missing_files
        }
    
    # 加载所有特征图文件
    source_t_64 = np.load(os.path.join(base_dir, "source_t_64x64.npy"))
    source_t_32 = np.load(os.path.join(base_dir, "source_t_32x32.npy"))
    source_t_16 = np.load(os.path.join(base_dir, "source_t_16x16.npy"))
    
    target_t_64 = np.load(os.path.join(base_dir, "target_t_64x64.npy"))
    target_t_32 = np.load(os.path.join(base_dir, "target_t_32x32.npy"))
    target_t_16 = np.load(os.path.join(base_dir, "target_t_16x16.npy"))
    
    target_s_64 = np.load(os.path.join(base_dir, "target_s_64x64.npy"))
    target_s_32 = np.load(os.path.join(base_dir, "target_s_32x32.npy"))
    target_s_16 = np.load(os.path.join(base_dir, "target_s_16x16.npy"))
    
    source_s_64 = np.load(os.path.join(base_dir, "source_s_64x64.npy"))
    source_s_32 = np.load(os.path.join(base_dir, "source_s_32x32.npy"))
    source_s_16 = np.load(os.path.join(base_dir, "source_s_16x16.npy"))
    
    print(f"  加载成功: 12个文件，形状示例: {source_t_64.shape}")
    
    # 转换为torch张量
    source_t_64_t = torch.from_numpy(source_t_64)
    source_t_32_t = torch.from_numpy(source_t_32)
    source_t_16_t = torch.from_numpy(source_t_16)
    
    target_t_64_t = torch.from_numpy(target_t_64)
    target_t_32_t = torch.from_numpy(target_t_32)
    target_t_16_t = torch.from_numpy(target_t_16)
    
    target_s_64_t = torch.from_numpy(target_s_64)
    target_s_32_t = torch.from_numpy(target_s_32)
    target_s_16_t = torch.from_numpy(target_s_16)
    
    source_s_64_t = torch.from_numpy(source_s_64)
    source_s_32_t = torch.from_numpy(source_s_32)
    source_s_16_t = torch.from_numpy(source_s_16)
    
    # 在计算异常图之前添加批次维度（如果需要）
    # 特征图形状可能是(C, H, W)，需要变成(1, C, H, W)
    def add_batch_dim_if_needed(tensor):
        if len(tensor.shape) == 3:
            return tensor.unsqueeze(0)
        return tensor
    
    source_t_64_t = add_batch_dim_if_needed(source_t_64_t)
    source_t_32_t = add_batch_dim_if_needed(source_t_32_t)
    source_t_16_t = add_batch_dim_if_needed(source_t_16_t)
    
    target_t_64_t = add_batch_dim_if_needed(target_t_64_t)
    target_t_32_t = add_batch_dim_if_needed(target_t_32_t)
    target_t_16_t = add_batch_dim_if_needed(target_t_16_t)
    
    target_s_64_t = add_batch_dim_if_needed(target_s_64_t)
    target_s_32_t = add_batch_dim_if_needed(target_s_32_t)
    target_s_16_t = add_batch_dim_if_needed(target_s_16_t)
    
    source_s_64_t = add_batch_dim_if_needed(source_s_64_t)
    source_s_32_t = add_batch_dim_if_needed(source_s_32_t)
    source_s_16_t = add_batch_dim_if_needed(source_s_16_t)
    
    # 根据uninet.py中的逻辑，需要计算1 - F.cosine_similarity(t, s)
    # 从代码中可以看到，t_tf包含6个特征图：前3个是source_t，后3个是target_t
    # de_features包含6个特征图：前3个是target_s，后3个是source_s
    # 所以需要计算：
    # 1. 1 - F.cosine_similarity(source_t, target_s)  # 对应前3个尺度
    # 2. 1 - F.cosine_similarity(target_t, source_s)  # 对应后3个尺度
    
    # 初始化输出列表，模拟uninet.py中的output_list
    # 注意：output_list应该是列表的列表，每个子列表对应一个尺度的所有图像输出
    # 对于单张图片，我们需要将每个尺度的输出包装在列表中
    output_list = [[] for _ in range(6)]
    
    # 计算前3个尺度：source_t 与 target_s 的相似度
    output_list[0].append(1 - F.cosine_similarity(source_t_64_t, target_s_64_t))
    output_list[1].append(1 - F.cosine_similarity(source_t_32_t, target_s_32_t))
    output_list[2].append(1 - F.cosine_similarity(source_t_16_t, target_s_16_t))
    
    # 计算后3个尺度：target_t 与 source_s 的相似度
    output_list[3].append(1 - F.cosine_similarity(target_t_64_t, source_s_64_t))
    output_list[4].append(1 - F.cosine_similarity(target_t_32_t, source_s_32_t))
    output_list[5].append(1 - F.cosine_similarity(target_t_16_t, source_s_16_t))
    
    print(f"  计算了 {len(output_list)} 个尺度的异常图")
    
    # 实现weighted_decision_mechanism逻辑（根据mechanism.py）
    # 参数设置
    num = 1  # 单张图片
    alpha = 0.01  # 默认值，根据uninet.py中的c.alpha
    beta = 0.00003  # 默认值，根据uninet.py中的c.beta
    out_size = 256  # 输出尺寸
    
    # 步骤1: 计算权重列表
    total_weights_list = []
    for i in range(num):
        low_similarity_list = []
        for j in range(len(output_list)):
            # 获取每个尺度的最大异常值
            low_similarity_list.append(torch.max(output_list[j][i]).cpu())
        
        # 计算softmax概率
        probs = F.softmax(torch.tensor(low_similarity_list), 0)
        
        # 选择高概率值（大于平均概率）
        weight_list = []
        for idx, prob in enumerate(probs):
            if prob > torch.mean(probs):
                weight_list.append(low_similarity_list[idx].numpy())
        
        # 计算权重：max(mean(weight_list) * alpha, beta)
        if weight_list:
            weight = np.max([np.mean(weight_list) * alpha, beta])
        else:
            weight = beta  # 如果没有高概率值，使用beta
        
        total_weights_list.append(weight)
    
    # 步骤2: 计算每个尺度的异常图并上采样
    am_lists = [list() for _ in output_list]
    for l, output in enumerate(output_list):
        # 将列表中的张量拼接
        output_tensor = torch.cat(output, dim=0)  # 形状: [B, H, W]
        a_map = torch.unsqueeze(output_tensor, dim=1)  # B*1*h*w
        # 上采样到out_size
        am_lists[l] = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)[:, 0, :, :]  # B*256*256
    
    # 步骤3: 聚合所有尺度的异常图
    anomaly_map = sum(am_lists)  # 形状: [B, out_size, out_size]
    
    # 步骤4: 计算异常分数
    anomaly_score_exp = anomaly_map  # 使用原始异常图
    batch = anomaly_score_exp.shape[0]
    anomaly_score = []  # 所有测试样本的异常分数
    
    for b in range(batch):
        # 计算top_k值
        top_k = int(out_size * out_size * total_weights_list[b])
        assert top_k >= 1 / (out_size * out_size), "weight不能小于1/(H*W)!"
        
        # 获取当前样本的异常图
        single_anomaly_score_exp = anomaly_score_exp[b]
        
        # 应用高斯滤波（sigma=4）
        single_anomaly_score_exp = torch.tensor(gaussian_filter(
            single_anomaly_score_exp.detach().cpu().numpy(), sigma=4
        ))
        
        # 重塑为一维
        single_map = single_anomaly_score_exp.reshape(1, -1)
        
        # 取top_k个最大值并计算平均
        single_anomaly_score = np.mean(single_map.topk(top_k, dim=-1)[0].detach().cpu().numpy(), axis=1)
        anomaly_score.append(single_anomaly_score[0])  # 转换为标量
    
    # 对于单张图片，取第一个（也是唯一一个）异常分数
    prediction_score = float(anomaly_score[0]) if anomaly_score else 0.0
    
    print(f"  预测分数: {prediction_score:.10f}")
    
    # 如果需要保存处理结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.basename(model_output_path)
        save_path = os.path.join(save_dir, f"uninet_{basename}")
        
        # 保存异常图
        anomaly_map_np = anomaly_map[0, :, :].cpu().detach().numpy()
        np.save(save_path, anomaly_map_np)
        print(f"  保存处理结果到: {save_path}")
    
    return {
        "model": "UniNet",
        "input_shape": f"12个特征图，形状示例: {source_t_64.shape}",
        "output_shape": (1,),  # 标量分数
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(base_dir),
        "num_scales": len(output_list),
        "out_size": out_size,
        "alpha": alpha,
        "beta": beta
    }

def process_urd(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理URD模型输出（根据student.py中的cal_anomaly_map方法实现）：
    
    1. URD模型输出包含6个npy文件（teacher_64x64.npy, teacher_32x32.npy, teacher_16x16.npy,
       student_64x64.npy, student_32x32.npy, student_16x16.npy）
    2. 根据student.py中的cal_anomaly_map方法计算异常图：
       - 对于每个尺度，计算1 - F.cosine_similarity(teacher, student)
       - 上采样到256x256
       - 根据anomaly_mode聚合（默认相加模式）
    3. 应用高斯滤波（sigma=4），与urd.py中的gaussian_filter一致
    4. 从异常图中提取最大分数作为预测分数
    
    Args:
        model_output_path: 模型输出文件路径（任意一个npy文件）
        save_dir: 保存目录（可选）
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理URD模型: {model_output_path}")
    
    # 获取文件所在目录
    base_dir = os.path.dirname(model_output_path)
    
    # 检查目录中是否有URD的6个输出文件
    required_files = [
        "teacher_64x64.npy",
        "teacher_32x32.npy", 
        "teacher_16x16.npy",
        "student_64x64.npy",
        "student_32x32.npy",
        "student_16x16.npy"
    ]
    
    # 验证所有必需文件都存在
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            print(f"  警告: 缺少文件 {file_name}")
    
    # 加载teacher特征图
    teacher_64 = np.load(os.path.join(base_dir, "teacher_64x64.npy"))
    teacher_32 = np.load(os.path.join(base_dir, "teacher_32x32.npy"))
    teacher_16 = np.load(os.path.join(base_dir, "teacher_16x16.npy"))
    
    # 加载student特征图
    student_64 = np.load(os.path.join(base_dir, "student_64x64.npy"))
    student_32 = np.load(os.path.join(base_dir, "student_32x32.npy"))
    student_16 = np.load(os.path.join(base_dir, "student_16x16.npy"))
    
    print(f"  加载成功: 6个文件，形状: {teacher_64.shape}, {teacher_32.shape}, {teacher_16.shape}")
    
    # 转换为torch张量
    teacher_64_t = torch.from_numpy(teacher_64)
    teacher_32_t = torch.from_numpy(teacher_32)
    teacher_16_t = torch.from_numpy(teacher_16)
    
    student_64_t = torch.from_numpy(student_64)
    student_32_t = torch.from_numpy(student_32)
    student_16_t = torch.from_numpy(student_16)

    # 在计算异常图之前添加批次维度
    # 特征图形状可能是(C, H, W)，需要变成(1, C, H, W)
    if len(teacher_64_t.shape) == 3:
        teacher_64_t = teacher_64_t.unsqueeze(0)
        teacher_32_t = teacher_32_t.unsqueeze(0)
        teacher_16_t = teacher_16_t.unsqueeze(0)
        
        student_64_t = student_64_t.unsqueeze(0)
        student_32_t = student_32_t.unsqueeze(0)
        student_16_t = student_16_t.unsqueeze(0)
    
    # 根据student.py中的cal_anomaly_map方法实现
    # 注意：这里使用默认的相加模式（anomaly_mode='add'）
    # 如果需要支持mul模式，可以从文件路径或配置中获取anomaly_mode
    anomaly_mode = 'add'  # 默认使用相加模式，与student.py中的默认值一致
    
    # 初始化融合的异常图
    anomaly_map_fuse = None
    img_size = 256  # 默认图像大小，与student.py中的默认值一致
    
    # 三个尺度的特征图列表
    teacher_features = [teacher_64_t, teacher_32_t, teacher_16_t]
    student_features = [student_64_t, student_32_t, student_16_t]
    
    # 遍历每个尺度
    for i in range(len(teacher_features)):
        # 计算当前尺度的异常图：1 - F.cosine_similarity(feature_s[i], feature_t[i])
        anomaly_map = torch.unsqueeze(1 - F.cosine_similarity(student_features[i], teacher_features[i]), dim=1)
        
        # 上采样到img_size
        anomaly_map = F.interpolate(anomaly_map, size=img_size, mode='bilinear', align_corners=True)
            
        if anomaly_map_fuse is None:
                anomaly_map_fuse = anomaly_map
                continue

        anomaly_map_fuse = anomaly_map_fuse + anomaly_map
    
    # 将异常图转换为numpy数组
    anomaly_map_np = anomaly_map_fuse[0, 0, :, :].cpu().detach().numpy()
    
    # 应用高斯滤波（sigma=4），与urd.py中的gaussian_filter一致
    anomaly_map_smoothed = gaussian_filter(anomaly_map_np, sigma=4)
    
    # 提取最大分数作为预测分数
    prediction_score = float(np.max(anomaly_map_smoothed))
    
    print(f"  预测分数: {prediction_score:.10f}")
    print(f"  异常图聚合模式: {anomaly_mode}")

    
    return {
        "model": "URD",
        "input_shape": f"多尺度特征: {teacher_64.shape}, {teacher_32.shape}, {teacher_16.shape}",
        "output_shape": anomaly_map_smoothed.shape,
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(base_dir),
        "anomaly_map_shape": anomaly_map_smoothed.shape,
        "anomaly_mode": anomaly_mode,
        "img_size": img_size
    }

def process_ml_destseg(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理ml-destseg-new模型输出
    
    根据用户提供的后处理代码实现：
    1. 插值到目标大小
    2. 排序并取前T个值的平均
    
    Args:
        model_output_path: 模型输出文件路径
        save_dir: 保存目录
        
    Returns:
        处理结果字典
    """
    print(f"开始处理ml-destseg模型: {model_output_path}")
    
    target_size = (256, 256)  # 目标大小 (H, W)
    T = 100  # 取前T个值计算平均
    
    # 加载数据
    data = np.load(model_output_path)
    print(f"  输入形状: {data.shape}")
    
    # 严格检查形状是否为 [1, 64, 64]
    if data.shape != (1, 64, 64):
        raise ValueError(f"形状错误: 期望 (1, 64, 64)，但得到 {data.shape}")
    
    # 确保数据是4D的 [B, C, H, W]
    data_tensor = torch.from_numpy(data).unsqueeze(0)

    #插值到目标大小（如果尺寸不同）
    current_size = data_tensor.size()[2:]  # 当前H, W
    print(f"  插值: {current_size} -> {target_size}")
    data_tensor = F.interpolate(
        data_tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    
    # 排序并取前T个值的平均
    # 将4D张量重塑为2D [B, C*H*W]
    data_flat = data_tensor.view(data_tensor.size(0), -1)
    
    # 降序排序
    sorted_data, _ = torch.sort(data_flat, dim=1, descending=True)
    
    # 取前T个值的平均
    processed_tensor = torch.mean(sorted_data[:, :T], dim=1)
    
    # 转换回numpy
    processed = processed_tensor.cpu().numpy()
    
    # 提取预测分数
    prediction_score = float(processed)
    
    print(f"  预测分数: {prediction_score:.10f}")
    
    return {
        "model": "ml-destseg",
        "input_shape": data.shape,
        "output_shape": processed.shape,
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(model_output_path),
        "target_size": target_size,
        "T": T
    }

def process_msflow(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理MSFlow模型输出
    
    根据msflow.py中的post_process函数逻辑：
    1. MSFlow输出三个尺度的z特征：z_64x64.npy, z_32x32.npy, z_16x16.npy
    2. 对于每个尺度的z，计算logp = -0.5 * torch.mean(z**2, 1)
    3. 对每个尺度的logp进行上采样到input_size（512x512）
    4. 计算异常分数图：mul（相乘）
    5. 计算最终的anomaly_score（取top_k个值的平均）
    
    Args:
        model_output_path: 模型输出文件路径（任意一个npy文件）
        save_dir: 保存目录
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理MSFlow模型: {model_output_path}")
    
    # 获取文件所在目录
    base_dir = os.path.dirname(model_output_path)
    
    # 检查目录中是否有MSFlow的3个输出文件
    required_files = [
        "z_64x64.npy",
        "z_32x32.npy", 
        "z_16x16.npy"
    ]
    
    # 验证所有必需文件都存在
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
    
    # 加载三个尺度的z特征
    z_64 = np.load(os.path.join(base_dir, required_files[0]))
    z_32 = np.load(os.path.join(base_dir, required_files[1]))
    z_16 = np.load(os.path.join(base_dir, required_files[2]))
    
    # 转换为torch张量
    z_64_t = torch.from_numpy(z_64)
    z_32_t = torch.from_numpy(z_32)
    z_16_t = torch.from_numpy(z_16)
    
    # 确保所有张量都是4D [B, C, H, W]
    if len(z_64_t.shape) == 3:
        z_64_t = z_64_t.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    if len(z_32_t.shape) == 3:
        z_32_t = z_32_t.unsqueeze(0)
    if len(z_16_t.shape) == 3:
        z_16_t = z_16_t.unsqueeze(0)
    
    # 根据inference_meta_epoch函数中的逻辑计算logp
    # logp = -0.5 * torch.mean(z**2, 1)
    def compute_logp(z_tensor):
        """计算log概率图"""
        # z_tensor形状: [B, C, H, W]
        # 计算每个空间位置的log概率
        logp = -0.5 * torch.mean(z_tensor**2, dim=1)  # [B, H, W]
        return logp
    
    # 计算每个尺度的logp
    logp_64 = compute_logp(z_64_t)
    logp_32 = compute_logp(z_32_t)
    logp_16 = compute_logp(z_16_t)
    
    # 获取每个尺度的空间尺寸
    _, _, H64, W64 = z_64_t.shape
    _, _, H32, W32 = z_32_t.shape
    _, _, H16, W16 = z_16_t.shape
    
    # MSFlow参数
    input_size = (512, 512)  # 输入图像尺寸
    top_k = 0.03 # 取top_k比例的值计算平均
    
    # 构建outputs_list，模拟post_process函数的输入
    outputs_list = [[logp_64], [logp_32], [logp_16]]
    
    # 初始化累加器
    logp_sum = None
    #prop_sum = None
    
    for i, outputs in enumerate(outputs_list):
        # 处理每个尺度的输出
        outputs_tensor = torch.cat(outputs, 0).float()
        
        # 计算logp_map (增量式处理)
        current_logp = F.interpolate(
            outputs_tensor.unsqueeze(1),
            size=input_size,
            mode='bilinear',
            align_corners=True
        ).squeeze(1)
        
        # 计算prob_map (增量式处理)
        output_norm = outputs_tensor - outputs_tensor.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        current_prob = torch.exp(output_norm)
        current_prop = F.interpolate(
            current_prob.unsqueeze(1),
            size=input_size,
            mode='bilinear',
            align_corners=True
        ).squeeze(1)
        
        # 增量累加
        if logp_sum is None:
            logp_sum = current_logp
            #prop_sum = current_prop
        else:
            logp_sum += current_logp
            #prop_sum += current_prop
        
        # 最终计算
        # 计算mul结果
        logp_sum_norm = logp_sum - logp_sum.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        prop_map_mul = torch.exp(logp_sum_norm)
        anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
        
        # # 计算add结果
        # prop_map_add = prop_sum.cpu().numpy()
        # anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add
        
        # 计算anomaly_score (取top_k个值的平均)
        batch_size = anomaly_score_map_mul.shape[0]
        top_k_pixels = int(input_size[0] * input_size[1] * top_k)
        
        # 直接计算整个batch的anomaly_score，不进行分块
        # 取top_k个最大值并计算平均
        anomaly_score_tensor = torch.mean(
            anomaly_score_map_mul.reshape(batch_size, -1).topk(top_k_pixels, dim=-1)[0],
            dim=1
        )
        
        prediction_score = float(anomaly_score_tensor[0].item())  # 假设batch_size=1
        
        # 最后转换mul结果到numpy
        anomaly_score_map_mul_np = anomaly_score_map_mul.cpu().numpy()
    
    print(f"  预测分数: {prediction_score:.10f}")
    
    return {
        "model": "MSFlow",
        "input_shape": f"多尺度特征: {z_64.shape}, {z_32.shape}, {z_16.shape}",
        "output_shape": anomaly_score_map_mul_np.shape,
        "prediction_score": prediction_score,
        "file_path": model_output_path,
        "filename": get_filename_from_path(base_dir),
        "input_size": input_size,
        "top_k": top_k
    }

def process_patchcore(model_output_path: str, save_dir: str) -> Dict[str, Any]:
    """
    处理patchcore模型输出
    
    根据patchcore.py中的注释代码实现完整的后处理流程：
    1. 加载三个模型的patch分数数组
    2. 对每个模型应用完整的后处理：
       a. unpatch_scores: 将一维数组重塑为(batchsize, -1, *x.shape[1:])
       b. reshape: x.reshape(*x.shape[:2], -1)
       c. score: self.patch_maker.score(x)
    3. 返回每个模型的图像级分数，用于后续的跨图像归一化
    
    Args:
        model_output_path: 模型输出文件路径（单个文件，如/path/to/densenet201.npy）
        save_dir: 保存目录
        
    Returns:
        处理结果字典，包含预测分数
    """
    print(f"处理patchcore模型: {model_output_path}")
    
    # 获取文件所在目录
    base_dir = os.path.dirname(model_output_path)
    
    # 检查目录中是否有patchcore的3个模型文件
    required_files = [
        "wideresnet101.npy",
        "resnext101.npy", 
        "densenet201.npy"
    ]
    
    # 验证所有必需文件都存在
    model_files = []
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            model_files.append(file_path)
        else:
            print(f"  警告: 缺少文件 {file_name}")
    
    # 加载所有模型的数据
    all_scores = []
    for i, file_path in enumerate(model_files):
        data = np.load(file_path)
        print(f"  模型{i}: {os.path.basename(file_path)}, 形状: {data.shape}")
        all_scores.append(data)
    
    # 实现完整的PatchMaker后处理流程
    class PatchMaker:
        def __init__(self, patchsize=3, stride=None):
            self.patchsize = patchsize
            self.stride = stride
        
        def unpatch_scores(self, x, batchsize):
            """模拟PatchMaker.unpatch_scores方法"""
            # 源代码：return x.reshape(batchsize, -1, *x.shape[1:])
            return x.reshape(batchsize, -1, *x.shape[1:])
        
        def score(self, x):
            """模拟PatchMaker.score方法"""
            # 转换为torch张量以模拟源代码
            was_numpy = False
            if isinstance(x, np.ndarray):
                was_numpy = True
                x_tensor = torch.from_numpy(x)
            else:
                x_tensor = x
            
            # 源代码逻辑
            while x_tensor.ndim > 1:
                x_tensor = torch.max(x_tensor, dim=-1).values
            
            if was_numpy:
                return x_tensor.numpy()
            return x_tensor
    
    # 创建PatchMaker实例
    patch_maker = PatchMaker(patchsize=3, stride=1)
    
    # 对每个模型的patch分数应用完整的后处理
    image_level_scores = []
    for i, scores in enumerate(all_scores):
        batchsize = 1  # 单张图片
        
        # 步骤1: unpatch_scores
        # 对于一维数组，需要先添加一个通道维度
        if len(scores.shape) == 1:
            scores_with_channel = scores.reshape(-1, 1)  # (num_patches, 1)
        else:
            scores_with_channel = scores
        
        # unpatch_scores: reshape(batchsize, -1, *x.shape[1:])
        unpatch_scores = patch_maker.unpatch_scores(scores_with_channel, batchsize=batchsize)
        
        # 步骤2: reshape: x.reshape(*x.shape[:2], -1)
        # 对于(batchsize, num_patches, 1) -> (batchsize, num_patches)
        reshaped_scores = unpatch_scores.reshape(batchsize, -1)
        
        # 步骤3: score
        image_score_array = patch_maker.score(reshaped_scores)  # 形状: (batchsize,)
        image_score = float(image_score_array[0])  # 转换为标量
        
        image_level_scores.append(image_score)
        print(f"  模型{i}图像级分数: {image_score:.6f} (来自{scores.shape[0]}个patch)")
    
    image_level_scores_array = np.array(image_level_scores)  # 形状: (num_models,)
    print(f"  图像级分数数组: {image_level_scores_array}")
    
    # 返回每个模型的原始分数，跨图像归一化将在process_all_models中进行
    print(f"  返回每个模型的原始分数，跨图像归一化将在process_all_models中进行")
    
    return {
        "model": "patchcore",
        "input_shape": all_scores[0].shape,
        "output_shape": (1,),  # 标量分数
        "prediction_score": 0.0,  # 设为0.0，因为最终分数应该在process_all_models中计算
        "file_path": model_output_path,
        "filename": get_filename_from_path(base_dir),  # 传递图片文件夹路径
        "num_models": len(all_scores),
        "image_level_scores": image_level_scores_array.tolist(),
        "raw_scores": image_level_scores_array.tolist()  # 保存每个模型的原始分数，用于跨图像归一化
    }

def normalize_patchcore_scores_correctly(all_raw_scores: List[List[float]]) -> List[float]:
    """
    对patchcore分数进行正确的跨图像归一化（根据源代码逻辑）
    
    源代码逻辑:
    1. scores形状: (num_models, num_images) = (3, num_images)
    2. 对每个模型单独归一化: min_scores = scores.min(axis=-1).reshape(-1, 1)
    3. max_scores = scores.max(axis=-1).reshape(-1, 1)
    4. scores = (scores - min_scores) / (max_scores - min_scores)
    5. scores = np.mean(scores, axis=0)
    
    Args:
        all_raw_scores: 每个图像的原始分数列表，每个元素是一个列表，包含3个模型的分数
        
    Returns:
        归一化后的分数列表
    """
    if not all_raw_scores:
        return []
    
    import numpy as np
    
    # 转换为numpy数组，形状为(num_images, num_models)
    # 需要转置为(num_models, num_images)
    num_images = len(all_raw_scores)
    num_models = len(all_raw_scores[0]) if all_raw_scores else 0
    
    if num_models == 0:
        return []
    
    # 创建scores数组，形状为(num_models, num_images)
    scores = np.zeros((num_models, num_images))
    for i, image_scores in enumerate(all_raw_scores):
        for j, model_score in enumerate(image_scores):
            scores[j, i] = model_score
    
    print(f"  归一化前分数形状: {scores.shape} (num_models={num_models}, num_images={num_images})")
    print(f"  每个模型的原始分数范围:")
    for i in range(num_models):
        print(f"    模型{i}: [{scores[i].min():.6f}, {scores[i].max():.6f}]")
    
    # 对每个模型单独归一化
    min_scores = scores.min(axis=-1).reshape(-1, 1)  # 形状: (num_models, 1)
    max_scores = scores.max(axis=-1).reshape(-1, 1)  # 形状: (num_models, 1)
    
    # 避免除零
    denominator = max_scores - min_scores
    denominator[denominator == 0] = 1.0  # 如果所有分数相同，避免除零
    
    # 归一化
    normalized_scores = (scores - min_scores) / denominator
    
    print(f"  每个模型的归一化后分数范围:")
    for i in range(num_models):
        print(f"    模型{i}: [{normalized_scores[i].min():.6f}, {normalized_scores[i].max():.6f}]")
    
    # 取所有模型的平均值
    final_scores = np.mean(normalized_scores, axis=0)  # 形状: (num_images,)
    
    print(f"  最终分数范围: [{final_scores.min():.6f}, {final_scores.max():.6f}]")
    
    return final_scores.tolist()



def process_model_by_name(model_name: str, model_output_path: str, save_dir: str, **kwargs) -> Dict[str, Any]:
    """
    根据模型名称选择对应的处理函数
    
    Args:
        model_name: 模型名称
        model_output_path: 模型输出文件路径
        save_dir: 保存目录（现在可选，因为只返回预测分数）
        **kwargs: 额外参数，传递给特定模型的处理函数
        
    Returns:
        处理结果字典
    """
    # 只有在save_dir不为空时才创建目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 使用if/elif处理不同模型
    if model_name == "comet":
        return process_comet(model_output_path, save_dir)
    elif model_name == "cflow_ad":
        # 使用全局归一化模式（每个类别的一层最大值）
        return process_cflow_ad_global(model_output_path, save_dir)
    elif model_name == "deco_diff":
        return process_deco_diff(model_output_path, save_dir)
    elif model_name == "dinomaly":
        return process_dinomaly(model_output_path, save_dir)
    elif model_name == "dsr":
        return process_dsr(model_output_path, save_dir)
    elif model_name == "rd4ad":
        return process_rd4ad(model_output_path, save_dir)
    elif model_name == "rrd":
        return process_rrd(model_output_path, save_dir)
    elif model_name == "simplenet":
        return process_simplenet(model_output_path, save_dir)
    elif model_name == "uninet":
        return process_uninet(model_output_path, save_dir)
    elif model_name == "urd":
        return process_urd(model_output_path, save_dir)
    elif model_name == "ml_destseg":
        return process_ml_destseg(model_output_path, save_dir)
    elif model_name == "msflow":
        return process_msflow(model_output_path, save_dir)
    elif model_name == "patchcore":
        return process_patchcore(model_output_path, save_dir)
    else:
        raise ValueError(f"未知模型: {model_name}")


def find_model_outputs(base_dir: str, model_name: str) -> Dict[str, List[str]]:
    """
    查找模型输出文件（总是按MVTec-AD类别返回结果）
    
    Args:
        base_dir: 基础目录
        model_name: 模型名称
        
    Returns:
        字典{类别名: 文件路径列表}
    """
    # MVTec-AD的15个类别
    # mvtec_classes = [
    #     "bottle", "cable", "capsule", "carpet", "grid",
    #     "hazelnut", "leather", "metal_nut", "pill", "screw",
    #     "tile", "toothbrush", "transistor", "wood", "zipper"
    # ]
    mvtec_classes = ["toothbrush"]
    
    # 根据模型名称构建搜索模式
    # 注意：每个模型的前缀路径不同，但文件夹层级结构相同
    # 示例1: /mnt/T4_1/xjj/destseg_test/FirstWorkData_numpy/bottle/test/good/00011.npy
    # 示例2: /mnt/T4_1/xjj/Dinomaly/FirstWorkData_numpy/bottle/test/good/0001/1.npy
    patterns = {
        "cflow_ad": "T4_1/xjj/cflow-ad/MVTec-AD_numpy/{cls}/*/*/*/*.npy",          # /mnt/T4_1/xjj/destseg/MVTec-AD_numpy/bottle/test/good/000/*.npy
        "dinomaly": "T4_1/xjj/Dinomaly/MVTec-AD_numpy/{cls}/*/*/*/*.npy",          # /mnt/T4_1/xjj/Dinomaly/MVTec-AD_numpy/bottle/test/good/000/*.npy
        "dsr": "T4_2/xjj/DSR/MVTec-AD_numpy/{cls}/*/*/*.npy",                      # /mnt/T4_2/xjj/DSR/MVTec-AD_numpy/bottle/test/good/000.npy
        "rd4ad": "T4_2/xjj/rd4ad/MVTec-AD_numpy/{cls}/*/*/*/*.npy",                # /mnt/T4_2/xjj/rd4ad/MVTec-AD_numpy/bottle/test/good/000/*.npy
        "rrd": "T4_1/xjj/RRD/MVTec-AD_numpy/{cls}/*/*/*/*.npy",                    # /mnt/T4_2/xjj/rrd/MVTec-AD_numpy/bottle/test/good/000/*.npy
        "simplenet": "T4_1/xjj/SimpleNet/MVTec-AD_numpy/{cls}/*/*/*.npy",          # /mnt/T4_1/xjj/SimpleNet/MVTec-AD_numpy/bottle/test/good/000.npy
        "uninet": "T4_2/xjj/UniNet/MVTec-AD_numpy/{cls}/*/*/*/*.npy",              # /mnt/T4_2/xjj/UniNet/MVTec-AD_numpy/bottle/test/good/000/*.npy
        "urd": "T4_2/xjj/URD/MVTec-AD_numpy/{cls}/*/*/*/*.npy",                    # /mnt/T4_2/xjj/urd/MVTec-AD_numpy/bottle/test/good/000/*.npy
        "ml_destseg": "T4_1/xjj/destseg/MVTec-AD_numpy/{cls}/*/*/*.npy",           # /mnt/T4_1/xjj/destseg/MVTec-AD_numpy/bottle/test/good/000.npy
        "msflow": "T4_1/xjj/msflow/MVTec-AD_numpy/{cls}/*/*/*/*.npy",              # /mnt/T4_1/xjj/msflow/MVTec-AD_numpy/bottle/test/good/000/*.npy
        "patchcore": "T4_2/xjj/patchcore/MVTec-AD_numpy/{cls}/*/*/*/*.npy",        # /mnt/T4_1/xjj/patchcore/MVTec-AD_numpy/bottle/test/good/000/*.npy
        "comet": "T4_1/xjj/CoMet/MVTec-AD_numpy/{cls}/*/*/*.npy",                  # /mnt/T4_1/xjj/CoMet/MVTec-AD_numpy/bottle/test/good/000.npy
        "deco_diff": "T4_1/xjj/DeCo-Diff/MVTec-AD_numpy/{cls}/*/*/*/*.npy",        # /mnt/T4_1/xjj/DeCo-Diff/MVTec-AD_numpy/bottle/test/good/000/*.npy
    }
    
    pattern_template = patterns.get(model_name, "*/*/{cls}/*/*/*.npy")
    
    # 按类别返回结果
    class_files = {}
    total_files = 0
    
    for class_name in mvtec_classes:
        # 替换{cls}为实际类别名
        pattern = pattern_template.format(cls=class_name)
        search_path = os.path.join(base_dir, pattern)
        
        # 使用glob查找所有匹配的npy文件
        # 支持多层目录和多个npy文件
        files = glob.glob(search_path, recursive=True)
        
        # 如果没有找到文件，尝试更通用的搜索模式
        if not files:
            # 尝试查找目录下的所有npy文件（包括子目录）
            alt_pattern = os.path.join(base_dir, class_name, "*", "*", "*.npy")
            files = glob.glob(alt_pattern, recursive=True)
        
        if files:
            # 按完整路径的数字顺序排序，确保样本按正确顺序加载
            def sort_key(filepath):
                # 提取路径中的数字部分进行排序
                # 例如：/mnt/T4_1/xjj/cflow-ad/MVTec-AD_numpy/zipper/test/broken_teeth/007/log_jac_det_reshaped_layer0_64x64.npy
                # 我们需要提取最后的数字部分（如007）
                path_parts = filepath.split(os.sep)
                
                # 从路径末尾开始查找数字部分
                for part in reversed(path_parts):
                    # 尝试提取数字
                    import re
                    match = re.search(r'(\d+)', part)
                    if match:
                        return int(match.group(1))
                
                # 如果没有找到数字，按完整路径排序
                return filepath
            
            files = sorted(files, key=sort_key)

            # 对于CFLOW-AD、MSFlow和PatchCore模型，我们需要特殊处理
            # 这些模型每张图片有多个文件，但我们只需要处理每个目录/文件夹中的一个文件
            # 因为对应的处理函数会处理目录中的所有相关文件
            if model_name in ["cflow_ad", "msflow", "patchcore", "rrd", "rd4ad", "urd", "uninet","dinomaly","deco_diff"]:
                # 去重：只保留每个目录中的一个文件，同时保持排序顺序
                unique_dirs = {}
                
                for file_path in files:
                    dir_path = os.path.dirname(file_path)
                    if dir_path not in unique_dirs:
                        unique_dirs[dir_path] = file_path
                
                # 重新按目录路径中的数字排序
                def dir_sort_key(dir_file_pair):
                    dir_path, file_path = dir_file_pair
                    # 提取目录路径中的数字部分
                    path_parts = dir_path.split(os.sep)
                    for part in reversed(path_parts):
                        import re
                        match = re.search(r'(\d+)', part)
                        if match:
                            return int(match.group(1))
                    return dir_path
                
                # 按目录排序后提取文件
                sorted_dirs = sorted(unique_dirs.items(), key=dir_sort_key)
                files = [file_path for _, file_path in sorted_dirs]
                
                # 根据模型类型显示不同的消息
                if model_name == "cflow_ad":
                    print(f"  类别 {class_name}: 去重后 {len(files)} 个目录（原 {len(unique_dirs)*6} 个文件）")
                elif model_name == "msflow":
                    print(f"  类别 {class_name}: 去重后 {len(files)} 个目录（原 {len(unique_dirs)*3} 个文件）")
                elif model_name == "patchcore":
                    print(f"  类别 {class_name}: 去重后 {len(files)} 个图片文件夹（原 {len(unique_dirs)*3} 个模型文件）")
            else:
                print(f"  类别 {class_name}: {len(files)} 个文件")
            
            class_files[class_name] = files
            total_files += len(files)
            
            # 显示前几个文件的示例
            if len(files) > 0:
                sample_file = files[0]
                # 只显示相对路径部分，避免太长
                rel_path = os.path.relpath(sample_file, base_dir) if os.path.exists(sample_file) else sample_file
                if len(rel_path) > 80:
                    rel_path = "..." + rel_path[-77:]
                print(f"    示例: {rel_path}")
    
    print(f"找到 {model_name} 的 {total_files} 个文件（按{len(class_files)}个类别分组）")
    
    return class_files



def process_all_models(base_dir: str, save_base_dir: str) -> None:
    """
    处理所有模型并保存为CSV表格
    
    Args:
        base_dir: 基础目录（包含所有模型输出）
        save_base_dir: 保存基础目录
    """
    #正确的
    # model_names = [
    #     "ml_destseg","dsr","simplenet","rd4ad","msflow","urd","uninet","comet","deco_diff","rrd","dinomaly","patchcore"
    # ]

    model_names = [
        "cflow_ad"
    ]



    for model_name in model_names:
        print(f"\n开始处理模型: {model_name}")
        
        # 按类别查找文件
        class_files = find_model_outputs(base_dir, model_name)
        
        if not class_files:
            print(f"  警告: 未找到 {model_name} 的文件")
            continue
        
        # 收集所有文件
        all_file_paths = []
        all_class_names = []

        all_raw_scores = []  # 用于存储每个图像的原始分数（每个模型的分数）
        
        for class_name, files in class_files.items():
            print(f"  类别: {class_name} ({len(files)} 个文件)")
            all_file_paths.extend(files)
            all_class_names.extend([class_name] * len(files))
        
        print(f"  总共 {len(all_file_paths)} 个文件需要处理")
        
        # 处理文件并收集结果
        processed_names = []
        processed_scores = []
        
        for i, (file_path, class_name) in enumerate(zip(all_file_paths, all_class_names), 1):
            # 每100个文件打印一次进度
            if i % 100 == 0 or i == len(all_file_paths):
                print(f"    进度: {i}/{len(all_file_paths)}")
            
            try:
                # 不再需要保存目录，因为只返回预测分数
                # 传递None或空字符串作为save_dir
                result = process_model_by_name(model_name, file_path, "")
                
                # 收集预测分数和文件名
                if "prediction_score" in result:
                    # 使用处理函数返回的filename字段
                    # 对于CFLOW-AD，这会返回图片ID（如bottle_test_good_000）
                    # 对于其他模型，这会返回从文件路径提取的文件名
                    processed_name = result.get("filename", get_filename_from_path(file_path))
                    processed_names.append(processed_name)
                    
                    # 对于patchcore，我们使用raw_scores而不是prediction_score
                    if model_name == "patchcore" and "raw_scores" in result:
                        # 收集每个模型的原始分数
                        all_raw_scores.append(result["raw_scores"])
                        # 对于patchcore，prediction_score是0.0，所以不添加到processed_scores
                        # 我们将在后面使用raw_scores计算归一化后的分数
                    else:
                        # 对于其他模型，使用prediction_score
                        processed_scores.append(result["prediction_score"])
            
            except Exception as e:
                print(f"    失败: {os.path.basename(file_path)} - {e}")

        # 对于patchcore，使用raw_scores计算归一化后的分数
        if model_name == "patchcore":
            if all_raw_scores and len(all_raw_scores) == len(processed_names):
                print(f"  收集到 {len(all_raw_scores)} 个PatchCore原始分数（每个分数包含{len(all_raw_scores[0])}个模型）")
                
                # 使用正确的跨图像归一化（每个模型单独归一化）
                print(f"  使用正确的跨图像归一化（每个模型单独归一化）")
                normalized_scores = normalize_patchcore_scores_correctly(all_raw_scores)
                
                # 用归一化后的分数作为processed_scores
                processed_scores = normalized_scores
                print(f"  归一化后分数范围: [{min(processed_scores):.6f}, {max(processed_scores):.6f}]")
            else:
                print(f"  警告: 缺少每个模型的原始分数，无法进行正确的归一化")
                # 如果没有raw_scores，使用默认值
                processed_scores = [0.0] * len(processed_names)
        
        # 保存为CSV表格（文件名和分数一一对应后排序再保存）
        if processed_names and processed_scores:
            csv_filename = f"{model_name}_predictions.csv"
            csv_path = os.path.join(save_base_dir, csv_filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            # 将文件名和分数配对
            name_score_pairs = list(zip(processed_names, processed_scores))
            
            # 按照文件名排序
            name_score_pairs.sort(key=lambda x: x[0])
            
            # 保存排序后的结果
            with open(csv_path, 'w', encoding='utf-8') as f:
                # 写入表头
                f.write("File_Path,Prediction_Score\n")
                
                for name, score in name_score_pairs:
                    # 写入CSV行
                    f.write(f"{name},{score:.5f}\n")
            
            print(f"  预测结果保存到: {csv_path}")
            print(f"  共保存 {len(name_score_pairs)} 条记录（已按文件名排序）")
        
        print(f"  完成 {model_name}: 处理了 {len(processed_names)} 个文件")


def main():
    """主函数"""
    print("模型后处理")
    print("=" * 60)
    
    # 配置路径
    base_dir = "/mnt/"  # 模型输出基础目录   
    save_base_dir = "./postprocessed_results"  # 保存目录
    
    # 处理所有模型（现在只保存CSV表格，不返回结果）
    process_all_models(base_dir, save_base_dir)
    
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"结果保存在: {save_base_dir}")
    print("每个模型的预测结果保存为CSV表格:")
    print(f"  - ml_destseg_predictions.csv")


if __name__ == "__main__":
    main()