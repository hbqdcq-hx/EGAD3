import torch
import numpy as np
import random
import os
import pandas as pd
from argparse import ArgumentParser
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test_heatmap import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import MVTecDataset_test, get_data_transforms

import cv2
from torch.nn import functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder', default = '/mnt/T38/bioinf/xjj/CheckPoints/RD++_CP', type=str)
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--classes', nargs="+", default=['carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper'])
    pars = parser.parse_args()
    return pars

def get_filename_from_path(img_path):
    """
    从图像路径中提取文件名，使用路径的最后4层作为文件名
    
    Args:
        img_path: 图像路径
        
    Returns:
        str: 使用路径最后4层组成的文件名
    """
    import os
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

def inference(_class_, pars):
    if not os.path.exists(pars.checkpoint_folder):
        os.makedirs(pars.checkpoint_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    
    test_path = '/mnt/T38/bioinf/xjj/Datasets/MVTec-AD/' + _class_             #MVTec-AD/good_dataset
    #test_path = '/mnt/T4_2/xjj/FirstWorkData1/' + _class_ 

    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Use pretrained wide_resnet50 for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)

    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    # Load trained weights for projection layer, bn (OCBE), decoder (student)    
    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    ckp = torch.load(checkpoint_class, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
  
    #auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device)       
    auroc_sp,pr_list_sp,anomaly_maps = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device, _class_)   

    # 保存预测分数到CSV
    # 直接新定义一个变量接受地址
    save_dir = "/mnt/T4_1/xjj/2/RRD/MVTec-AD/"  # 直接定义保存目录 FirstWorkData1
    
    # 获取所有 img_paths
    img_paths = test_dataloader.dataset.img_paths
    # 归一化预测分数
    pr_list_sp_norm = (pr_list_sp - np.min(pr_list_sp)) / (np.max(pr_list_sp) - np.min(pr_list_sp))
    
    # 使用自定义函数生成文件名
    file_names = [get_filename_from_path(path) for path in img_paths]
    
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存预测分数到CSV，只包含图片名字和预测分数的一一对应
    csv_path = os.path.join(save_dir, f'anomaly_scores_{_class_}.csv')
    
    with open(csv_path, 'w') as f:
        # 写入表头
        f.write("File_Name,Anomaly_Score\n")
        # 使用zip确保一一对应
        for name, score in zip(file_names, pr_list_sp_norm):
            f.write(f"{name},{score:.10f}\n")  # 使用逗号分隔的表格格式
    
    print(f"异常分数已保存到: {csv_path}")

     # 创建类别文件夹
    class_dir = os.path.join(save_dir, _class_)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    print(f"开始保存热力图到: {class_dir}")

    count = 0
    for batch_idx, anomaly_map in enumerate(anomaly_maps):
        # 获取原始图像大小
        # 从数据集的img_paths中获取原始图像路径
        img_path = test_dataloader.dataset.img_paths[batch_idx]
        # 读取原始图像获取大小
        original_img = cv2.imread(img_path)
        if original_img is not None:
            orig_height, orig_width = original_img.shape[:2]
        else:
            # 如果无法读取原始图像，使用默认大小
            orig_height, orig_width = 256, 256
        
        
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

    #print('{}: Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(_class_, auroc_sp, auroc_px, aupro_px))
    print('{}: Sample Auroc: {:.4f}'.format(_class_, auroc_sp))
    #return auroc_sp, auroc_px, aupro_px
    return auroc_sp


if __name__ == '__main__':
    pars = get_args()

    item_list = [ 'carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
    setup_seed(111)
    #metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    metrics = {'class': [], 'AUROC_sample':[]}
    
    for c in pars.classes:
        #auroc_sp, auroc_px, aupro_px = inference(c, pars)
        auroc_sp = inference(c, pars)
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        #metrics['AUROC_pixel'].append(auroc_px)
        #metrics['AUPRO_pixel'].append(aupro_px)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(f'./metrics_checkpoints.csv', index=False)