import torch
import numpy as np
import random
import os
import csv
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from models.student import Student
from models.teacher import Teacher
from dataset.mvtec_ad import MVTecADTestDataset
from utils.evaluation import compute_pro

import cv2
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    
    print(f"开始保存热力图到: {class_dir}")
    
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

# def test_model(device, classname, data_root, log, img_size, ckp_path, results_csv='test_results.csv'):
#     """
#     单独测试模型并将结果保存到CSV文件
#     Args:
#         device: 使用的设备 (cuda/cpu)
#         classname: 测试的类别名称
#         data_root: 数据根目录
#         log: 日志文件对象
#         img_size: 图像大小
#         ckp_path: 模型检查点路径
#         results_csv: 结果保存的CSV文件名
#     """
def test_model(device, classname, data_root, log, img_size, ckp_path, results_csv='test_results.csv',
               save_predictions=True, save_heatmaps_flag=True, save_dir="/mnt/T4_1/xjj/2/URD/MVTec-AD/"):
    """
    单独测试模型并将结果保存到CSV文件
    Args:
        device: 使用的设备 (cuda/cpu)
        classname: 测试的类别名称
        data_root: 数据根目录
        log: 日志文件对象
        img_size: 图像大小
        ckp_path: 模型检查点路径
        results_csv: 结果保存的CSV文件名
        save_predictions: 是否保存预测分数到CSV文件
        save_heatmaps_flag: 是否保存热力图
        save_dir: 保存预测分数和热力图的目录
    """
    # 准备数据变换
    test_mean = [0.485, 0.456, 0.406]
    test_std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(test_mean, test_std)])
    
    gt_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    # 准备测试数据
    test_dir = os.path.join(data_root, classname, 'test')
    gt_dir = os.path.join(data_root, classname, 'ground_truth')
    test_data = MVTecADTestDataset(data_dir=test_dir, gt_dir=gt_dir, 
                                   transform=test_transform, gt_transform=gt_transform)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 加载模型
    teacher = Teacher(pretrained=False)
    student = Student(img_size=img_size, pretrained=False)
    
    checkpoint = torch.load(ckp_path,map_location=device)
    teacher.load_state_dict(checkpoint['teacher'])
    student.load_state_dict(checkpoint['student'])
    
    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.eval()

    # 初始化结果存储
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    # 用于保存预测分数和热力图的数据
    anomaly_maps_list = []
    img_paths_list = []

    print(f"类别: {classname}")

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            image, gt, ad_label, ad_type, img_path = data   
            img = image.to(device)
            gt = gt.to(device)

            output_t = teacher(img)
            output_s = student(output_t, skip=True, attn=True)
  
            anomaly_map = student.cal_anomaly_map(output_s, output_t)
            anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)


            # 处理img_path可能是元组的情况
            if isinstance(img_path, tuple) and len(img_path) == 1:
                img_path = img_path[0]  # 提取字符串
           # 保存异常图用于热力图生成
            if save_heatmaps_flag or save_predictions:
                anomaly_maps_list.append(anomaly_map.copy())
                # 获取图像路径 (已从数据集返回)
                img_paths_list.append(img_path)
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            # if ad_label.item() != 0:
            #     aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
            #                                 anomaly_map[np.newaxis, :, :]))
            
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
            

    # 计算指标
    # auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
    #auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    # aupro = round(np.mean(aupro_list), 4) if aupro_list else 0.0
    # ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
    # ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

    auroc_sp=0

    # 打印结果
    print(f'Test results for {classname}:')
    #print(f'Pixel-level AUROC: {auroc_px}')
    print(f'Sample-level AUROC: {auroc_sp}')
    # print(f'AUPRO: {aupro}')
    # print(f'Pixel-level AP: {ap_px}')
    # print(f'Sample-level AP: {ap_sp}')
    print(f'----------------------------------------------')

    # 保存到日志文件
    print(f'Test results for {classname}:', file=log)
    # print(f'Pixel-level AUROC: {auroc_px}', file=log)
    print(f'Sample-level AUROC: {auroc_sp}', file=log)
    # print(f'AUPRO: {aupro}', file=log)
    # print(f'Pixel-level AP: {ap_px}', file=log)
    # print(f'Sample-level AP: {ap_sp}', file=log)
    print(f'----------------------------------------------',file=log)

    # 保存到CSV文件
    file_exists = os.path.isfile(results_csv)
    with open(results_csv, 'a', newline='') as csvfile:
        fieldnames = ['classname', 'auroc_px', 'auroc_sp', 'aupro', 'ap_px', 'ap_sp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # writer.writerow({
        #     'classname': classname,
        #     'auroc_px': auroc_px,
        #     'auroc_sp': auroc_sp,
        #     'aupro': aupro,
        #     'ap_px': ap_px,
        #     'ap_sp': ap_sp
        # })

    # 保存预测分数
    if save_predictions and len(pr_list_sp) > 0:
        # 确保预测分数和图像路径对齐
        assert len(img_paths_list) == len(pr_list_sp), \
            f"预测分数数量({len(pr_list_sp)})与图像路径数量({len(img_paths_list)})不匹配"
        pr_list_sp_np = np.array(pr_list_sp)
        save_prediction_scores(img_paths_list, pr_list_sp_np, classname, save_dir)
    
    # 保存热力图
    if save_heatmaps_flag and len(anomaly_maps_list) > 0:
        # 确保异常图和图像路径对齐
        assert len(anomaly_maps_list) == len(img_paths_list), \
            f"异常图数量({len(anomaly_maps_list)})与图像路径数量({len(img_paths_list)})不匹配"
        save_heatmaps(anomaly_maps_list, img_paths_list, classname, save_dir)

    auroc_px = 0.0
    auroc_sp = 0.0
    aupro = 0.0
    ap_px = 0.0
    ap_sp = 0.0

    return {
        'auroc_px': auroc_px,
        'auroc_sp': auroc_sp,
        'aupro': aupro,
        'ap_px': ap_px,
        'ap_sp': ap_sp
    }


if __name__ == "__main__":

    setup_seed(111)

    classnames = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                  'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    log = open("./log_test.txt", 'a')
    #data_root = '/mnt/T4_2/xjj/FirstWorkData1/'
    data_root = '/mnt/T38/bioinf/xjj/Datasets/MVTec-AD/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 256

    for classname in classnames:
        ckp_path = '/mnt/T38/bioinf/xjj/CheckPoints/urd_CP/' + classname + '.pth'
        test_model(device, classname, data_root, log, img_size, ckp_path, results_csv='test_results.csv')

    



    