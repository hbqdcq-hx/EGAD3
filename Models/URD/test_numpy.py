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
               save_predictions=True, save_heatmaps_flag=True, save_dir="/mnt/T4_2/xjj/URD/FirstWorkData1/"):
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

    print(f"类别: {classname}")

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            image, gt, ad_label, ad_type, img_path = data   
            img = image.to(device)
            gt = gt.to(device)

            output_t = teacher(img)
            output_s = student(output_t, skip=True, attn=True)
#########################################################################################TODO  #([1,256,64,64]) ([1,512,32,32])  ([1,1024,16,16])
            img_path_str = img_path[0]  
            # 构建文件夹路径
            # folder_path = img_path_str.replace(
            #     '/mnt/T4_2/xjj/FirstWorkData1/', 
            #     '/mnt/T4_2/xjj/URD/FirstWorkData_numpy/'
            # ).rsplit('.', 1)[0]

            folder_path = img_path_str.replace(
                '/mnt/T38/bioinf/xjj/Datasets/MVTec-AD/', 
                '/mnt/T4_2/xjj/URD/MVTec-AD_numpy/'
            ).rsplit('.', 1)[0]
            
            # 创建文件夹
            os.makedirs(folder_path, exist_ok=True)
            
            # 保存 teacher 的3个特征图
            np.save(os.path.join(folder_path, f"teacher_64x64.npy"), output_t[0][0].cpu().detach().numpy())
            np.save(os.path.join(folder_path, f"teacher_32x32.npy"), output_t[1][0].cpu().detach().numpy())
            np.save(os.path.join(folder_path, f"teacher_16x16.npy"), output_t[2][0].cpu().detach().numpy())
            
            # 保存 student 的3个特征图
            np.save(os.path.join(folder_path, f"student_64x64.npy"), output_s[0][0].cpu().detach().numpy())
            np.save(os.path.join(folder_path, f"student_32x32.npy"), output_s[1][0].cpu().detach().numpy())
            np.save(os.path.join(folder_path, f"student_16x16.npy"), output_s[2][0].cpu().detach().numpy())
#########################################################################################
            '''
            anomaly_map = student.cal_anomaly_map(output_s, output_t)
            anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            if ad_label.item() != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                            anomaly_map[np.newaxis, :, :]))
            
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

    #注释
    # auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    # aupro = round(np.mean(aupro_list), 4) if aupro_list else 0.0
    # ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
    # ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

    #auroc_sp=0

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
    '''
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

    



    