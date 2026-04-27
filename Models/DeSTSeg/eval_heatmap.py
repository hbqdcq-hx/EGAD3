import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from model.destseg import DeSTSeg
from model.metrics import AUPRO, IAPS

import numpy as np
import cv2
from PIL import Image

warnings.filterwarnings("ignore")
def evaluate(args, category, model, visualizer, global_step=0):
    model.eval()
    with torch.no_grad():
        dataset = MVTecDataset(
            is_train=False,
            mvtec_dir=args.mvtec_path + category + "/test/",
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
        )
        # de_st_IAPS = IAPS().cuda()
        # de_st_AUPRO = AUPRO().cuda()
        # de_st_AUROC = AUROC().cuda()
        # de_st_AP = AveragePrecision().cuda()
        # de_st_detect_AUROC = AUROC().cuda()
        # seg_IAPS = IAPS().cuda()
        # seg_AUPRO = AUPRO().cuda()
        # seg_AUROC = AUROC().cuda()
        # seg_AP = AveragePrecision().cuda()
        seg_detect_AUROC = AUROC().cuda()
        #seg_detect_AUROC = AUROC(task="binary").cuda()

        output_segmentation_samples = []
        all_file_names = []  # Collect file names for each batch
        scores = torch.tensor([])  # Initialize scores to avoid UnboundLocalError

        # Create heatmap directory  添加
        heatmap_dir = os.path.join(args.heatmap_path, f"heatmaps_{category}")
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)
            print(f"Created heatmap directory: {heatmap_dir}")
        for _, sample_batched in enumerate(dataloader):
            img = sample_batched["img"].cuda()
            mask = sample_batched["mask"].to(torch.int64).cuda()
            file_paths = sample_batched["file_path"]
            output_segmentation, output_de_st, output_de_st_list = model(img)
            output_segmentation = F.interpolate(
                output_segmentation,
                size=mask.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            output_de_st = F.interpolate(
                output_de_st, size=mask.size()[2:], mode="bilinear", align_corners=False
            )
            mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]
            output_segmentation_sample, _ = torch.sort(
                output_segmentation.view(output_segmentation.size(0), -1),
                dim=1,
                descending=True,
            )
            output_segmentation_sample = torch.mean(
                output_segmentation_sample[:, : args.T], dim=1
            )
            output_de_st_sample, _ = torch.sort(
                output_de_st.view(output_de_st.size(0), -1), dim=1, descending=True
            )
            output_de_st_sample = torch.mean(output_de_st_sample[:, : args.T], dim=1)
            # de_st_IAPS.update(output_de_st, mask)
            # de_st_AUPRO.update(output_de_st, mask)
            # de_st_AP.update(output_de_st.flatten(), mask.flatten())
            # de_st_AUROC.update(output_de_st.flatten(), mask.flatten())
            # de_st_detect_AUROC.update(output_de_st_sample, mask_sample)
            # seg_IAPS.update(output_segmentation, mask)
            # seg_AUPRO.update(output_segmentation, mask)
            # seg_AP.update(output_segmentation.flatten(), mask.flatten())
            # seg_AUROC.update(output_segmentation.flatten(), mask.flatten())
            seg_detect_AUROC.update(output_segmentation_sample, mask_sample)
            # Get file names for current batch - FIXED: use file_path from sample_batched
            batch_file_names = sample_batched["file_path"]
            all_file_names.extend(batch_file_names)
            output_segmentation_samples.append(output_segmentation_sample.cpu())
            scores = torch.cat(output_segmentation_samples)
            # Save heatmaps for current batch  添加
            save_heatmaps(output_segmentation, batch_file_names, heatmap_dir, category)
        # 转换为NumPy数组
        scores_np = scores.numpy()
        # Check if scores array is empty
        if len(scores_np) == 0:
            print(f"Warning: No scores generated for category {category}. Skipping score normalization and file writing.")
            normalized_scores = np.array([])
        else:
            # 最大最小归一化
            normalized_scores = (scores_np - np.min(scores_np)) / (np.max(scores_np) - np.min(scores_np))
            
            # Save anomaly scores to heatmap_path directory as CSV table
            score_filename = f"anomaly_scores_{category}.csv"
            score_path = os.path.join(args.heatmap_path, score_filename)
            
            # Ensure heatmap directory exists
            if not os.path.exists(args.heatmap_path):
                os.makedirs(args.heatmap_path)
            
            with open(score_path, 'a') as f:
                # Write header for table format
                f.write("File_Path,Anomaly_Score\n")
                for name, score in zip(all_file_names, normalized_scores):
                    # Process file name using get_filename_from_path function
                    processed_name = get_filename_from_path(name)
                    f.write(f"{processed_name},{score:.10f}\n")  # 使用逗号分隔的表格格式
            
            print(f"Anomaly scores saved to: {score_path}")

    

        # iap_de_st, iap90_de_st = de_st_IAPS.compute()
        # aupro_de_st, ap_de_st, auc_de_st, auc_detect_de_st = (
        #     de_st_AUPRO.compute(),
        #     de_st_AP.compute(),
        #     de_st_AUROC.compute(),
        #     de_st_detect_AUROC.compute(),
        # )
        # iap_seg, iap90_seg = seg_IAPS.compute()
        # aupro_seg, ap_seg, auc_seg, auc_detect_seg = (
        #     seg_AUPRO.compute(),
        #     seg_AP.compute(),
        #     seg_AUROC.compute(),
        #     seg_detect_AUROC.compute(),
        # )
    
        auc_detect_seg = seg_detect_AUROC.compute()

        # visualizer.add_scalar("DeST_IAP", iap_de_st, global_step)
        # visualizer.add_scalar("DeST_IAP90", iap90_de_st, global_step)
        # visualizer.add_scalar("DeST_AUPRO", aupro_de_st, global_step)
        # visualizer.add_scalar("DeST_AP", ap_de_st, global_step)
        # visualizer.add_scalar("DeST_AUC", auc_de_st, global_step)
        # visualizer.add_scalar("DeST_detect_AUC", auc_detect_de_st, global_step)

        # visualizer.add_scalar("DeSTSeg_IAP", iap_seg, global_step)
        # visualizer.add_scalar("DeSTSeg_IAP90", iap90_seg, global_step)
        # visualizer.add_scalar("DeSTSeg_AUPRO", aupro_seg, global_step)
        # visualizer.add_scalar("DeSTSeg_AP", ap_seg, global_step)
        # visualizer.add_scalar("DeSTSeg_AUC", auc_seg, global_step)
        # visualizer.add_scalar("DeSTSeg_detect_AUC", auc_detect_seg, global_step)

        #print("Eval at step", global_step)
        #print("================================")
        # print("Denoising Student-Teacher (DeST)")
        # print("pixel_AUC:", round(float(auc_de_st), 4))
        # print("pixel_AP:", round(float(ap_de_st), 4))
        # print("PRO:", round(float(aupro_de_st), 4))
        # print("image_AUC:", round(float(auc_detect_de_st), 4))
        # print("IAP:", round(float(iap_de_st), 4))
        # print("IAP90:", round(float(iap90_de_st), 4))
        #print()
        #print("Segmentation Guided Denoising Student-Teacher (DeSTSeg)")
        # print("pixel_AUC:", round(float(auc_seg), 4))
        # print("pixel_AP:", round(float(ap_seg), 4))
        # print("PRO:", round(float(aupro_seg), 4))
        #print("image_AUC:", round(float(auc_detect_seg), 4))
        # print("IAP:", round(float(iap_seg), 4))
        # print("IAP90:", round(float(iap90_seg), 4))
        #print()

        # de_st_IAPS.reset()
        # de_st_AUPRO.reset()
        # de_st_AUROC.reset()
        # de_st_AP.reset()
        # de_st_detect_AUROC.reset()
        # seg_IAPS.reset()
        # seg_AUPRO.reset()
        # seg_AUROC.reset()
        # seg_AP.reset()
        seg_detect_AUROC.reset()

def test(args, category):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    run_name = f"DeSTSeg_MVTec_test_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))
    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))
    model = DeSTSeg(dest=True, ed=True).cuda()
    assert os.path.exists(
        os.path.join(args.checkpoint_path, args.base_model_name + category + ".pckl")
    )
    model.load_state_dict(
        torch.load(
            os.path.join(
                args.checkpoint_path, args.base_model_name + category + ".pckl"
            )
        )
    )
    evaluate(args, category, model, visualizer)
## 添加
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

def save_heatmaps(heatmap_tensor, file_paths, output_dir, category):
    """
    Save heatmaps as images, upsampled to original image size
    Args:
        heatmap_tensor: Tensor of shape [batch_size, 1, H, W] containing heatmap data
        file_paths: List of original file paths
        output_dir: Directory to save heatmaps
        category: Current category name
    """
    batch_size = heatmap_tensor.size(0)
    
    for i in range(batch_size):
        # Get original image size
        original_path = file_paths[i]
        img_pil = Image.open(original_path)
        # PIL size is (width, height), but tensor expects (height, width)
        target_size = (img_pil.size[1], img_pil.size[0])  # (H_original, W_original)
        img_pil.close()
        
        # Get heatmap for current image as tensor [1, 1, H, W]
        single_heatmap = heatmap_tensor[i:i+1, 0:1]  # keep batch and channel dims
        
        # Upsample to original size
        upsampled = F.interpolate(
            single_heatmap,
            size=target_size,
            mode='bilinear',
            align_corners=True
        ).squeeze()  # [H_original, W_original]
        
        # Convert to numpy and check anomaly value range
        heatmap = upsampled.cpu().numpy()
        
        # 检查异常值是否在理论范围内 [0, 1]，超出范围则报错暂停
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        if min_val < 0 or max_val > 1:
            raise ValueError(f"错误: 异常值超出理论范围 [0, 1]! 检测到范围: [{min_val:.6f}, {max_val:.6f}]。文件: {original_path}")
        
        # 值在理论范围内，直接映射到 [0, 255]
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)  
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Get filename using the new naming convention
        filename = get_filename_from_path(original_path)
        
        # Save heatmap directly in the category directory (no subdirectories)
        output_path = os.path.join(output_dir, f"{filename}_heatmap.png")
        cv2.imwrite(output_path, heatmap_colored)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--mvtec_path", type=str, default="/mnt/T38/bioinf/xjj/Datasets/good_dataset/")#/mnt/T38/bioinf/xjj/Datasets/MVTec-AD/      /mnt/T4_2/xjj/FirstWorkData1/
    parser.add_argument("--dtd_path", type=str, default="/mnt/T38/bioinf/xjj/Datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="/mnt/T38/bioinf/xjj/CheckPoints/destseg_CP/")
    parser.add_argument("--base_model_name", type=str, default="DeSTSeg_MVTec_5000_")
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--heatmap_path", type=str, default="/mnt/T4_1/xjj/2/destseg/good_dataset")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--T", type=int, default=100)  # for image-level inference
    parser.add_argument("--category", nargs="*", type=str, default=ALL_CATEGORY)
    args = parser.parse_args()

    obj_list = args.category
    # obj_list = {'carpet', 'leather', 'grid', 'tile', 'wood', 'bottle', 'hazelnut', 'cable', 'capsule',
    #           'pill', 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper'}
    for obj in obj_list:
        assert obj in ALL_CATEGORY

    with torch.cuda.device(args.gpu_id):
        for obj in obj_list:
            print(obj)
            test(args, obj)