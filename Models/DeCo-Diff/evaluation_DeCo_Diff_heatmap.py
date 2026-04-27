import torch
from skimage.transform import resize
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import UNET_models
import argparse
import numpy as np
import os
import cv2
import logging
from PIL import Image
torch.set_grad_enabled(False)
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
from glob import glob

from torch.utils.data import DataLoader
from torchvision import transforms
from MVTECDataLoader import MVTECDataset
from VISADataLoader import VISADataset
from scipy.ndimage import gaussian_filter

from anomalib import metrics
from sklearn.metrics import average_precision_score
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import auc

# Setup logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#添加
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
    Save heatmaps as images
    Args:
        heatmap_tensor: Tensor of shape [batch_size, 1, H, W] or [batch_size, H, W] containing heatmap data
        file_paths: List of original file paths
        output_dir: Base directory to save heatmaps
        category: Current category name (will be used as subdirectory name)
    """
    
    # 创建类别子目录: output_dir/category/
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    batch_size = heatmap_tensor.size(0)
    tensor_shape = heatmap_tensor.shape
    LOGGER.info(f"Saving heatmaps for category: {category}, batch_size: {batch_size}, tensor_shape: {tensor_shape}")
    LOGGER.info(f"Heatmaps will be saved to category directory: {category_dir}")
    
    saved_count = 0
    for i in range(batch_size):
        try:
            # 处理不同的张量形状
            if len(tensor_shape) == 4:  # [batch_size, 1, H, W]
                single_heatmap = heatmap_tensor[i:i+1, 0:1]  # [1, 1, H, W]
            elif len(tensor_shape) == 3:  # [batch_size, H, W]
                single_heatmap = heatmap_tensor[i:i+1].unsqueeze(1)  # [1, 1, H, W]
            else:
                LOGGER.warning(f"Unexpected tensor shape: {tensor_shape}, skipping heatmap {i}")
                continue
            
            # 检查heatmap是否有效
            if single_heatmap.numel() == 0:
                LOGGER.warning(f"Heatmap {i} is empty, skipping")
                continue
            
            # 获取原始图像尺寸
            original_path = file_paths[i]
            try:
                with Image.open(original_path) as img:
                    original_size = img.size  # (width, height)
                    target_size = (original_size[1], original_size[0])  # (height, width) for PyTorch
            except Exception as e:
                LOGGER.warning(f"Could not get original image size for {original_path}: {e}. Using heatmap size.")
                target_size = single_heatmap.shape[2:]  # (H, W)
            
            # 上采样到原始尺寸
            if single_heatmap.shape[2:] != target_size:
                upsampled = F.interpolate(
                    single_heatmap,
                    size=target_size,
                    mode='bilinear',
                    align_corners=True
                ).squeeze()  # [H_original, W_original]
            else:
                upsampled = single_heatmap.squeeze()
            
            # 转换为numpy数组
            heatmap = upsampled.cpu().numpy()
            
            # Normalize heatmap to [0, 1]
            heatmap_min = np.min(heatmap)
            heatmap_max = np.max(heatmap)
            if heatmap_max - heatmap_min > 0:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
            else:
                heatmap = np.zeros_like(heatmap)
            
            # Convert to uint8 and apply colormap
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # Get filename using the new naming convention
            filename = get_filename_from_path(original_path)
            
            # Save heatmap in the category subdirectory
            output_path = os.path.join(category_dir, f"{filename}_heatmap.png")
            success = cv2.imwrite(output_path, heatmap_colored)
            
            if success:
                LOGGER.debug(f"Heatmap saved: {output_path} (size: {target_size[1]}x{target_size[0]})")
                saved_count += 1
            else:
                LOGGER.warning(f"Failed to save heatmap: {output_path}")
                
        except Exception as e:
            LOGGER.error(f"Error saving heatmap {i}: {str(e)}")
            continue
    
    LOGGER.info(f"Heatmaps saved to category directory: {category_dir}")
    LOGGER.info(f"Successfully saved {saved_count}/{batch_size} heatmaps for category: {category}")


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": [np.mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc




def calculate_metrics(ground_truth, prediction, test_data=None, category=None):
    """
    计算评估指标并保存热力图和预测分数。
    
    参数:
        ground_truth: 真实分割掩码
        prediction: 预测异常图（也就是segmentations）
        
    添加：
        test_data: 测试数据集（可选）
        category: 类别名称（可选）
    """
    flat_gt = ground_truth.flatten()
    flat_pred = prediction.flatten()
    
    # 保存热力图和预测分数（如果提供了必要的数据）
    if test_data is not None:
        # 保存热力图
        save_path = "/mnt/T4_1/xjj/2/DeCo-Diff/FirstWorkData"
        os.makedirs(save_path, exist_ok=True)
        
        # 将prediction（异常图）转换为张量
        prediction_tensor = torch.from_numpy(np.array(prediction))
        if len(prediction_tensor.shape) == 3:  # [N, H, W]
            prediction_tensor = prediction_tensor.unsqueeze(1)  # [N, 1, H, W]
        
        # 获取图像路径 - 从数据集的 image_paths 属性获取
        test_dataset = test_data.dataset
        if hasattr(test_dataset, 'image_paths'):
            all_image_paths = test_dataset.image_paths
        else:
            # 如果数据集没有 image_paths 属性，尝试其他方法
            LOGGER.warning("Dataset does not have 'image_paths' attribute. Cannot save heatmaps.")
            all_image_paths = []
        
        # 如果没有提供category，则从数据集中获取
        if category is None:
            classname = test_dataset.object_class if hasattr(test_dataset, 'object_class') else "unknown"
        else:
            classname = category
        
        # 保存热力图（如果有图像路径）
        if all_image_paths:
            save_heatmaps(prediction_tensor, all_image_paths, save_path, classname)
            LOGGER.info(f"Heatmaps saved to: {save_path}")

            # -------------------- 保存预测分数 --------------------
            file_names = [get_filename_from_path(path) for path in all_image_paths]
            
            # 计算预测分数（每个异常图的最大值）
            scores = []
            for idx in range(len(prediction)):
                score = np.max(prediction[idx])
                scores.append(score)
            
            # 注意：分数归一化代码被故意注释掉，保存原始分数
            # 使用assert语句检查分数范围，如果为0就停止
            score_range = np.max(scores) - np.min(scores)
            assert score_range > 0, f"分数范围为零，无法归一化。所有分数相同: {scores[0]}"
            # 最大最小归一化
            normalized_scores = (scores - np.min(scores)) / score_range

            # 保存异常分数到CSV文件
            score_filename = f"anomaly_scores_{classname}.csv"
            csv_file_path = os.path.join(save_path, score_filename)
            
            with open(csv_file_path, 'w') as f:
                # 写入表头
                f.write("File_Path,Anomaly_Score\n")
                for name, score in zip(file_names, normalized_scores):
                    f.write(f"{name},{score:.10f}\n")  # 使用逗号分隔的表格格式
            
            LOGGER.info(f"Anomaly scores saved to: {csv_file_path}")
        else:
            LOGGER.warning("No image paths available. Skipping heatmap and score saving.")

    # auprc = metrics.AUPR()
    # auprc_score = auprc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    
    # aupro = metrics.AUPRO(fpr_limit=0.3)
    # #aupro_score = compute_pro(ground_truth, prediction)
    
    # auroc = metrics.AUROC()
    # auroc_score = auroc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    # f1max = metrics.F1Max()
    # f1_max_score = f1max(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))
    
    # ap = average_precision_score(ground_truth.flatten(), prediction.flatten())
    
    # gt_list_sp = []
    # pr_list_sp = []
    # for idx in range(len(ground_truth)):
    #     gt_list_sp.append(np.max(ground_truth[idx]))
    #     sp_score = np.max(prediction[idx])
    #     pr_list_sp.append(sp_score)

    # gt_list_sp = np.array(gt_list_sp).astype(np.int32)
    # pr_list_sp = np.array(pr_list_sp)

    # apsp = average_precision_score(gt_list_sp, pr_list_sp)
    # aurocsp = auroc(torch.from_numpy(pr_list_sp), torch.from_numpy(gt_list_sp))
    # f1sp = f1max(torch.from_numpy(pr_list_sp), torch.from_numpy(gt_list_sp))

    f1sp,apsp,aurocsp,ap,f1_max_score,auroc_score,aupro_score = 0, 0, 0, 0, 0, 0, 0
    
    #return auroc_score.numpy(), aupro_score ,f1_max_score.numpy(), ap, aurocsp.numpy(), apsp, f1sp.numpy()
    return auroc_score, aupro_score ,f1_max_score, ap, aurocsp, apsp, f1sp


def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask


    

def calculate_anomaly_maps(x0_s, encoded_s,  image_samples_s, latent_samples_s, center_size=256):
    pred_geometric = []
    pred_aritmetic = []
    image_differences = []
    latent_differences = []
    input_images = []
    output_images = []
    for x, encoded,  image_samples, latent_samples in zip(x0_s, encoded_s,  image_samples_s, latent_samples_s):
            
        input_image = ((np.clip(x[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
        output_image = ((np.clip(image_samples[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
        input_images.append(input_image)
        output_images.append(output_image)

        image_difference = (((((torch.abs(image_samples-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
        image_difference = (np.clip(image_difference, 0.0, 0.4) ) * 2.5
        image_difference = smooth_mask(image_difference, sigma=3)
        image_differences.append(image_difference)
        
        latent_difference = (((((torch.abs(latent_samples-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
        latent_difference = (np.clip(latent_difference, 0.0 , 0.2)) * 5
        latent_difference = smooth_mask(latent_difference, sigma=1)
        latent_difference = resize(latent_difference, (center_size, center_size))
        latent_differences.append(latent_difference)
        
        final_anomaly = image_difference * latent_difference
        final_anomaly = np.sqrt(final_anomaly)
        final_anomaly2 = 1/2*image_difference + 1/2*latent_difference
        pred_geometric.append(final_anomaly)
        pred_aritmetic.append(final_anomaly2)
            
    pred_geometric = np.stack(pred_geometric, axis=0)
    pred_aritmetic = np.stack(pred_aritmetic, axis=0)
    latent_differences = np.stack(latent_differences, axis=0)
    image_differences = np.stack(image_differences, axis=0)

    #return {'anomaly_geometric':pred_geometric, 'anomaly_aritmetic':pred_aritmetic, 'latent_discrepancy':latent_differences, 'image_discrepancy':image_differences}
    return {'anomaly_geometric':pred_geometric}


def evaluate_anomaly_maps(anomaly_maps, segmentation, test_data=None, category=None):
    """
    评估异常图并打印结果。
    
    参数:
        anomaly_maps: 异常图字典
        segmentation: 真实分割掩码
        test_data: 测试数据集（可选）
        category: 类别名称（可选）
    """
    for key in anomaly_maps.keys():
        #添加了数据和类别两个参数
        auroc_score, aupro_score ,f1_max_score, ap, aurocsp, apsp, f1sp = calculate_metrics(
            segmentation, 
            anomaly_maps[key],
            test_data=test_data,
            category=category
        )
        auroc_score, aupro_score, f1_max_score, ap, aurocsp, apsp, f1sp, = np.round(auroc_score, 4), np.round(aupro_score, 4), np.round(f1_max_score, 4), np.round(ap, 4), np.round(aurocsp, 4), np.round(apsp, 4), np.round(f1sp, 4)
        print('{}: auroc:{:.4f}, aupro:{:.4f}, f1_max:{:.4f}, ap:{:.4f}, aurocsp:{:.4f}, apsp:{:.4f}, f1sp:{:.4f}'.format(key, auroc_score, aupro_score, f1_max_score, ap, aurocsp, apsp, f1sp))


def evaluation(args):
    vae_model = f"stabilityai/sd-vae-ft-{args.vae_type}" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    vae.eval()
    try:
        if args.model_path != '':
            ckpt = args.model_path
        else:
            path = f"./DeCo-Diff_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
            try:
                ckpt = sorted(glob(f'{path}/last.pt'))[-1]
            except:
                ckpt = sorted(glob(f'{path}/*/last.pt'))[-1]
    except:
        raise Exception("Please provide the trained model's path using --model_path")
    

    latent_size = int(args.center_size) // 8
    model = UNET_models[args.model_size](latent_size=latent_size, ncls=args.num_classes)
    
    state_dict = torch.load(ckpt)['model']
    print(model.load_state_dict(state_dict))
    model.eval() # important!
    model.to(device)
    print('model loaded')


    print('=='*30)
    print('Starting Evaluation...')
    print('=='*30)

    for category in args.categories:


        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
            
        # Create diffusion object:
        diffusion = create_diffusion(f'ddim{args.reverse_steps}', predict_deviation=True, sigma_small=False, predict_xstart=False, diffusion_steps=10)
            

        encoded_s = []
        image_samples_s = []
        latent_samples_s = []
        x0_s = []
        segmentation_s = []
        
        if args.dataset=='mvtec':
            test_dataset = MVTECDataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop)
        else:
            test_dataset = VISADataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)
        
        for ii, (x, seg, object_cls, img_paths) in enumerate(test_loader):
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                encoded = vae.encode(x.to(device)).latent_dist.mean.mul_(0.18215)
                model_kwargs = {
                'context':object_cls.to(device).unsqueeze(1),
                'mask': None
                }
                latent_samples = diffusion.ddim_deviation_sample_loop(
                    model, encoded.shape, noise = encoded, clip_denoised=False, 
                    start_t = args.reverse_steps,
                    model_kwargs=model_kwargs, progress=False, device=device,
                    eta = 0
                )

                image_samples = vae.decode(latent_samples / 0.18215).sample 
                x0 = vae.decode(encoded / 0.18215).sample 

            segmentation_s += [_seg.squeeze() for _seg in seg]
            encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
            image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_samples]
            x0_s += [_x0.unsqueeze(0) for _x0 in x0]

        print(category)        
        anomaly_maps = calculate_anomaly_maps(x0_s, encoded_s,  image_samples_s, latent_samples_s, center_size=args.center_size)
        #添加了数据和类别两个参数
        evaluate_anomaly_maps(
            anomaly_maps, 
            np.stack(segmentation_s, axis=0),
            test_data=test_loader,
            category=category
        )
        print('=='*30)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['mvtec','visa'], default="mvtec")
    parser.add_argument("--data-dir", type=str, default='./mvtec-dataset/')
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_L')
    parser.add_argument("--image-size", type=int, default= 288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--object-category", type=str, default='all')
    parser.add_argument("--model-path", type=str, default='.')
    parser.add_argument("--anomaly-class", type=str, default='all')
    parser.add_argument("--reverse-steps", type=int, default=5)
    
    
    args = parser.parse_args()
    if args.dataset == 'mvtec':
        args.num_classes = 15
    elif args.dataset == 'visa':
        args.num_classes = 12
    args.results_dir = f"./DeCo-Diff_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
    if args.center_crop:
        args.results_dir += "_CenterCrop"
        args.actual_image_size = args.center_size
    else:
        args.actual_image_size = args.image_size

    if args.object_category=='all' and args.dataset=='mvtec':
        args.categories=[
            "bottle",
            "cable",
            "capsule",
            "hazelnut",
            "metal_nut",
            "pill",
            "screw",
            "toothbrush",
            "transistor",
            "zipper",
            "carpet",
            "grid",
            "leather",
            "tile",
            "wood",
            ]
    elif args.object_category=='all' and args.dataset=='visa':
        args.categories=[
            "candle",
            "cashew",
            "fryum",
            "macaroni2",
            "pcb2",
            "pcb4",
            "capsules",
            "chewinggum",
            "macaroni1",
            "pcb1",
            "pcb3",
            "pipe_fryum"
            ]
    else:
        args.categories = [args.object_category]
        
    evaluation(args)
