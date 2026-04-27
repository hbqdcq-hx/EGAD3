import os
import time
import datetime
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from datasets import MVTecDataset, VisADataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from post_process import post_process
from utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights
from evaluations import eval_det_loc

import cv2


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

def save_heatmaps(c, anomaly_score_map, file_names, original_sizes, class_name, save_dir="./results"):
    """
    保存异常分数热力图，并上采样到原始图片大小
    
    Args:
        c: 配置对象
        anomaly_score_map: 异常图数组，形状为 (N, H, W)
        file_names: 文件名列表，长度 N
        original_sizes: 原始图片大小列表，每个元素为 (width, height)
        class_name: 类别名称
        save_dir: 保存目录
    """
    
    # 创建类别文件夹
    class_dir = os.path.join(save_dir,  class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    print(f"开始保存热力图到: {class_dir}")
    
    count = 0
    for idx in range(len(file_names)):
        # 获取当前异常图
        anomaly_map = anomaly_score_map[idx]  # (H, W)
        
        # # 最小最大归一化
        # min_val = anomaly_map.min()
        # max_val = anomaly_map.max()
        # if max_val > min_val:
        #     anomaly_map = (anomaly_map - min_val) / (max_val - min_val)
        # else:
        #     anomaly_map = np.zeros_like(anomaly_map)
        
        # 上采样到原始图片大小
        orig_width, orig_height = original_sizes[idx]  # (width, height)
        # 确保宽度和高度是整数
        orig_width = int(orig_width)
        orig_height = int(orig_height)
        target_size = (orig_height, orig_width)  # (height, width) for interpolation
        
        # 将异常图转换为tensor进行插值
        # 使用 torch 和 F，它们已经在顶部导入
        anomaly_map_tensor = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        upsampled = F.interpolate(
            anomaly_map_tensor,
            size=target_size,
            mode='bilinear',
            align_corners=True
        ).squeeze()  # [H_original, W_original]
        upsampled_np = upsampled.cpu().numpy()
        
        # # 再次归一化（确保在0-1之间）
        # if upsampled_np.max() > upsampled_np.min():
        #     upsampled_np = (upsampled_np - upsampled_np.min()) / (upsampled_np.max() - upsampled_np.min())
        # else:
        #     upsampled_np = np.zeros_like(upsampled_np)
        
        # 转换为热力图（Jet颜色映射）
        heatmap = cv2.applyColorMap(np.uint8(upsampled_np * 255), cv2.COLORMAP_JET)
        
        # 获取文件名
        filename = file_names[idx]
        
        # 保存热力图
        heatmap_path = os.path.join(class_dir, f"{filename}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        
        count += 1
    
    print(f"热力图保存完成，共保存 {count} 张热力图到 {class_dir}")


def model_forward(c, extractor, parallel_flows, fusion_flow, image):
    h_list = extractor(image)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_list = []
    parallel_jac_list = []
    for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list, parallel_flows, c.c_conds)):
        y = pool_layer(h)
        B, _, H, W = y.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        z, jac = parallel_flow(y, [cond, ])
        z_list.append(z)
        parallel_jac_list.append(jac)

    z_list, fuse_jac = fusion_flow(z_list)
    jac = fuse_jac + sum(parallel_jac_list)

    return z_list, jac

def train_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler=None):
    parallel_flows = [parallel_flow.train() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.train()

    for sub_epoch in range(c.sub_epochs):
        epoch_loss = 0.
        image_count = 0
        for idx, (image, _, _) in enumerate(loader):
            optimizer.zero_grad()
            image = image.to(c.device)
            if scaler:
                with autocast():
                    z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                    loss = 0.
                    for z in z_list:
                        loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                    loss = loss - jac
                    loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 2)
                scaler.step(optimizer)
                scaler.update()
            else:
                z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                loss = 0.
                for z in z_list:
                    loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                loss = loss - jac
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2)
                optimizer.step()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if warmup_scheduler:
            warmup_scheduler.step()
        if decay_scheduler:
            decay_scheduler.step()

        mean_epoch_loss = epoch_loss / image_count
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                epoch, sub_epoch, mean_epoch_loss, lr))
        

def inference_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow):
    parallel_flows = [parallel_flow.eval() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.eval()
    epoch_loss = 0.
    image_count = 0
    gt_label_list = list()
    #gt_mask_list = list()
    outputs_list = [list() for _ in parallel_flows]
    size_list = []
    original_sizes = []  # 存储原始图片大小
    start = time.time()

    category=c.class_name

    with torch.no_grad():
        # MVTecDataset 返回 (image, label, original_size)
        for idx, (image, label, original_size) in enumerate(loader):
            # original_size 是一个包含两个张量的列表，每个张量形状为 [batch_size]
            # 分别代表宽度和高度
            width_tensor, height_tensor = original_size
            # 遍历批次中的每个图像
            for w, h in zip(width_tensor, height_tensor):
                width = int(w.item())
                height = int(h.item())
                original_sizes.append((width, height))
            image = image.to(c.device)
            gt_label_list.extend(t2np(label))
            #gt_mask_list.extend(t2np(mask))
            z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
            loss = 0.
            for lvl, z in enumerate(z_list):
                if idx == 0:
                    size_list.append(list(z.shape[-2:]))
                logp = - 0.5 * torch.mean(z**2, 1)
                outputs_list[lvl].append(logp)
                loss += 0.5 * torch.sum(z**2, (1, 2, 3))

            loss = loss - jac
            loss = loss.mean()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]

        mean_epoch_loss = epoch_loss / image_count
        fps = len(loader.dataset) / (time.time() - start)
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                epoch, mean_epoch_loss, fps))

    #return gt_label_list, gt_mask_list, outputs_list, size_list
    return gt_label_list, outputs_list, size_list, original_sizes


def train(c):
    
    #Weights & Biases (WandB) 的初始化和配置，主要用于实验跟踪和可视化。
    if c.wandb_enable:
        wandb.finish()
        wandb_run = wandb.init(
            project='65001-msflow', 
            group=c.version_name,
            name=c.class_name)
    
    Dataset = MVTecDataset if c.dataset == 'mvtec' else VisADataset

    #train_dataset = Dataset(c, is_train=True)
    test_dataset  = Dataset(c, is_train=False)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
    parallel_flows = [parallel_flow.to(c.device) for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.to(c.device)

    # if c.wandb_enable:
    #     for idx, parallel_flow in enumerate(parallel_flows):
    #         wandb.watch(parallel_flow, log='all', log_freq=100, idx=idx)
    #     wandb.watch(fusion_flow, log='all', log_freq=100, idx=len(parallel_flows))
    params = list(fusion_flow.parameters())
    for parallel_flow in parallel_flows:
        params += list(parallel_flow.parameters())
        
    optimizer = torch.optim.Adam(params, lr=c.lr)
    if c.amp_enable:
        scaler = GradScaler()

    det_auroc_obs = Score_Observer('Det.AUROC', c.meta_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)

    start_epoch = 0
    if c.mode == 'test':
        start_epoch = load_weights(parallel_flows, fusion_flow, c.eval_ckpt)      #好费时间
        epoch = start_epoch + 1
        #gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow)
        gt_label_list, outputs_list, size_list, original_sizes = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
        #best_det_auroc, best_loc_auroc, best_loc_pro = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)
        best_det_auroc = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)
        

        anomaly_score_array = np.array(anomaly_score)

        # 计算最小值和最大值
        min_score = np.min(anomaly_score_array)
        max_score = np.max(anomaly_score_array)

        if max_score == min_score:
            normalized_score = anomaly_score_array
        else:
            range_score = max_score - min_score
            normalized_score = (anomaly_score_array - min_score) / range_score


        #save_dir = "/mnt/T4_2/xjj/msflow/MVTec-AD/"
        save_dir = "/mnt/T4_1/xjj/1/msflow/MVTec-AD/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 使用get_filename_from_path提取文件名
        file_names_new = [get_filename_from_path(path) for path in test_dataset.x]
        
        csv_path = os.path.join(save_dir, f'anomaly_scores_{c.class_name}.csv')
        
        with open(csv_path, 'w') as f:
            # 写入表头
            f.write("File_Name,Anomaly_Score\n")
            # 使用zip确保一一对应
            for name, score in zip(file_names_new, normalized_score):
                f.write(f"{name},{score:.10f}\n")  # 使用逗号分隔的表格格式
        
        print(f"异常分数已保存到: {csv_path}")
        
        # 保存热力图
        save_heatmaps(c, anomaly_score_map_add, file_names_new, original_sizes, c.class_name, save_dir)

        return
    
    if c.resume:
        last_epoch = load_weights(parallel_flows, fusion_flow, os.path.join(c.ckpt_dir, 'last.pt'), optimizer)
        start_epoch = last_epoch + 1
        print('Resume from epoch {}'.format(start_epoch))

    if c.lr_warmup and start_epoch < c.lr_warmup_epochs:
        if start_epoch == 0:
            start_factor = c.lr_warmup_from
            end_factor = 1.0
        else:
            start_factor = 1.0
            end_factor = c.lr / optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=(c.lr_warmup_epochs - start_epoch)*c.sub_epochs)
    else:
        warmup_scheduler = None

    mile_stones = [milestone - start_epoch for milestone in c.lr_decay_milestones if milestone > start_epoch]

    if mile_stones:
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, c.lr_decay_gamma)
    else:
        decay_scheduler = None

    for epoch in range(start_epoch, c.meta_epochs):
        print()
        train_meta_epoch(c, epoch, train_loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler if c.amp_enable else None)

        gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

        if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
            pro_eval = True
        else:
            pro_eval = False

        det_auroc, loc_auroc, loc_pro_auc, \
            best_det_auroc, best_loc_auroc, best_loc_pro = \
                eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, pro_eval)

        if c.wandb_enable:
            wandb_run.log(
                {
                    'Det.AUROC': det_auroc,
                    'Loc.AUROC': loc_auroc,
                    'Loc.PRO': loc_pro_auc
                },
                step=epoch
            )

        save_weights(epoch, parallel_flows, fusion_flow, 'last', c.ckpt_dir, optimizer)
        if best_det_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_det', c.ckpt_dir)
        if best_loc_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_auroc', c.ckpt_dir)
        if best_loc_pro and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_pro', c.ckpt_dir)
