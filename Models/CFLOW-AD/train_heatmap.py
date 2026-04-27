import os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from skimage.measure import label, regionprops
from tqdm import tqdm
from visualize import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_datasets import *
from custom_models import *

import cv2

#from datetime import datetime



OUT_DIR = './viz/'

gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()


def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, pool_layers, N):
    P = c.condition_vec
    L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    iterator = iter(loader)
    for sub_epoch in range(c.sub_epochs):
        train_loss = 0.0
        train_count = 0
        for i in range(I):
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizer)
            # sample batch
            try:
                image, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, _ = next(iterator)
            # encoder prediction
            image = image.to(c.device)  # single scale
            with torch.no_grad():
                _ = encoder(image)
            # train decoder
            e_list = list()
            c_list = list()
            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer].detach()  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S    
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                perm = torch.randperm(E).to(c.device)  # BHW
                decoder = decoders[l]
                #
                FIB = E//N  # number of fiber batches
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                for f in range(FIB):  # per-fiber processing
                    idx = torch.arange(f*N, (f+1)*N)
                    c_p = c_r[perm[idx]]  # NxP
                    e_p = e_r[perm[idx]]  # NxC
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)
        #
        mean_train_loss = train_loss / train_count
        if c.verbose:
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))
    #


def test_meta_epoch(c, epoch, loader, encoder, decoders, pool_layers, N):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    
    # 打印当前时间（默认格式）
    print("当前时间1:", datetime.datetime.now())
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    original_sizes = list()  # 新增：保存原始图像尺寸
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    start = time.time()
    with torch.no_grad():
        #for i, (image, label, mask) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
        for i, (image, label, original_size) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # save
            if c.viz:
                image_list.extend(t2np(image))
            gt_label_list.extend(t2np(label))

             # original_size 是一个包含两个张量的列表，每个张量形状为 [batch_size]
            # 分别代表宽度和高度
            width_tensor, height_tensor = original_size
            # 遍历批次中的每个图像
            for w, h in zip(width_tensor, height_tensor):
                width_ = int(w.item())
                height_ = int(h.item())
                original_sizes.append((width_, height_))

            #gt_mask_list.extend(t2np(mask))
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
            # test decoder
            e_list = list()
            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer]  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                #m = F.interpolate(mask, size=(H, W), mode='nearest')
                #m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                #
                decoder = decoders[l]
                # FIB = E//N + int(E%N > 0)  # number of fiber batches
                # for f in range(FIB):
                #     if f < (FIB-1):
                #         idx = torch.arange(f*N, (f+1)*N)
                #     else:
                #         idx = torch.arange(f*N, E)
                idx = torch.arange(E)
                c_p = c_r[idx]  # NxP
                e_p = e_r[idx]  # NxC
                #m_p = m_r[idx] > 0.5  # Nx1
                #
                if 'cflow' in c.dec_arch:
                    z, log_jac_det = decoder(e_p, [c_p,])
                else:
                    z, log_jac_det = decoder(e_p)
                #
                decoder_log_prob = get_logp(C, z, log_jac_det)
                log_prob = decoder_log_prob / C  # likelihood per dim
                loss = -log_theta(log_prob)
                test_loss += t2np(loss.sum())
                test_count += len(loss)
                test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
    #
    fps = len(loader.dataset) / (time.time() - start)
    mean_test_loss = test_loss / test_count
    if c.verbose:
        print('Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps'.format(epoch, mean_test_loss, fps))
    #
    #return height, width, image_list, test_dist, gt_label_list, gt_mask_list
    return height, width, image_list, test_dist, gt_label_list, original_sizes


def test_meta_fps(c, epoch, loader, encoder, decoders, pool_layers, N):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    A = len(loader.dataset)
    with torch.no_grad():
        # warm-up
        for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
        # measure encoder only
        torch.cuda.synchronize()
        start = time.time()
        for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
        # measure encoder + decoder
        torch.cuda.synchronize()
        time_enc = time.time() - start
        start = time.time()
        for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
            # test decoder
            e_list = list()
            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer]  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                decoder = decoders[l]
                FIB = E//N + int(E%N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB-1):
                        idx = torch.arange(f*N, (f+1)*N)
                    else:
                        idx = torch.arange(f*N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
    #
    torch.cuda.synchronize()
    time_all = time.time() - start
    fps_enc = A / time_enc
    fps_all = A / time_all
    print('Encoder/All {:.2f}/{:.2f} fps'.format(fps_enc, fps_all))
    #
    return height, width, image_list, test_dist, gt_label_list, gt_mask_list


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
        
        # 最小最大归一化
        min_val = anomaly_map.min()
        max_val = anomaly_map.max()
        if max_val > min_val:
            anomaly_map = (anomaly_map - min_val) / (max_val - min_val)
        else:
            anomaly_map = np.zeros_like(anomaly_map)

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
        
        # 再次归一化（确保在0-1之间）
        if upsampled_np.max() > upsampled_np.min():
            upsampled_np = (upsampled_np - upsampled_np.min()) / (upsampled_np.max() - upsampled_np.min())
        else:
            upsampled_np = np.zeros_like(upsampled_np)
        
        # 转换为热力图（Jet颜色映射）
        heatmap = cv2.applyColorMap(np.uint8(upsampled_np * 255), cv2.COLORMAP_JET)
        
        # 获取文件名
        filename = file_names[idx]
        
        # 保存热力图
        heatmap_path = os.path.join(class_dir, f"{filename}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        
        count += 1
    
    print(f"热力图保存完成，共保存 {count} 张热力图到 {class_dir}")


def train(c):
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    L = c.pool_layers # number of pooled layers
    print('Number of pool layers =', L)
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(c.device).eval()
    #print(encoder)
    # NF decoder
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(c.device) for decoder in decoders]
    params = list(decoders[0].parameters())
    for l in range(1, L):
        params += list(decoders[l].parameters())
    # optimizer
    optimizer = torch.optim.Adam(params, lr=c.lr)
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    # task data
    if c.dataset == 'mvtec':
        #train_dataset = MVTecDataset(c, is_train=True)
        test_dataset  = MVTecDataset(c, is_train=False)
    elif c.dataset == 'stc':
        train_dataset = StcDataset(c, is_train=True)
        test_dataset  = StcDataset(c, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    #
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    N = 256  # hyperparameter that increases batch size for the decoder model by N
    # print('train/test loader length', len(train_loader.dataset), len(test_loader.dataset))
    # print('train/test loader batches', len(train_loader), len(test_loader))
    print('test loader length', len(test_loader.dataset))
    print('test loader batches', len(test_loader))
    # stats
    det_roc_obs = Score_Observer('DET_AUROC')
    seg_roc_obs = Score_Observer('SEG_AUROC')
    seg_pro_obs = Score_Observer('SEG_AUPRO')
    if c.action_type == 'norm-test':
        c.meta_epochs = 1
    for epoch in range(c.meta_epochs):
        if c.action_type == 'norm-test' and c.checkpoint:
            load_weights(encoder, decoders, c.checkpoint)
        elif c.action_type == 'norm-train':
            print('Train meta epoch: {}'.format(epoch))
            train_meta_epoch(c, epoch, train_loader, encoder, decoders, optimizer, pool_layers, N)
        else:
            raise NotImplementedError('{} is not supported action type!'.format(c.action_type))
        
        #height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_fps(
        #    c, epoch, test_loader, encoder, decoders, pool_layers, N)

        # height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_epoch(
        #     c, epoch, test_loader, encoder, decoders, pool_layers, N)

        height, width, test_image_list, test_dist, gt_label_list, original_sizes = test_meta_epoch(
            c, epoch, test_loader, encoder, decoders, pool_layers, N)

        # 打印当前时间（默认格式）
        print("当前时间2:", datetime.datetime.now())

        # PxEHW
        print('Heights/Widths', height, width)
        #test_map = [list() for p in pool_layers]
        test_map = []  # 直接用列表存储，避免预分配
        upsampled = [list() for p in pool_layers]
        for l, p in enumerate(pool_layers):
            #test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
            test_norm = torch.as_tensor(test_dist[l], dtype=torch.double)  # EHWx1   改到GPU上
            test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_norm) # convert to probs in range [0:1]

            #test_mask = test_prob.reshape(-1, height[l], width[l])
            test_mask = test_prob.view(-1, height[l], width[l])      #reshape改View
            #test_mask = test_prob.reshape(-1, height[l], width[l])
            # upsample
            # test_map[l] = F.interpolate(test_mask.unsqueeze(1),
            #     size=c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
            #直接在GPU上操作
            upsampled = F.interpolate(test_mask.unsqueeze(1),
                size=c.crp_size, mode='bilinear', align_corners=True).squeeze()
            test_map.append(upsampled)  # 不转numpy，保留在GPU
        # score aggregation
        #score_map = np.zeros_like(test_map[0])

        # 打印当前时间（默认格式）
        print("当前时间4:", datetime.datetime.now())

        # for l, p in enumerate(pool_layers):
        #     score_map += test_map[l]

        # 直接在GPU上求和（避免CPU传输）
        score_map = torch.stack(test_map).sum(dim=0)  # 替代np.sum
        #score_mask = score_map

        # invert probs to anomaly scores
        #super_mask = score_mask.max() - score_mask
        super_mask = score_map.max() - score_map  # GPU计算

        # calculate detection AUROC
        # score_label = np.max(super_mask, axis=(1, 2))
        # gt_label = np.asarray(gt_label_list, dtype=bool)
        score_label = super_mask.amax(dim=(1, 2))  # 替代np.max，保持GPU
        gt_label = torch.as_tensor(gt_label_list, dtype=torch.bool, device='cuda')  # GPU


        # 仅最终结果转CPU（如需）
        score_label = score_label.cpu().numpy()
        gt_label = gt_label.cpu().numpy()

        # 打印当前时间（默认格式）
        print("当前时间5:", datetime.datetime.now())

        #det_roc_auc = roc_auc_score(gt_label, score_label)
        # print(f"Detection AUROC: {det_roc_auc:.4f} ({det_roc_auc*100:.2f}%)")
        # _ = det_roc_obs.update(100.0*det_roc_auc, epoch)


        # 获取当前时间
        current_time = datetime.datetime.now()
        # 打印当前时间（默认格式）
        print("当前时间3:", current_time)



        # 计算最小值和最大值
        min_score = np.min(score_label)
        max_score = np.max(score_label)

        range_score = max_score - min_score
        # 归一化处理
        normalized_score = (score_label - min_score) / (range_score)

        # 获取所有 img_paths
        img_paths = test_dataset.x
        
        # 使用自定义函数生成文件名
        file_names = [get_filename_from_path(path) for path in img_paths]
        
        # 设置保存目录
        save_dir = "/mnt/T4_1/xjj/1/cflow-ad/FirstWorkData"
        
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存预测分数到CSV，只包含图片名字和预测分数的一一对应
        csv_path = os.path.join(save_dir, f'anomaly_scores_{c.class_name}_{run_date}.csv')
        
        with open(csv_path, 'w') as f:
            # 写入表头
            f.write("File_Name,Anomaly_Score\n")
            # 使用zip确保一一对应
            for name, score in zip(file_names, normalized_score):
                f.write(f"{name},{score:.10f}\n")  # 使用逗号分隔的表格格式
        
        print(f"异常分数已保存到: {csv_path}")

        # 保存热力图
        # 将super_mask转换为numpy数组（如果需要）
        if hasattr(super_mask, 'cpu'):
            super_mask_np = super_mask.cpu().numpy()
        else:
            super_mask_np = super_mask
        
        # 调用save_heatmaps函数保存热力图
        #save_heatmaps(c, super_mask_np, file_names, original_sizes, c.class_name, save_dir)

        # calculate segmentation AUROC
        #gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
        #seg_roc_auc = roc_auc_score(gt_mask.flatten(), super_mask.flatten())
        # save_best_seg_weights = seg_roc_obs.update(100.0*seg_roc_auc, epoch)
        # if save_best_seg_weights and c.action_type != 'norm-test':
        #     save_weights(encoder, decoders, c.model, run_date)  # avoid unnecessary saves
        # calculate segmentation AUPRO
        # from https://github.com/YoungGod/DFR:
        # if c.pro:  # and (epoch % 4 == 0):  # AUPRO is expensive to compute
        #     max_step = 1000
        #     expect_fpr = 0.3  # default 30%
        #     max_th = super_mask.max()
        #     min_th = super_mask.min()
        #     delta = (max_th - min_th) / max_step
        #     ious_mean = []
        #     ious_std = []
        #     pros_mean = []
        #     pros_std = []
        #     threds = []
        #     fprs = []
        #     binary_score_maps = np.zeros_like(super_mask, dtype=bool)
        #     for step in range(max_step):
        #         thred = max_th - step * delta
        #         # segmentation
        #         binary_score_maps[super_mask <= thred] = 0
        #         binary_score_maps[super_mask >  thred] = 1
        #         pro = []  # per region overlap
        #         iou = []  # per image iou
        #         # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        #         # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
        #         for i in range(len(binary_score_maps)):    # for i th image
        #             # pro (per region level)
        #             label_map = label(gt_mask[i], connectivity=2)
        #             props = regionprops(label_map)
        #             for prop in props:
        #                 x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
        #                 cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
        #                 # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
        #                 cropped_mask = prop.filled_image    # corrected!
        #                 intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
        #                 pro.append(intersection / prop.area)
        #             # iou (per image level)
        #             intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
        #             union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
        #             if gt_mask[i].any() > 0:    # when the gt have no anomaly pixels, skip it
        #                 iou.append(intersection / union)
        #         # against steps and average metrics on the testing data
        #         ious_mean.append(np.array(iou).mean())
        #         #print("per image mean iou:", np.array(iou).mean())
        #         ious_std.append(np.array(iou).std())
        #         pros_mean.append(np.array(pro).mean())
        #         pros_std.append(np.array(pro).std())
        #         # fpr for pro-auc
        #         gt_masks_neg = ~gt_mask
        #         fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        #         fprs.append(fpr)
        #         threds.append(thred)
        #     # as array
        #     threds = np.array(threds)
        #     pros_mean = np.array(pros_mean)
        #     pros_std = np.array(pros_std)
        #     fprs = np.array(fprs)
        #     ious_mean = np.array(ious_mean)
        #     ious_std = np.array(ious_std)
        #     # best per image iou
        #     best_miou = ious_mean.max()
        #     #print(f"Best IOU: {best_miou:.4f}")
        #     # default 30% fpr vs pro, pro_auc
        #     idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
        #     fprs_selected = fprs[idx]
        #     fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
        #     pros_mean_selected = pros_mean[idx]    
        #     seg_pro_auc = auc(fprs_selected, pros_mean_selected)
        #     #_ = seg_pro_obs.update(100.0*seg_pro_auc, epoch)
        
    #
    save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, c.model, c.class_name, run_date)
    # # export visualuzations
    # if c.viz:
    #     precision, recall, thresholds = precision_recall_curve(gt_label, score_label)
    #     a = 2 * precision * recall
    #     b = precision + recall
    #     f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    #     det_threshold = thresholds[np.argmax(f1)]
    #     print('Optimal DET Threshold: {:.2f}'.format(det_threshold))
    #     precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), super_mask.flatten())
    #     a = 2 * precision * recall
    #     b = precision + recall
    #     f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    #     seg_threshold = thresholds[np.argmax(f1)]
    #     print('Optimal SEG Threshold: {:.2f}'.format(seg_threshold))
    #     export_groundtruth(c, test_image_list, gt_mask)
    #     export_scores(c, test_image_list, super_mask, seg_threshold)
    #     export_test_images(c, test_image_list, gt_mask, super_mask, seg_threshold)
    #     export_hist(c, gt_mask, super_mask, seg_threshold)
