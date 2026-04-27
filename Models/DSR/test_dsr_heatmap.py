import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from dsr_model import SubspaceRestrictionModule, ImageReconstructionNetwork, AnomalyDetectionModule, UpsamplingModule
from discrete_model import DiscreteLatentModel
import sys
from sklearn.metrics import roc_auc_score, average_precision_score
from data_loader_test import TestMVTecDataset
import cv2

import torch.nn.functional as F

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

def save_heatmaps(c, anomaly_score_map, file_names, original_sizes, class_name, save_dir):
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

def crop_image(image, img_dim):
    b,c,h,w = image.shape
    hdif = max(0,h - img_dim) // 2
    wdif = max(0,w - img_dim) // 2
    image_cropped = image[:,:,hdif:-hdif,wdif:-wdif]
    return image_cropped

def evaluate_model(model, model_normal, model_normal_top, model_decode, decoder_seg, model_upsample, obj_name, mvtec_path, cnt_total):
    img_dim = 256
    dataset = TestMVTecDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim,img_dim])

    test_dir = mvtec_path + obj_name + "/test/"
    print(f"数据路径: {test_dir} | 路径存在: {os.path.exists(test_dir)} | 数据集样本数: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)


    img_dim = 224
    total_pixel_scores = np.zeros((img_dim * img_dim * 500))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * 500))
    mask_cnt = 0

    total_gt = []
    total_score = []
    iter = cnt_total

    anomaly_maps = []  # 存储异常图
    original_sizes = []  # 存储原始图片大小

    for i_batch, sample_batched in enumerate(dataloader):

        gray_batch = sample_batched["image"].cuda()

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0,0]
        total_gt.append(is_normal)
        # true_mask = sample_batched["mask"]
        # true_mask = crop_image(true_mask, img_dim)
        # true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

        loss_b, loss_t, data_recon, embeddings_t, embeddings = model(gray_batch)
        embeddings = embeddings.detach()
        embeddings_t = embeddings_t.detach()

        embedder = model._vq_vae_bot
        embedder_top = model._vq_vae_top

        anomaly_embedding_copy = embeddings.clone()
        anomaly_embedding_top_copy = embeddings_t.clone()
        recon_feat, recon_embeddings, _ = model_normal(anomaly_embedding_copy, embedder)
        recon_feat_top, recon_embeddings_top, loss_b_top = model_normal_top(anomaly_embedding_top_copy,
                                                                            embedder_top)

        up_quantized_recon_t = model.upsample_t(recon_embeddings_top)
        quant_join = torch.cat((up_quantized_recon_t, recon_embeddings), dim=1)
        recon_image_recon = model_decode(quant_join)

        up_quantized_embedding_t = model.upsample_t(embeddings_t)
        quant_join_real = torch.cat((up_quantized_embedding_t, embeddings), dim=1)
        recon_image = model._decoder_b(quant_join_real)
        out_mask = decoder_seg(recon_image_recon.detach(),
                               recon_image.detach())
        out_mask_sm = torch.softmax(out_mask, dim=1)

        upsampled_mask = model_upsample(recon_image_recon.detach(), recon_image.detach(), out_mask_sm)

        
        out_mask_sm_up = torch.softmax(upsampled_mask, dim=1)
        out_mask_sm_up = crop_image(out_mask_sm_up, img_dim)

        iter += 1


        out_mask_cv = out_mask_sm_up[0,1,:,:].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:,1:,:,:], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        total_score.append(image_score)

        anomaly_maps.append(out_mask_cv)  # 保存异常图
        # 获取原始图片大小
        original_size = sample_batched["original_size"].detach().numpy()[0]
        original_sizes.append((original_size[0], original_size[1]))  # (width, height)


        #flat_true_mask = true_mask_cv.flatten()
        # flat_out_mask = out_mask_cv.flatten()
        # total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        # total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        # mask_cnt += 1

    
   

    total_score = np.array(total_score)
    total_gt = np.array(total_gt)
    #auroc = roc_auc_score(total_gt, total_score)
    auroc=0

    # total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    # total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    # total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    # auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    # ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    # ap = average_precision_score(total_gt, total_score)
    # print(obj_name+" AUC Image: "+str(auroc)+",  AUC Pixel: "+str(auroc_pixel)+", AP Pixel:"+str(ap_pixel)+", AP :"+str(ap))

 # 保存预测分数到CSV
    # 直接新定义一个变量接受地址
    save_dir = "/mnt/T4_1/xjj/2/DSR/MVTec-AD/"  # 直接定义保存目录FirstWorkData1
    
    # 获取所有 img_paths
    img_paths = dataset.images
    # 归一化预测分数，避免除以零
    # if np.max(total_score) > np.min(total_score):
    #     pr_list_sp_norm = (total_score - np.min(total_score)) / (np.max(total_score) - np.min(total_score))
    # else:
    #     pr_list_sp_norm = np.zeros_like(total_score)
    
    # 使用自定义函数生成文件名
    file_names = [get_filename_from_path(path) for path in img_paths]
    
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存预测分数到CSV，只包含图片名字和预测分数的一一对应
    csv_path = os.path.join(save_dir, f'anomaly_scores_{obj_name}.csv')

    # 使用assert语句检查分数范围，如果为0就停止 
    score_range = np.max(total_score) - np.min(total_score) 
    # 最大最小归一化 
    normalized_scores = (total_score - np.min(total_score)) / score_range 
    
    with open(csv_path, 'w') as f:
        # 写入表头
        f.write("File_Name,Anomaly_Score\n")
        # 使用zip确保一一对应
        for name, score in zip(file_names, normalized_scores):
            f.write(f"{name},{score:.10f}\n")  # 使用逗号分隔的表格格式
    
    print(f"异常分数已保存到: {csv_path}")

    # 保存热力图
    if len(anomaly_maps) > 0:
        # 转换异常图为numpy数组
        anomaly_score_map = np.array(anomaly_maps)  # (N, H, W)
        # 调用保存热力图函数，c参数设为None
        save_heatmaps(None, anomaly_score_map, file_names, original_sizes, obj_name, save_dir)
    else:
        print("警告：未生成异常图，跳过热力图保存")


    #return auroc, auroc_pixel, ap_pixel, ap, iter
    return auroc, iter,obj_name,total_score

def train_on_device(obj_names, mvtec_path, run_basename):
    auroc_list = []
    auroc_pixel_list = []
    ap_pixel_list = []
    ap_list = []
    cnt_total = 0
    for obj_name in obj_names:
        run_name_pre = 'vq_model_pretrained_128_4096'

        run_name = run_basename+'_'

        num_hiddens = 128
        num_residual_hiddens = 64
        num_residual_layers = 2
        embedding_dim = 128
        num_embeddings = 4096
        commitment_cost = 0.25
        decay = 0.99
        model_vq = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                      num_embeddings, embedding_dim,
                      commitment_cost, decay)
        model_vq.cuda()
        model_vq.load_state_dict(
            torch.load("/mnt/T38/bioinf/xjj/CheckPoints/dsr_CP/" + run_name_pre + ".pckl", map_location='cuda:0'))
        model_vq.eval()



        sub_res_hi_module = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_hi_module.load_state_dict(
            torch.load("/mnt/T38/bioinf/xjj/CheckPoints/dsr_CP/" + run_name + "subspace_restriction_hi_"+obj_name+".pckl", map_location='cuda:0'))
        sub_res_hi_module.cuda()
        sub_res_hi_module.eval()

        sub_res_lo_module = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_lo_module.load_state_dict(
            torch.load("/mnt/T38/bioinf/xjj/CheckPoints/dsr_CP/" + run_name + "subspace_restriction_lo_"+obj_name+".pckl", map_location='cuda:0'))
        sub_res_lo_module.cuda()
        sub_res_lo_module.eval()


        anom_det_module = AnomalyDetectionModule(embedding_size=embedding_dim)
        anom_det_module.load_state_dict(
            torch.load("/mnt/T38/bioinf/xjj/CheckPoints/dsr_CP/" + run_name + "anomaly_det_module_"+obj_name+".pckl", map_location='cuda:0'))
        anom_det_module.cuda()
        anom_det_module.eval()

        upsample_module = UpsamplingModule(embedding_size=embedding_dim)
        upsample_module.load_state_dict(
            torch.load("/mnt/T38/bioinf/xjj/CheckPoints/dsr_CP/" + run_name + "upsample_module_"+obj_name+".pckl", map_location='cuda:0'))
        upsample_module.cuda()
        upsample_module.eval()


        image_recon_module = ImageReconstructionNetwork(embedding_dim * 2,
                   num_hiddens,
                   num_residual_layers,
                   num_residual_hiddens)
        image_recon_module.load_state_dict(
            torch.load("/mnt/T38/bioinf/xjj/CheckPoints/dsr_CP/" + run_name + "image_recon_module_"+obj_name+".pckl", map_location='cuda:0'), strict=False)
        image_recon_module.cuda()
        image_recon_module.eval()


        with torch.no_grad():
            #auroc, auroc_pixel, ap_pixel, ap, cnt = evaluate_model(model_vq, sub_res_hi_module, sub_res_lo_module, image_recon_module, anom_det_module, upsample_module, obj_name, mvtec_path, cnt_total)
            auroc, cnt ,obj_name,total_score= evaluate_model(model_vq, sub_res_hi_module, sub_res_lo_module, image_recon_module, anom_det_module, upsample_module, obj_name, mvtec_path, cnt_total)
            cnt_total += cnt
            #ap_list.append(ap)
            auroc_list.append(auroc)
            #auroc_pixel_list.append(auroc_pixel)
            #ap_pixel_list.append(ap_pixel)

            print(f"类别 {obj_name} | 检测AUROC: {auroc:.4f}")

    print(run_basename)
    auroc_mean = np.mean(auroc_list)
    #auroc_pixel_mean = np.mean(auroc_pixel_list)
    print("Detection AUROC: "+str(auroc_mean))
    # print("Localization AUROC: "+str(auroc_pixel_mean))
    # print("Localization AP: "+str(np.mean(ap_pixel_list)))


if __name__=="__main__":
    obj_names = ['capsule', 'bottle', 'grid', 'leather', 'pill', 'tile', 'transistor', 'zipper', 'cable', 'carpet',
                  'hazelnut', 'metal_nut', 'screw', 'toothbrush', 'wood']
    # obj_names = [ 'pill' ,'zipper']

    list1 = sys.argv[1].split(',')

    with torch.cuda.device(int(sys.argv[1])):
        train_on_device(obj_names, sys.argv[2], sys.argv[3])

