import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle

import os

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap



def evaluation(encoder, bn, decoder, dataloader,device,_class_=None):
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        #for img, gt, label, _ in dataloader:
        for batch_idx, (img, label, _, _) in enumerate(dataloader):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
############################################################################################TODO      #(1，256,64,64)  (1,512,32,32)   (1，1024,16,16)
            batch_size = img.size(0)
            for i in range(batch_size):
                # 计算在dataset中的索引
                sample_idx = batch_idx * dataloader.batch_size + i
                file_path = dataloader.dataset.img_paths[sample_idx]
                # 提取图片的基础名字（不带扩展名）
                base_name = os.path.splitext(os.path.basename(file_path))[0]  # 例如 '0001'
                # 构建文件夹路径
                folder_path = file_path.replace(
                    '/mnt/T38/bioinf/xjj/Datasets/MVTec-AD/', 
                    '/mnt/T4_2/xjj/rd4ad/MVTec-AD_numpy/'
                ).rsplit('.', 1)[0]  # 去掉扩展名，得到文件夹路径
                # 创建文件夹
                os.makedirs(folder_path, exist_ok=True)
                # 在文件夹内保存6个npy文件
                np.save(os.path.join(folder_path, f"input_64x64.npy"), inputs[0][i].cpu().detach().numpy())
                np.save(os.path.join(folder_path, f"input_32x32.npy"), inputs[1][i].cpu().detach().numpy())
                np.save(os.path.join(folder_path, f"input_16x16.npy"), inputs[2][i].cpu().detach().numpy())
                np.save(os.path.join(folder_path, f"output_64x64.npy"), outputs[0][i].cpu().detach().numpy())
                np.save(os.path.join(folder_path, f"output_32x32.npy"), outputs[1][i].cpu().detach().numpy())
                np.save(os.path.join(folder_path, f"output_16x16.npy"), outputs[2][i].cpu().detach().numpy())
############################################################################################
        '''
            anomaly_map, a_map_list= cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            #gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            gt_list_sp.append(np.max(label.cpu().numpy().flatten().tolist()))
            pr_list_sp.append(np.max(anomaly_map))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        '''
        auroc_sp=0
        pr_list_sp=0
    #return auroc_px, auroc_sp, round(np.mean(aupro_list),3)
    return  auroc_sp, pr_list_sp



def test(_class_, save_dir=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)
    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '/mnt/T38/bioinf/xjj/Datasets/MVTec-AD/' + _class_  #MVTec-AD/take-image
    #test_path = '/mnt/T4_2/xjj/FirstWorkData1/' + _class_ 
    #ckp_path = '/home/greatwall/XJJ/CheckPoints/RD4AD_CP/' + 'rm_1105_wres50_ff_mm_' + _class_ + '.pth'
    ckp_path = '/mnt/T38/bioinf/xjj/CheckPoints/rd4ad_CP/' + 'wres50_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    auroc_sp, _ = evaluation(encoder, bn, decoder, test_dataloader, device, _class_)
    #auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device,_class_)
    #print(_class_,':',auroc_px,',',auroc_sp,',',aupro_px)
    print(_class_,':',auroc_sp)
    return auroc_sp

def visualization(_class_):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '/home/greatwall/XJJ/Datasets/MVTec-AD/' + _class_
    #test_path = '/home/greatwall/XJJ/Datasets/take-image/' + _class_
    ckp_path = '/home/greatwall/XJJ/CheckPoints/RD4AD_CP/' + 'rm_1105_wres50_ff_mm_'+_class_+'.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            if (label.item() == 0):
                continue
            decoder.eval()
            bn.eval()
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)
            plt.imshow(ano_map)
            plt.axis('off')
            plt.show()
            gt = gt.cpu().numpy().astype(int)[0][0]*255
            count += 1

def vis_nd(name, _class_):
    print(name,':',_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    ckp_path = '/home/greatwall/XJJ/CheckPoints/RD4AD_CP/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)
    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'org.png',img)
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'ad.png', ano_map)
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1
        ano_score = (prmean_list_sp-np.min(prmean_list_sp))/(np.max(prmean_list_sp)-np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)
        with open('vis.pkl','wb') as f:
            pickle.dump(vis_data,f,pickle.HIGHEST_PROTOCOL)

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
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

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

        #df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        new_data = pd.DataFrame({"pro": [mean(pros)], "fpr": [fpr], "threshold": [th]})
        df = pd.concat([df, new_data], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def detection(encoder, bn, decoder, dataloader,device,_class_):
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))#np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))
        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1
        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean
