from operator import index
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# import torchio as tio
import warnings
from glob import glob
import albumentations as A

warnings.filterwarnings("ignore")

class MVTECDataset(Dataset):
    """ABIDE dataset."""

    def __init__(self, mode, object_class, rootdir= './mvtec-dataset/',transform=None,  normal=True, anomaly_class='good', image_size=288, center_size=256, augment=False, center_crop=False):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
        """
        self.mode = mode
        self.center_size = center_size
        if mode == 'train' and normal==False:
            raise Exception('training data should be normal')
        self.augment = augment
        self.normal = normal
        self.center_crop = center_crop
        object_cls_dict = {
        "capsule":0,
        "bottle":1,
        "grid":2,
        "leather":3,
        "metal_nut":4,
        "tile":5,
        "transistor":6,
        "zipper":7,
        "cable":8,
        "carpet":9,
        "hazelnut":10,
        "pill":11,
        "screw":12,
        "toothbrush":13,
        "wood":14,
        }
        self.anomaly_class = anomaly_class
        self.transform = transform
        self.object_class = object_class
        self.image_size = image_size
        
        #df = pd.read_csv(os.path.join('.', 'splits', 'mvtec-split.csv'))    
        df = pd.read_csv(os.path.join('.', 'splits', 'firstworkdata-split.csv'))
        #df = pd.read_csv(os.path.join('.', 'splits', 'good_dataset_split.csv'))

        if object_class == 'all':
            df = df.query(f'split=="{mode}"')    
        else:
            df = df.query(f'split=="{mode}" and object=="{object_class}"')    
              
            
        if anomaly_class=='good' or self.normal:
            df = df.query(f'category=="good"') 
        elif anomaly_class=='all': 
            pass
        else:
            df = df.query(f'category=="{anomaly_class}"')    
            
        if len(df)==0 :
            raise Exception('No data found')


        self.images = []
        self.image_paths = []
        self.segs = []
        self.object_classes = []
        for i, row in df.iterrows():
            data_path = os.path.join(rootdir, row['image'])
            img = np.array(Image.open(data_path).convert('RGB').resize((self.image_size, self.image_size))).astype(np.uint8)
            
            self.images.append(img)
            self.image_paths.append(data_path)
            self.object_classes.append(object_cls_dict[row['object']])
            if row['category']!='good':
                # seg_path = os.path.join(rootdir, row['mask'])
                # seg = (np.array(Image.open(seg_path).convert('L').resize((self.image_size, self.image_size)))>0).astype(np.uint8)
                # self.segs.append((seg))
                # 第一步：检查mask值是否为空/非字符串
                mask_val = row['mask']
                # 核心修改：异常样本默认掩码改为全1（dtype保持uint8）
                seg = np.ones((self.image_size, self.image_size), dtype=np.uint8)  # 全1数组
                
                # 仅当mask值非空且为有效字符串时，才尝试加载
                if not pd.isna(mask_val) and isinstance(mask_val, str) and mask_val.strip() != '':
                    try:
                        seg_path = os.path.join(rootdir, mask_val)
                        # 检查路径是否存在
                        if os.path.exists(seg_path):
                            # 加载并处理掩码
                            with Image.open(seg_path).convert('L') as img:
                                img_resized = img.resize((self.image_size, self.image_size))
                                seg = (np.array(img_resized) > 0).astype(np.uint8)
                        else:
                            print(f"警告：掩码文件不存在 - {seg_path}，使用空掩码")
                    except Exception as e:
                        # 捕获所有加载错误（文件损坏/路径错误等）
                        print(f"警告：加载掩码失败 {seg_path} - {str(e)}，使用空掩码")
                
                self.segs.append(seg)
            else:
                self.segs.append(np.zeros((self.image_size, self.image_size)))
        if self.augment:
           self.aug = A.Compose([
                A.Affine (translate_px=int(self.image_size/8 - self.center_size/8), p=0.5),
                A.CenterCrop(p=1, height=self.center_size, width=self.center_size)])
        else:
            self.aug = A.CenterCrop(p=1, height=self.center_size, width=self.center_size)

    def transform_volume(self, x):
        x = torch.from_numpy(x.transpose((-1, 0 , 1)))
        return x

    def __len__(self):
        return len(self.images)
        

    def __getitem__(self, index):
        img = self.images[index].astype(np.uint8)
        seg = self.segs[index].astype(np.int32)
        if self.center_crop:
            augmented = self.aug(image=img, mask=seg)
            img = augmented['image']
            seg = augmented['mask']
            
        img = img.astype(np.float32) / 255.0
        y = self.object_classes[index]
        if self.transform:
            img = self.transform(img)
        else:
            img = self.transform_volume(img)
            img = (img-0.5)/0.5
        return img, seg.astype(np.float32), int(y), self.image_paths[index]
