import contextlib
import gc
import logging
import os
import sys
import click
import numpy as np
import torch
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
import cv2
import torch.nn.functional as F
from PIL import Image

import pandas as pd

LOGGER = logging.getLogger(__name__)
_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}
@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--score_path", type=str, default=None, help="Path to save anomaly scores as CSV tables (default: results_path)")
def main(**kwargs):
    pass


@main.result_callback()
def run(methods, results_path, gpu, seed, save_segmentation_images, score_path):
    methods = {key: item for (key, item) in methods}
    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    result_collect = []
    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(seed)
    patchcore_iter, n_patchcores = methods["get_patchcore_iter"]
    patchcore_iter = patchcore_iter(device)
    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )
    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )
        patchcore.utils.fix_seeds(seed, device)
        dataset_name = dataloaders["testing"].name
        with device_context:
            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            # 获取所有图片路径（只需要获取一次）
            paths = [item[2] for item in dataloaders["testing"].dataset.data_to_iterate]
            
            # 初始化存储所有模型分数的数据结构
            all_model_scores = []  # 形状: [n_models][n_images]

            aggregator = {"scores": [], "segmentations": []}
            for model_idx, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        model_idx + 1, len(PatchCore_list)
                    )
                )
                #scores, segmentations, labels_gt, masks_gt = PatchCore.predict(dataloaders["testing"])
                scores, segmentations, labels_gt = PatchCore.predict(dataloaders["testing"])
##############################################################################################################TODO   #scores是单个数值，segmentations是（N，320,320）
                # 调试信息：检查scores的结构
                print(f"DEBUG: 模型{model_idx}, len(scores) = {len(scores)}")
                if len(scores) > 0:
                    print(f"DEBUG: 模型{model_idx}, type(scores[0]) = {type(scores[0])}")
                    if hasattr(scores[0], 'shape'):
                        print(f"DEBUG: 模型{model_idx}, scores[0].shape = {scores[0].shape}")
                    else:
                        print(f"DEBUG: 模型{model_idx}, scores[0] is not an array, type = {type(scores[0])}")
                
                # 存储当前模型的分数
                all_model_scores.append(scores)
                
                # 将分数添加到aggregator中用于后续处理（如果需要）
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
            
            # 所有模型处理完成后，按图片组织保存文件
            print(f"\n开始按图片组织保存文件...")
            
            # 定义模型名称映射（根据参数文件中的backbone.name）
            model_names = []
            for model_idx in range(len(PatchCore_list)):
                # 根据模型索引确定模型名称
                if model_idx == 0:
                    model_name = "wideresnet101"
                elif model_idx == 1:
                    model_name = "resnext101"
                elif model_idx == 2:
                    model_name = "densenet201"
                else:
                    model_name = f"model{model_idx}"  # 备用名称
                model_names.append(model_name)
            
            for img_idx, path in enumerate(paths):
                # 获取图片基本名（不带扩展名）
                img_basename = os.path.basename(path).rsplit('.', 1)[0]
                
                # 创建图片文件夹路径
                # 替换基础路径：从原始数据集路径到numpy保存路径
                base_save_path = path.replace('/mnt/T38/bioinf/xjj/Datasets/MVTec-AD/', '/mnt/T4_2/xjj/patchcore/MVTec-AD_numpy/').rsplit('.', 1)[0]
                img_folder_path = base_save_path  # 使用图片名作为文件夹名
                os.makedirs(img_folder_path, exist_ok=True)
                
                # 保存每个模型的分数到图片文件夹，使用真实模型名称
                saved_files = []
                for model_idx in range(len(PatchCore_list)):
                    score_array = all_model_scores[model_idx][img_idx]
                    model_filename = f"{model_names[model_idx]}.npy"
                    model_filepath = os.path.join(img_folder_path, model_filename)
                    
                    np.save(model_filepath, score_array)
                    saved_files.append(model_filename)
                
                print(f"保存成功: 图片 {img_basename} -> 文件夹 {img_folder_path}")
                print(f"  包含文件: {', '.join(saved_files)}")
            
            print(f"总共处理了{len(paths)}张图片")
##############################################################################################################
            '''
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            #预测分数
            scores = np.mean(scores, axis=0)
            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            #分割图
            segmentations = np.mean(segmentations, axis=0)
            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                patchcore.utils.plot_segmentation_images(
                    results_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            # Compute Image-level AUROC scores for all images.
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]
            
            #Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            #Compute PRO score & PW Auroc only for images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs], [masks_gt[i] for i in sel_idxs]
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]
            '''
            auroc=0
            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    #"full_pixel_auroc": full_pixel_auroc,
                    #"anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            del PatchCore_list
            gc.collect()

        LOGGER.info("\n\n-----\n")

    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        results_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core_loader")
# Pretraining-specific parameters.
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            #n_patchcores=3
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name, data_path, subdatasets, batch_size, resize, imagesize, num_workers, augment
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
