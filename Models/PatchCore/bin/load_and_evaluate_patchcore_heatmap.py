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

LOGGER = logging.getLogger(__name__)

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
    
    # If score_path is not provided, use results_path as default
    if score_path is None:
        score_path = results_path
    else:
        os.makedirs(score_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
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

            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                #scores, segmentations, labels_gt, masks_gt = PatchCore.predict(dataloaders["testing"]
                scores, segmentations, labels_gt,_ = PatchCore.predict(dataloaders["testing"])
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
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
            segmentations = np.mean(segmentations, axis=0)


            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # Plot Example Images.
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
            # auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            #     scores, anomaly_labels
            # )["auroc"]

            # -------------------- 提取图片路径 --------------------
            # 获取当前数据集（MVTecDataset实例）
            test_dataset = dataloaders["testing"].dataset
            # 从 data_to_iterate 中提取所有图片路径（每个元素为 [classname, anomaly, image_path]）
            all_image_paths = [item[2] for item in test_dataset.data_to_iterate]
            file_names = [get_filename_from_path(path) for path in all_image_paths]
            
            # 使用assert语句检查分数范围，如果为0就停止
            score_range = np.max(scores) - np.min(scores)
            assert score_range > 0, f"分数范围为零，无法归一化。所有分数相同: {scores[0]}"
            
            # 最大最小归一化
            normalized_scores = (scores - np.min(scores)) / score_range

            # Save anomaly scores to CSV table in score_path directory
            score_filename = f"anomaly_scores_{dataset_name}.csv"
            csv_file_path = os.path.join(score_path, score_filename)
            
            with open(csv_file_path, 'w') as f:
                # Write header for table format
                f.write("File_Path,Anomaly_Score\n")
                for name, score in zip(file_names, normalized_scores):
                    f.write(f"{name},{score:.10f}\n")  # 使用逗号分隔的表格格式
            
            LOGGER.info(f"Anomaly scores saved to: {csv_file_path}")

            # Save heatmaps for anomaly visualization
            # Convert segmentations numpy array to torch tensor
            # segmentations shape: [num_images, 1, H, W]
            segmentations_tensor = torch.from_numpy(segmentations)
            save_heatmaps(segmentations_tensor, all_image_paths, score_path, dataset_name)
            # Compute PRO score & PW Auroc for all images
            # pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            #     segmentations, masks_gt
            # )
            # full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only for images with anomalies
            # sel_idxs = []
            # for i in range(len(masks_gt)):
            #     if np.sum(masks_gt[i]) > 0:
            #         sel_idxs.append(i)
            # pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            #     [segmentations[i] for i in sel_idxs], [masks_gt[i] for i in sel_idxs]
            # )
            #anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    #"instance_auroc": auroc,
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
