import argparse
import json
import os
import re
import sys
import torch
import numpy as np
import time
import logging
import imageio
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import evaluate_model
from dataset import FullDataset
from dino_unet import DINOv3_S_UNet

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def clean_path(path):
    """Clean path by removing extra quotes and whitespace."""
    if isinstance(path, str):
        # Remove quotes if present
        if (path.startswith('"') and path.endswith('"')) or \
           (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        # Strip whitespace
        path = path.strip()
    return path


def sanitize_filename(name):
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('_')
    return safe_name or 'dataset'


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def process_dataset(checkpoint, image_path, gt_path, save_base_path, dataset_name, device, save_results, target_size=224, dino_pretrained=True, save_orig_size=False, use_dilation=False):
    # Create save directory for this dataset
    save_path = os.path.join(save_base_path, dataset_name)
    if save_results.lower() == "true":
        os.makedirs(save_path, exist_ok=True)
    
    # Additional path cleaning
    image_path = clean_path(image_path)
    gt_path = clean_path(gt_path)
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Image path: {image_path}")
    print(f"GT path: {gt_path}")
    print(f"Save path: {save_path}")
    
    # Check path existence before loading dataset
    if not os.path.exists(image_path):
        print(f"Error: Image directory does not exist: {image_path}")
        return 0.0, 0.0
    
    if not os.path.exists(gt_path):
        print(f"Error: Mask directory does not exist: {gt_path}")
        return 0.0, 0.0
    
    # List directory contents for debugging
    try:
        image_files = os.listdir(image_path)
        mask_files = os.listdir(gt_path)
        print(f"Found {len(image_files)} files in image directory")
        print(f"Found {len(mask_files)} files in mask directory")
    except Exception as e:
        print(f"Error listing directory contents: {e}")
    
    # Load model (可配置是否使用预训练权重)
    print("Loading model...")
    model = DINOv3_S_UNet(pretrained=dino_pretrained, use_dilation=use_dilation).to(device)
    # Only load local checkpoint
    try:
        checkpoint_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_dict, strict=False)
        print(f"Successfully loaded checkpoint from {checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0.0, 0.0
    
    # 加载测试数据集，使用与训练时相同的尺寸
    print(f"Loading dataset with FullDataset using size: {target_size}x{target_size}")
    test_dataset = FullDataset(image_path, gt_path, target_size, mode='val')
    
    # 检查数据集是否为空
    if len(test_dataset) == 0:
        print(f"Error: Test dataset {dataset_name} is empty!")
        return 0.0, 0.0
    
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=8)
    print(f"Dataset loaded: {len(test_dataset)} images found")
    
    # 计算评估指标（返回字典，包含 mean + CI95）
    print(f"Calculating evaluation metrics for dataset: {dataset_name}")
    results = evaluate_model(model, test_loader, device)
    dice_mean = results.get('Dice', {}).get('mean', 0.0)
    dice_ci95 = results.get('Dice', {}).get('CI95', (0.0, 0.0))
    hd95_mean = results.get('HD95', {}).get('mean', 0.0)
    hd95_ci95 = results.get('HD95', {}).get('CI95', (0.0, 0.0))
    ece_mean = results.get('ECE', {}).get('mean', 0.0)
    ece_ci95 = results.get('ECE', {}).get('CI95', (0.0, 0.0))
    print(f"Dice Mean: {dice_mean:.4f}, Dice CI95: ({dice_ci95[0]:.4f}, {dice_ci95[1]:.4f})")
    print(f"HD95 Mean: {hd95_mean:.4f}, HD95 CI95: ({hd95_ci95[0]:.4f}, {hd95_ci95[1]:.4f})")
    print(f"ECE Mean: {ece_mean:.4f}, ECE CI95: ({ece_ci95[0]:.4f}, {ece_ci95[1]:.4f})")
    
    # 保存预测结果（如果需要）
    if save_results.lower() == "true":
        model.eval()
        for i, batch in enumerate(tqdm(test_loader, desc='Saving predictions', unit='image')):
            with torch.no_grad():
                image = batch['image'].to(device=device)
                name = batch.get('filename', [f'image_{i}'])[0]
                # print('Processing image:', name)
                
                # Forward pass
                res = model(image)
                if type(res) == type([]):
                    res = res[0]
                
                # 后处理和保存
                res_sigmoid = res.sigmoid().data.cpu()
                res_np = res_sigmoid.numpy().squeeze()
                res_normalized = (res_np - res_np.min()) / (res_np.max() - res_np.min() + 1e-8)
                res_uint8 = (res_normalized * 255).astype(np.uint8)

                # Optional: resize prediction back to original image size
                if save_orig_size:
                    try:
                        orig_size = batch.get('orig_size', None)
                        w, h = orig_size[0], orig_size[1]
                        if (res_uint8.shape[1], res_uint8.shape[0]) != (w, h):
                            res_uint8 = np.array(
                                Image.fromarray(res_uint8).resize((w, h), resample=Image.NEAREST)
                            ).astype(np.uint8)
                    except Exception as e:
                        print(f"Warning: Failed to resize {name} to orig_size: {e}")
                
                # 使用从数据集中获取的原始文件名（不包含扩展名）
                try:
                    imageio.imsave(os.path.join(save_path, name), res_uint8)
                    # print(f"Saved prediction: {output_filename}")
                except Exception as e:
                    print(f"Error saving prediction for {name}: {e}")
    print(f"Dataset {dataset_name} processing completed.")
    
    return results

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser("DINOV3-UNet Test")
    parser.add_argument("--checkpoint", "--checkpoint_path", type=str, required=True,
                    help="path to the checkpoint of dino_unet")
    parser.add_argument('--test_image_paths', '--image_paths', type=str, action='append', default=[],
                        help='paths to the test image directories (can be used multiple times)')
    parser.add_argument('--test_gt_paths', '--gt_paths', '--test_mask_paths', type=str, action='append', default=[],
                        help='paths to the test mask directories (can be used multiple times)')
    parser.add_argument('--test_dataset_names', '--dataset_names', type=str, action='append', default=[],
                        help='names of the test datasets (can be used multiple times)')
    parser.add_argument("--save_path", type=str, default="./predictions",
                        help="base path to save the predicted masks")
    parser.add_argument("--save_results", type=str, default="true",
                        help="Whether to save prediction results (true/false)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="input image size for dataset (default: 224)")
    parser.add_argument('--dino_pretrained', type=str, default='True',
                        help='whether to load pretrained weights for the DINO backbone (True/False)')
    parser.add_argument(
        "--save_orig_size",
        type=str,
        default="false",
        help="Whether to resize predicted masks back to original image size before saving (true/false).",
    )
    parser.add_argument("--device", type=str, default=None,
                        help='device to use for testing (e.g., "cuda", "cuda:0", "cpu"). If not specified, will use cuda if available, otherwise cpu')
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory to save log files")
    parser.add_argument('--use_dilation', type=str, default='False',
                        help='whether to use dilation layers in the model (True/False)')
    args = parser.parse_args()
    # normalize boolean-like arg
    args.dino_pretrained = str(args.dino_pretrained).lower() in ('true', '1', 'yes', 'y')
    args.save_orig_size = str(args.save_orig_size).lower() in ('true', '1', 'yes', 'y')
    args.use_dilation = str(args.use_dilation).lower() in ('true', '1', 'yes', 'y')
    
    
    # Configure logging system
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"test_{timestamp}.log")
    metrics_json_file = os.path.join(args.log_dir, f"test_{timestamp}_metrics.json")

    # Set up logger
    sys.stdout = Logger(log_file)
    
    # 设置设备：优先使用命令行参数，否则自动选择
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Image size: {args.img_size} x {args.img_size}")
    print(f"DINO pretrained: {args.dino_pretrained}")
    print(f"Use Dilation: {args.use_dilation}")
    print(f"Save pred masks in orig size: {args.save_orig_size}")
    
    # Log configuration
    print(f"Checkpoint path: {args.checkpoint}")
    print(f"Save path: {args.save_path}")
    print(f"Save results: {args.save_results}")
    print(f"Test dataset names: {args.test_dataset_names}")
    
    # Ensure test dataset paths are provided
    if not args.test_image_paths or not args.test_gt_paths:
        print("Error: No test datasets provided. Please use --test_image_paths and --test_gt_paths arguments.")
        return
    
    # Clean test paths by removing any extra quotes
    args.test_image_paths = [clean_path(path) for path in args.test_image_paths]
    args.test_gt_paths = [clean_path(path) for path in args.test_gt_paths]
    
    # Log information about test paths
    print(f"Number of test image paths: {len(args.test_image_paths) if args.test_image_paths else 0}")
    if args.test_image_paths:
        print(f"Test image paths: {args.test_image_paths}")
        # Validate each path and log existence
        for i, path in enumerate(args.test_image_paths):
            if os.path.exists(path):
                print(f"Image path {i+1} exists: {path}")
            else:
                print(f"Warning: Image path {i+1} does not exist: {path}")
    
    print(f"Number of test ground truth paths: {len(args.test_gt_paths) if args.test_gt_paths else 0}")
    if args.test_gt_paths:
        print(f"Test ground truth paths: {args.test_gt_paths}")
        # Validate each path and log existence
        for i, path in enumerate(args.test_gt_paths):
            if os.path.exists(path):
                print(f"GT path {i+1} exists: {path}")
            else:
                print(f"Warning: GT path {i+1} does not exist: {path}")
    
    # Ensure test image paths and mask paths数量匹配
    if len(args.test_image_paths) != len(args.test_gt_paths):
        print(f"Warning: Number of test image paths ({len(args.test_image_paths)}) and mask paths ({len(args.test_gt_paths)}) do not match.")
        print("Using the minimum number of pairs.")
        min_len = min(len(args.test_image_paths), len(args.test_gt_paths))
        args.test_image_paths = args.test_image_paths[:min_len]
        args.test_gt_paths = args.test_gt_paths[:min_len]
    
    # Create base save directory
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Created base save directory: {args.save_path}")
    
    # Storage for all datasets' metrics
    all_metrics = []
    
    # Process each test dataset
    start_time = time.time()
    for i, (img_path, gt_path) in enumerate(zip(args.test_image_paths, args.test_gt_paths)):
        # Use provided dataset name if available, otherwise use default naming
        if i < len(args.test_dataset_names) and args.test_dataset_names[i]:
            dataset_name = args.test_dataset_names[i]
        else:
            dataset_name = f"Test_Set_{i+1}"
        
        print(f"\nProcessing dataset {i+1}/{len(args.test_image_paths)}")
        result = process_dataset(
            args.checkpoint,
            img_path,
            gt_path,
            args.save_path,
            dataset_name,
            device,
            args.save_results,
            args.img_size,
            args.dino_pretrained,
            args.save_orig_size,
            args.use_dilation,
        )
        all_metrics.append((dataset_name, result))

        dataset_safe_name = sanitize_filename(dataset_name)
        dataset_metrics_json_file = os.path.join(args.log_dir, f"test_{timestamp}_{dataset_safe_name}_metrics.json")
        dice = result.get('Dice', {})
        hd95 = result.get('HD95', {})
        dataset_metrics_payload = {
            "timestamp": timestamp,
            "log_file": log_file,
            "checkpoint": args.checkpoint,
            "save_path": args.save_path,
            "dataset_name": dataset_name,
            "dice": {
                "mean": dice.get('mean', 0.0),
                "ci95": list(dice.get('CI95', (0.0, 0.0))),
                "values": dice.get('values', []),
            },
            "hd95": {
                "mean": hd95.get('mean', 0.0),
                "ci95": list(hd95.get('CI95', (0.0, 0.0))),
                "values": hd95.get('values', []),
            },
        }
        with open(dataset_metrics_json_file, "w", encoding="utf-8") as f:
            json.dump(dataset_metrics_payload, f, ensure_ascii=False, indent=2, default=json_default)
        print(f"Metrics JSON file location: {dataset_metrics_json_file}")
    
    total_time = time.time() - start_time
    print(f"All datasets processed in {total_time:.2f} seconds")
    
    # Print summary of all datasets
    print("\n===== Summary of All Datasets =====")
    for dataset_name, result in all_metrics:
        dice_mean = result.get('Dice', {}).get('mean', 0.0)
        dice_ci95 = result.get('Dice', {}).get('CI95', (0.0, 0.0))
        hd95_mean = result.get('HD95', {}).get('mean', 0.0)
        hd95_ci95 = result.get('HD95', {}).get('CI95', (0.0, 0.0))
        ece_mean = result.get('ECE', {}).get('mean', 0.0)
        ece_ci95 = result.get('ECE', {}).get('CI95', (0.0, 0.0))
        print(f"Dataset: {dataset_name}")
        print(f"  Dice Mean: {dice_mean:.4f}  CI95: ({dice_ci95[0]:.4f}, {dice_ci95[1]:.4f})")
        print(f"  HD95 Mean: {hd95_mean:.4f}  CI95: ({hd95_ci95[0]:.4f}, {hd95_ci95[1]:.4f})")
        print(f"  ECE  Mean: {ece_mean:.4f}  CI95: ({ece_ci95[0]:.4f}, {ece_ci95[1]:.4f})")

    print("\nAll datasets processing completed successfully!")
    print(f"Log file location: {log_file}")
    
    # Close logger properly
    try:
        if hasattr(sys.stdout, 'close') and hasattr(sys.stdout, '_closed') and not sys.stdout._closed:
            sys.stdout.close()
    except Exception as e:
        # Just handle the exception to avoid script failure
        pass

if __name__ == "__main__":
    main()