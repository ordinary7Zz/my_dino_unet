import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import tensorboardX
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from dino_unet import DINOv3_S_UNet
from utils.metrics import evaluate_model
from utils.loss import structure_loss

parser = argparse.ArgumentParser("DINOV3-UNet with multi-dataset validation")
parser.add_argument("--method", type=str, required=True)
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument("--test_image_paths", type=str, nargs='+', required=True,
                    help="paths to the test image datasets")
parser.add_argument("--test_mask_paths", type=str, nargs='+', required=True,
                    help="paths to the test mask datasets")
parser.add_argument("--test_dataset_names", type=str, nargs='+', required=True,
                    help="names of the test datasets")
parser.add_argument("--epoch", type=int, default=50, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--weight_decay", default=5e-4, type=float, 
                    help="weight decay for the optimizer")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument('--dir_checkpoint', type=str, default='/checkpoint/')
parser.add_argument('--checkpoint_interval', type=int, default=1, 
                    help='interval between saving checkpoints')
parser.add_argument('--eval_interval', type=int, default=5, 
                    help='interval between evaluating metrics')
parser.add_argument('--dataset_name', type=str, default='default', 
                    help='name of the dataset for logging and checkpoint naming')
parser.add_argument('--img_size', type=int, default=224,
                    help='input image size for dataset (default: 224)')
parser.add_argument('--dino_pretrained', type=str, default='True',
                    help='whether to load pretrained weights for the DINO backbone (True/False)')
parser.add_argument('--use_dilation', type=str, default='False',
                    help='whether to use dilation layers in the model (True/False)')
parser.add_argument('--device', type=str, default=None,
                    help='device to use for training (e.g., "cuda", "cuda:0", "cpu"). If not specified, will use cuda if available, otherwise cpu')
args = parser.parse_args()
# normalize boolean-like arg
args.dino_pretrained = str(args.dino_pretrained).lower() in ('true', '1', 'yes', 'y')
args.use_dilation = str(args.use_dilation).lower() in ('true', '1', 'yes', 'y')

def log_print(message, log_file=None):
    print(message, end='')
    if log_file is not None:
        log_file.write(message)
        log_file.flush() 

def main(args):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join("logs", args.dataset_name, f'{args.method}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    cur_time = time.ctime()
    log_name = os.path.join(save_dir, f'{args.method}_{args.dataset_name}_log.log')
    
    log_file = open(log_name, 'w')
    
    try:
        # 记录开始训练信息
        log_print(f"Training started at {cur_time}\n", log_file)
        log_print(f"Method: {args.method}\n", log_file)
        log_print(f"Dataset: {args.dataset_name}\n", log_file)
        log_print(f"Epochs: {args.epoch}\n", log_file)
        log_print(f"Learning Rate: {args.lr}\n", log_file)
        log_print(f"Batch Size: {args.batch_size}\n", log_file)
        log_print(f"Train Image Path: {args.train_image_path}\n", log_file)
        log_print(f"Train Mask Path: {args.train_mask_path}\n", log_file)
        log_print(f"Image Size: {args.img_size} x {args.img_size}\n", log_file)
        log_print(f"DINO pretrained: {args.dino_pretrained}\n", log_file)
        log_print(f"Use Dilation: {args.use_dilation}\n", log_file)
        # 记录测试数据集信息
        log_print(f"Number of test datasets: {len(args.test_dataset_names)}\n", log_file)
        for i, (name, img_path, mask_path) in enumerate(zip(args.test_dataset_names, args.test_image_paths, args.test_mask_paths)):
            log_print(f"Test Dataset {i+1}: {name}\n", log_file)
            log_print(f"  Image path: {img_path}\n", log_file)
            log_print(f"  Mask path: {mask_path}\n", log_file)
        
        dataset = FullDataset(args.train_image_path, args.train_mask_path, args.img_size, mode='train')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        # 设置设备：优先使用命令行参数，否则自动选择
        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_print(f"Using device: {device}\n", log_file)
        
        model = DINOv3_S_UNet(pretrained=args.dino_pretrained, use_dilation=args.use_dilation)
        model.to(device)
        
        optim = opt.AdamW([{"params":model.parameters(), "initial_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
        
        loss_fn = structure_loss
          
        writer = tensorboardX.SummaryWriter(os.path.join(save_dir, 'tensorboard_logs'))

        # 初始化指标数组，用于存储每个epoch的验证结果
        epoch_dice_scores = {}
        epoch_hd95_scores = {}
        epoch_ece_scores = {}
        for dataset_name in args.test_dataset_names:
            epoch_dice_scores[dataset_name] = []
            epoch_hd95_scores[dataset_name] = []
            epoch_ece_scores[dataset_name] = []

        global_step = 0
        for epoch in range(args.epoch):
            # 训练循环
            model.train()
            epoch_loss = 0
            for i, batch in enumerate(dataloader):
                x = batch['image']
                target = batch['label']
                x = x.to(device)
                target = target.to(device)
                optim.zero_grad()

                pred0 = model(x)
                loss0 = loss_fn(pred0, target)
                loss = loss0

                loss.backward()
                optim.step()
                global_step += 1
                epoch_loss += loss0.item()
                writer.add_scalar('TRAIN_LOSS', loss0.item(), global_step)
                writer.add_scalar('LR', scheduler.get_lr()[0], global_step)
            
            scheduler.step()
            avg_epoch_loss = epoch_loss / len(dataloader)
            
            # 保存检查点
            checkpoint_path = os.path.join(args.dir_checkpoint, args.dataset_name, timestamp)
            os.makedirs(checkpoint_path, exist_ok=True)
            log_print(f"Epoch:{epoch+1}: loss:{avg_epoch_loss:.6f}\n", log_file)
            if (epoch+1) % args.checkpoint_interval == 0 or epoch == args.epoch-1:
                torch.save(model.state_dict(), str(checkpoint_path + '/' + args.method + "_" + args.dataset_name + "_" +'epoch_' + str(epoch+1) + '.pth'))
                log_print(f"Saved checkpoint at epoch {epoch+1}\n", log_file)
            
            if(epoch+1) % args.eval_interval == 0:
                # 在每个eval_interval epoch后对所有测试数据集进行验证
                log_print(f"\nValidating epoch {epoch+1} on all test datasets...\n", log_file)
                # evaluate_model 函数内部会处理模型状态（eval/train）
                with torch.no_grad():
                    for i, (dataset_name, img_path, mask_path) in enumerate(zip(args.test_dataset_names, args.test_image_paths, args.test_mask_paths)):
                        log_print(f"Testing on {dataset_name}...\n", log_file)
                        # 创建测试数据集和数据加载器
                        test_dataset = FullDataset(img_path, mask_path, args.img_size, mode='test')
                        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
                        
                        # 计算指标（evaluate_model 返回 dict，包含 mean + CI95）
                        results = evaluate_model(model, test_dataloader, device)
                        dice_mean = results.get('Dice', {}).get('mean', 0.0)
                        dice_ci95 = results.get('Dice', {}).get('CI95', (0.0, 0.0))
                        hd95_mean = results.get('HD95', {}).get('mean', 0.0)
                        hd95_ci95 = results.get('HD95', {}).get('CI95', (0.0, 0.0))
                        ece_mean = results.get('ECE', {}).get('mean', 0.0)
                        ece_ci95 = results.get('ECE', {}).get('CI95', (0.0, 0.0))

                        # 保存到数组（使用 mean 值）
                        epoch_dice_scores[dataset_name].append(dice_mean)
                        epoch_hd95_scores[dataset_name].append(hd95_mean)
                        epoch_ece_scores[dataset_name].append(ece_mean)

                        # 记录到tensorboard
                        writer.add_scalar(f'VAL_DICE/{dataset_name}', dice_mean, epoch+1)
                        writer.add_scalar(f'VAL_HD95/{dataset_name}', hd95_mean, epoch+1)
                        writer.add_scalar(f'VAL_ECE/{dataset_name}', ece_mean, epoch+1)

                        # 记录到日志文件（包含 CI95）
                        log_print(
                            f"  {dataset_name} - Dice Mean: {dice_mean:.4f}, Dice CI95: ({dice_ci95[0]:.4f},{dice_ci95[1]:.4f}), "
                            f"HD95 Mean: {hd95_mean:.4f}, HD95 CI95: ({hd95_ci95[0]:.4f},{hd95_ci95[1]:.4f}), "
                            f"ECE Mean: {ece_mean:.4f}, ECE CI95: ({ece_ci95[0]:.4f},{ece_ci95[1]:.4f})\n",
                            log_file,
                        )

        # 训练完成后，打印所有指标到日志文件
        log_print(f"\n========== Training Completed ==========\n", log_file)
        log_print(f"Training completed at {time.ctime()}\n\n", log_file)
        
        log_print("===== Dice Scores Summary =====\n", log_file)
        for dataset_name in args.test_dataset_names:
            log_print(f"Dataset: {dataset_name}\n", log_file)
            log_print(f"Dice scores per epoch: {epoch_dice_scores[dataset_name]}\n", log_file)
            if epoch_dice_scores[dataset_name]:
                best_idx = epoch_dice_scores[dataset_name].index(max(epoch_dice_scores[dataset_name]))
                best_epoch = (best_idx + 1) * args.eval_interval
                log_print(f"Best Dice: {max(epoch_dice_scores[dataset_name]):.4f} at epoch {best_epoch}\n", log_file)
            log_print(f"\n", log_file)
        
        log_print("===== HD95 Scores Summary =====\n", log_file)
        for dataset_name in args.test_dataset_names:
            log_print(f"Dataset: {dataset_name}\n", log_file)
            log_print(f"HD95 scores per epoch: {epoch_hd95_scores[dataset_name]}\n", log_file)
            if epoch_hd95_scores[dataset_name]:
                best_idx = epoch_hd95_scores[dataset_name].index(min(epoch_hd95_scores[dataset_name]))
                best_epoch = (best_idx + 1) * args.eval_interval
                log_print(f"Best HD95: {min(epoch_hd95_scores[dataset_name]):.4f} at epoch {best_epoch}\n", log_file)
            log_print(f"\n", log_file)

        log_print("===== ECE Scores Summary =====\n", log_file)
        for dataset_name in args.test_dataset_names:
            log_print(f"Dataset: {dataset_name}\n", log_file)
            log_print(f"ECE scores per epoch: {epoch_ece_scores[dataset_name]}\n", log_file)
            if epoch_ece_scores[dataset_name]:
                # 对于 ECE，越小越好
                best_idx = epoch_ece_scores[dataset_name].index(min(epoch_ece_scores[dataset_name]))
                best_epoch = (best_idx + 1) * args.eval_interval
                log_print(f"Best ECE: {min(epoch_ece_scores[dataset_name]):.4f} at epoch {best_epoch}\n", log_file)
            log_print(f"\n", log_file)
        
    except Exception as e:
        import traceback
        error_msg = f"Error occurred: {str(e)}\n"
        error_msg += traceback.format_exc() + "\n"
        log_print(error_msg, log_file)
        raise
    finally:
        log_file.close()
        if 'writer' in locals():
            writer.close()

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    seed_torch(1024)
    main(args)