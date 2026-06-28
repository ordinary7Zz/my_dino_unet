import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dino_unet import DINOv3_S_UNet

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean_path(path):
    if isinstance(path, str):
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        path = path.strip()
    return path


class InferenceDataset(Dataset):
    """仅加载图像（无 GT），记录文件名和原始尺寸用于推理后保存。"""

    def __init__(self, input_dir, img_size):
        self.input_dir = Path(input_dir)
        self.image_paths = sorted(
            p for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            img_tensor = self.transform(img)
        return {
            "image": img_tensor,
            "filename": img_path.name,
            "orig_w": orig_w,
            "orig_h": orig_h,
        }


def main():
    parser = argparse.ArgumentParser("DINOv3-UNet Segmentation Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for predicted binary masks")
    parser.add_argument("--img_size", type=int, default=224, help="Model input image size (default: 224)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference (default: 4)")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers (default: 4)")
    parser.add_argument("--device", type=str, default=None, help='Device, e.g. "cuda", "cuda:0", "cpu"')
    parser.add_argument("--dino_pretrained", type=str, default="false",
                        help="Load pretrained DINO backbone weights (true/false, default: false)")
    parser.add_argument("--use_dilation", type=str, default="false",
                        help="Use dilation block in model (true/false, default: false)")
    args = parser.parse_args()

    # 路径清理
    args.input_dir = clean_path(args.input_dir)
    args.output_dir = clean_path(args.output_dir)
    args.checkpoint = clean_path(args.checkpoint)
    args.dino_pretrained = str2bool(args.dino_pretrained)
    args.use_dilation = str2bool(args.use_dilation)

    # 校验输入
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"--input_dir must be a directory: {args.input_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Image size: {args.img_size}")

    # 数据集
    dataset = InferenceDataset(args.input_dir, args.img_size)
    if len(dataset) == 0:
        print("No image files found in input_dir.")
        return
    print(f"Found {len(dataset)} images.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 模型
    model = DINOv3_S_UNet(pretrained=args.dino_pretrained, use_dilation=args.use_dilation).to(device)
    model.eval()

    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    print("Checkpoint loaded successfully.")

    # 推理
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit="batch"):
            images = batch["image"].to(device)
            filenames = batch["filename"]
            orig_ws = batch["orig_w"]
            orig_hs = batch["orig_h"]

            pred = model(images)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            # sigmoid + 二值化
            probs = torch.sigmoid(pred)  # (B, 1, H, W)
            masks = (probs > 0.5).float()  # (B, 1, H, W)

            for i in range(masks.shape[0]):
                mask_np = masks[i, 0].cpu().numpy()  # (H, W), float 0/1
                mask_uint8 = (mask_np * 255).astype(np.uint8)

                # Resize 回原始尺寸
                ow = int(orig_ws[i].item())
                oh = int(orig_hs[i].item())
                if mask_uint8.shape[::-1] != (ow, oh):  # (H, W) vs (W, H)
                    mask_uint8 = np.array(
                        Image.fromarray(mask_uint8).resize((ow, oh), resample=Image.NEAREST)
                    ).astype(np.uint8)

                # 保存：命名与原图相同，后缀改为 .png
                name = Path(filenames[i]).stem + ".png"
                out_path = os.path.join(args.output_dir, name)
                Image.fromarray(mask_uint8).save(out_path)

    print(f"Inference complete. Masks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
