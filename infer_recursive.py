import argparse
import os
import time
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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


def find_images_recursive(input_dir):
    root = Path(input_dir)
    image_paths = [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


class RecursiveInferenceDataset(Dataset):
    def __init__(self, input_dir, img_size):
        self.input_root = Path(input_dir)
        self.image_paths = find_images_recursive(self.input_root)
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            image_tensor = self.transform(img)

        rel_path = image_path.relative_to(self.input_root).as_posix()
        return {
            "image": image_tensor,
            "rel_path": rel_path,
            "orig_w": orig_w,
            "orig_h": orig_h,
        }


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Compatible with checkpoints saved from DataParallel.
    if len(state_dict) > 0 and next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return missing_keys, unexpected_keys


def predict_to_uint8(pred_logits, output_type="binary", threshold=0.5):
    prob = torch.sigmoid(pred_logits).detach().cpu().numpy().squeeze()
    if output_type == "binary":
        return (prob >= threshold).astype(np.uint8) * 255

    prob_norm = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
    return (prob_norm * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser("DINOv3-UNet Recursive Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory (must be a directory)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for predicted masks")
    parser.add_argument("--img_size", type=int, default=224, help="Model input image size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default=None, help='Device, e.g. "cuda", "cuda:0", "cpu"')
    parser.add_argument(
        "--dino_pretrained",
        type=str,
        default="false",
        help="Whether to initialize DINO backbone with pretrained weights (true/false)",
    )
    parser.add_argument(
        "--use_dilation",
        type=str,
        default="false",
        help="Whether to enable dilation block in model (true/false)",
    )
    parser.add_argument(
        "--save_orig_size",
        type=str,
        default="true",
        help="Resize predicted masks back to each image original size before saving (true/false)",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="binary",
        choices=["binary", "prob"],
        help="binary: thresholded mask; prob: normalized probability map",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used when output_type=binary")
    args = parser.parse_args()

    args.input_dir = clean_path(args.input_dir)
    args.output_dir = clean_path(args.output_dir)
    args.checkpoint = clean_path(args.checkpoint)
    args.dino_pretrained = str2bool(args.dino_pretrained)
    args.use_dilation = str2bool(args.use_dilation)
    args.save_orig_size = str2bool(args.save_orig_size)

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"--input_dir must be an existing directory: {args.input_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")

    dataset = RecursiveInferenceDataset(args.input_dir, args.img_size)
    if len(dataset) == 0:
        print("No image files found under input_dir.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DINOv3_S_UNet(pretrained=args.dino_pretrained, use_dilation=args.use_dilation).to(device)
    model.eval()

    missing_keys, unexpected_keys = load_checkpoint(model, args.checkpoint, device)
    if missing_keys:
        print(f"Warning: Missing keys when loading checkpoint: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading checkpoint: {len(unexpected_keys)}")

    start_time = time.time()
    total = len(dataset)
    done = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            rel_paths = batch["rel_path"]
            orig_ws = batch["orig_w"]
            orig_hs = batch["orig_h"]

            pred = model(images)
            if isinstance(pred, list):
                pred = pred[0]

            for i in range(pred.shape[0]):
                mask_uint8 = predict_to_uint8(pred[i : i + 1], args.output_type, args.threshold)

                if args.save_orig_size:
                    ow = int(orig_ws[i].item())
                    oh = int(orig_hs[i].item())
                    if (mask_uint8.shape[1], mask_uint8.shape[0]) != (ow, oh):
                        mask_uint8 = np.array(
                            Image.fromarray(mask_uint8).resize((ow, oh), resample=Image.NEAREST)
                        ).astype(np.uint8)

                rel_file = Path(rel_paths[i]).with_suffix(".png")
                out_file = Path(args.output_dir) / rel_file
                out_file.parent.mkdir(parents=True, exist_ok=True)
                imageio.imsave(out_file.as_posix(), mask_uint8)

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"Progress: {done}/{total}")

    elapsed = time.time() - start_time
    print(f"Inference finished. Saved {done} masks in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
