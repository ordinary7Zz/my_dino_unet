import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dino_unet import DINOv3_S_UNet
from LNM_screening.io_utils import save_binary_mask, save_prob_npy as save_prob_npy_file, save_prob_png


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


def logits_to_prob_array(pred_logits):
    return torch.sigmoid(pred_logits).detach().cpu().numpy().squeeze().astype(np.float32)


def prob_to_binary_uint8(prob, threshold=0.5):
    return ((np.asarray(prob, dtype=np.float32) >= threshold).astype(np.uint8) * 255).astype(np.uint8)


def prob_to_legacy_prob_uint8(prob):
    prob = np.asarray(prob, dtype=np.float32)
    prob_norm = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
    return (prob_norm * 255.0).round().astype(np.uint8)


def prob_to_visual_uint8(prob):
    return (np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def resize_binary_mask(mask_uint8, size):
    return np.array(Image.fromarray(mask_uint8).resize(size, resample=Image.NEAREST)).astype(np.uint8)


def resize_prob_map(prob, size):
    prob_img = Image.fromarray(np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0), mode="F")
    return np.asarray(prob_img.resize(size, resample=Image.BILINEAR), dtype=np.float32)


def resolve_output_flags(args):
    explicit_new_outputs = any(
        [
            str2bool(args.save_binary),
            str2bool(args.save_prob_png),
            str2bool(args.save_prob_npy),
        ]
    )

    if not explicit_new_outputs:
        return {
            "legacy_mode": True,
            "save_binary": args.output_type == "binary",
            "save_prob_png": args.output_type == "prob",
            "save_prob_npy": False,
        }

    return {
        "legacy_mode": False,
        "save_binary": str2bool(args.save_binary),
        "save_prob_png": str2bool(args.save_prob_png),
        "save_prob_npy": str2bool(args.save_prob_npy),
    }


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
        help="Legacy single-output mode: binary saves thresholded masks, prob saves probability PNGs.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used for binary mask generation")
    parser.add_argument("--save_binary", type=str, default="false", help="Whether to save thresholded binary masks")
    parser.add_argument("--save_prob_png", type=str, default="false", help="Whether to save probability PNGs scaled from sigmoid outputs")
    parser.add_argument("--save_prob_npy", type=str, default="false", help="Whether to save raw sigmoid probability maps as .npy")
    parser.add_argument("--binary_subdir", type=str, default="binary", help="Subdirectory for binary mask outputs")
    parser.add_argument("--prob_png_subdir", type=str, default="prob_png", help="Subdirectory for probability PNG outputs")
    parser.add_argument("--prob_npy_subdir", type=str, default="prob", help="Subdirectory for probability NPY outputs")
    args = parser.parse_args()

    args.input_dir = clean_path(args.input_dir)
    args.output_dir = clean_path(args.output_dir)
    args.checkpoint = clean_path(args.checkpoint)
    args.dino_pretrained = str2bool(args.dino_pretrained)
    args.use_dilation = str2bool(args.use_dilation)
    args.save_orig_size = str2bool(args.save_orig_size)
    output_flags = resolve_output_flags(args)
    legacy_mode = output_flags["legacy_mode"]
    save_binary = output_flags["save_binary"]
    save_prob_png = output_flags["save_prob_png"]
    save_prob_npy = output_flags["save_prob_npy"]

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
    print(f"Legacy mode: {legacy_mode}")
    print(f"Save binary: {save_binary}")
    print(f"Save prob png: {save_prob_png}")
    print(f"Save prob npy: {save_prob_npy}")

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
                prob = logits_to_prob_array(pred[i : i + 1])
                legacy_uint8 = None
                binary_uint8 = prob_to_binary_uint8(prob, args.threshold) if save_binary else None
                prob_png_uint8 = prob_to_visual_uint8(prob) if save_prob_png else None

                if legacy_mode:
                    legacy_uint8 = (
                        prob_to_binary_uint8(prob, args.threshold)
                        if args.output_type == "binary"
                        else prob_to_legacy_prob_uint8(prob)
                    )

                if args.save_orig_size:
                    ow = int(orig_ws[i].item())
                    oh = int(orig_hs[i].item())
                    target_size = (ow, oh)
                    if legacy_uint8 is not None and (legacy_uint8.shape[1], legacy_uint8.shape[0]) != target_size:
                        legacy_uint8 = resize_binary_mask(legacy_uint8, target_size)
                    if binary_uint8 is not None and (binary_uint8.shape[1], binary_uint8.shape[0]) != target_size:
                        binary_uint8 = resize_binary_mask(binary_uint8, target_size)
                    if prob_png_uint8 is not None and (prob_png_uint8.shape[1], prob_png_uint8.shape[0]) != target_size:
                        prob_png_uint8 = prob_to_visual_uint8(resize_prob_map(prob, target_size))
                    if save_prob_npy and (prob.shape[1], prob.shape[0]) != target_size:
                        prob = resize_prob_map(prob, target_size)

                rel_png = Path(rel_paths[i]).with_suffix(".png")
                rel_npy = Path(rel_paths[i]).with_suffix(".npy")
                output_root = Path(args.output_dir)

                if legacy_mode and legacy_uint8 is not None:
                    out_file = output_root / rel_png
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(legacy_uint8).save(out_file.as_posix())
                else:
                    if binary_uint8 is not None:
                        save_binary_mask(output_root / args.binary_subdir / rel_png, binary_uint8)
                    if prob_png_uint8 is not None:
                        save_prob_png(output_root / args.prob_png_subdir / rel_png, prob_png_uint8.astype(np.float32) / 255.0)
                    if save_prob_npy:
                        save_prob_npy_file(output_root / args.prob_npy_subdir / rel_npy, prob)

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"Progress: {done}/{total}")

    elapsed = time.time() - start_time
    print(f"Inference finished. Saved {done} masks in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
