# 本脚本用于对单张图像在 DINOv3_S_UNet 分割网络中的预测结果进行 SHAP 分析，
# 生成像素级归因热力图和原图+SHAP 叠加图，用于解释哪些图像区域对分割结果贡献最大。

import argparse
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt  # type: ignore
import shap  # type: ignore

from dino_unet import DINOv3_S_UNet


def load_image(
    image_path: str,
    img_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """加载单张图像并做与训练类似的预处理（缩放到 img_size，转为张量，简单归一化）。"""
    img = Image.open(image_path).convert("RGB")
    orig_np = np.array(img)

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # 这里使用简单的 [0,1] 归一化；如果训练时有特定均值/方差，可在此处改成对应的 Normalize
        ]
    )
    img_t = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)
    return img_t, orig_np


def load_mask_as_region(
    mask_path: str,
    img_size: int,
) -> np.ndarray:
    """加载单张分割 mask（可选），返回大小为 img_size×img_size 的二值区域掩膜。"""
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((img_size, img_size), resample=Image.NEAREST)
    mask_np = np.array(mask)
    # 简单二值化
    thresh = mask_np.mean()
    region = (mask_np > thresh).astype(np.float32)
    return region


class SegTargetWrapper(torch.nn.Module):
    """
    将分割网络包装成“输入 → 标量”的形式，便于 SHAP 计算。

    - base_model(x) 输出：(B, 1, H, W) 的 logit
    - 转为概率后：
        * 若提供 region_mask，则计算该区域内的平均 foreground 概率
        * 否则计算整幅图的平均 foreground 概率
    """

    def __init__(self, base_model: torch.nn.Module, region_mask: Optional[np.ndarray] = None):
        super().__init__()
        self.base_model = base_model
        if region_mask is not None:
            # region_mask: (H, W) numpy → buffer，便于迁移到 device
            region = torch.from_numpy(region_mask.astype(np.float32))  # (H, W)
        else:
            region = None
        self.register_buffer("region_mask", region if region is not None else torch.tensor([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        logits = self.base_model(x)  # (B, 1, H, W)
        probs = torch.sigmoid(logits)[:, 0]  # (B, H, W)

        if self.region_mask.numel() > 0:
            # 使用指定区域的平均概率作为目标
            m = self.region_mask.to(probs.device)  # (H, W)
            m = m.clamp(min=0.0, max=1.0)
            # 自动 broadcast 到 batch 维
            while m.dim() < probs.dim():
                m = m.unsqueeze(0)
            m = m.expand_as(probs)  # (B, H, W)
            num = (probs * m).sum(dim=(1, 2))
            den = m.sum(dim=(1, 2)).clamp_min(1e-6)
            out = num / den  # (B,)
        else:
            # 整幅图的平均 foreground 概率
            out = probs.mean(dim=(1, 2))  # (B,)

        # GradientExplainer 期望输出形状为 (B, num_outputs)，
        # 这里只有一个标量输出，因此在最后增加一个维度 → (B, 1)
        return out.unsqueeze(1)


def build_model(
    checkpoint: str,
    device: torch.device,
    dino_pretrained: bool = True,
) -> DINOv3_S_UNet:
    """构建 DINOv3_S_UNet 并加载 checkpoint 权重。"""
    model = DINOv3_S_UNet(pretrained=dino_pretrained).to(device)
    model.eval()

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # 与 test_parallel 一样，使用 strict=False 以兼容略有不同的 key
    model.load_state_dict(state_dict, strict=False)
    return model


def run_shap_for_single_image(
    checkpoint: str,
    image_path: str,
    output_dir: str,
    mask_path: Optional[str] = None,
    img_size: int = 224,
    dino_pretrained: bool = True,
    background_mode: str = "zeros",
) -> None:
    """
    对单张图像进行 SHAP 分析，生成：
    - shap_map.png：像素级 SHAP 归因热力图
    - overlay.png：原图 + SHAP 叠加图
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 加载图像和（可选）mask
    img_tensor, orig_np = load_image(image_path, img_size, device)  # (1, 3, H, W)

    region_mask_np: Optional[np.ndarray] = None
    if mask_path is not None and os.path.isfile(mask_path):
        region_mask_np = load_mask_as_region(mask_path, img_size)
        print(f"Loaded region mask from {mask_path}")
    else:
        print("No valid mask_path provided, use whole image as target region.")

    # 2) 构建模型与包装器
    base_model = build_model(checkpoint, device, dino_pretrained=dino_pretrained)
    wrapped_model = SegTargetWrapper(base_model, region_mask=region_mask_np).to(device)
    wrapped_model.eval()

    # 3) 构造背景样本（用于 GradientExplainer）
    if background_mode == "zeros":
        background = torch.zeros_like(img_tensor)
    elif background_mode == "blur":
        blur = F.avg_pool2d(img_tensor, kernel_size=15, stride=1, padding=7)
        background = blur
    else:
        # 默认使用全黑作为 baseline
        background = torch.zeros_like(img_tensor)

    # 4) 运行 SHAP GradientExplainer
    print("Running SHAP GradientExplainer ...")
    explainer = shap.GradientExplainer(
        model=wrapped_model,
        data=background,
    )

    # shap_values: 对于 GradientExplainer，通常是 numpy 数组（或数组列表）
    shap_values = explainer.shap_values(img_tensor)
    if isinstance(shap_values, list):
        shap_arr = shap_values[0]
    else:
        shap_arr = shap_values

    # 将 SHAP 结果统一转为 numpy，兼容 torch.Tensor / np.ndarray 两种情况
    if isinstance(shap_arr, torch.Tensor):
        shap_arr_np = shap_arr.detach().cpu().numpy()
    else:
        shap_arr_np = np.asarray(shap_arr)

    # 对通道维求和，得到 (H, W) 的像素级归因图
    shap_map_np = shap_arr_np.sum(axis=1)[0]  # (H, W)

    # 归一化到 [0,1] 便于可视化
    shap_min, shap_max = np.percentile(shap_map_np, [2, 98])
    shap_norm = np.clip((shap_map_np - shap_min) / (shap_max - shap_min + 1e-8), 0, 1)

    # 5) 保存 SHAP 热力图和叠加图
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    if orig_np.ndim == 2:
        plt.imshow(orig_np, cmap="gray")
    else:
        plt.imshow(orig_np)
    plt.axis("off")
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    plt.imshow(shap_norm, cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.title("SHAP attribution")

    plt.subplot(1, 3, 3)
    if orig_np.ndim == 2:
        plt.imshow(orig_np, cmap="gray")
    else:
        plt.imshow(orig_np)
    plt.imshow(shap_norm, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.title("Overlay")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "shap_single_image.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    # 同时单独保存归因热力图
    plt.figure(figsize=(5, 4))
    plt.imshow(shap_norm, cmap="jet")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "shap_map.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    print(f"Saved SHAP visualization to: {save_path}")
    print(f"Saved SHAP heatmap to: {heatmap_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对单张图像的 DINOv3_S_UNet 分割结果进行 SHAP 分析（像素级热力图）。"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="dino_unet 的权重文件路径（.pth 或 .pt）。",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="待分析的单张图像路径。",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="可选：对应的 mask 路径，用作 SHAP 目标区域（不提供则使用整幅图）。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./shap_single_image_out",
        help="保存 SHAP 可视化结果的目录。",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="输入图像缩放尺寸，需与训练/测试时保持一致。",
    )
    parser.add_argument(
        "--dino_pretrained",
        type=str,
        default="True",
        help="是否为 DINOv3 backbone 加载 ImageNet 预训练权重（True/False）。",
    )
    parser.add_argument(
        "--background_mode",
        type=str,
        default="zeros",
        choices=["zeros", "blur"],
        help="SHAP baseline 的背景构造方式：zeros=全黑，blur=模糊图像。",
    )
    args = parser.parse_args()
    args.dino_pretrained = str(args.dino_pretrained).lower() in ("true", "1", "yes", "y")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_shap_for_single_image(
        checkpoint=args.checkpoint,
        image_path=args.image_path,
        output_dir=args.output_dir,
        mask_path=args.mask_path,
        img_size=args.img_size,
        dino_pretrained=args.dino_pretrained,
        background_mode=args.background_mode,
    )

