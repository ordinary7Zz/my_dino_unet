# 本脚本对单张图像在 DINOv3_S_UNet 分割网络中使用 Grad-CAM 生成热力图，
# 输出风格与 heatmap.jpg 一致：大面积扩散热力 + 原图清晰透出 + jet colormap + 固定alpha叠加。

import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
from torchvision import transforms

from dino_unet import DINOv3_S_UNet


def load_image(
    image_path: str,
    img_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """加载单张图像并做预处理。"""
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    orig_np = np.array(resized_img)  # (H, W, 3) uint8

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    img_t = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)
    return img_t, orig_np


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

    model.load_state_dict(state_dict, strict=False)
    return model


class GradCAM:
    """
    Grad-CAM for segmentation model.
    
    通过 hook 获取指定层的特征图和梯度，
    计算加权特征图作为 CAM 热力图。
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册前向和反向 hook
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        生成 Grad-CAM 热力图。

        Args:
            input_tensor: (1, 3, H, W) 输入张量
            target_mask: 可选的目标区域掩膜 (H, W)，用于指定计算梯度的目标区域

        Returns:
            cam: (H, W) 归一化到 [0, 1] 的热力图
        """
        self.model.zero_grad()

        # 前向传播
        output = self.model(input_tensor)  # (1, 1, H, W)

        if target_mask is not None:
            mask_tensor = torch.from_numpy(target_mask).float().to(input_tensor.device)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            mask_tensor = F.interpolate(mask_tensor, size=output.shape[2:], mode="nearest")
            target_score = (output * mask_tensor).sum() / (mask_tensor.sum() + 1e-6)
        else:
            flat_logits = output.flatten()
            k = max(1, flat_logits.numel() // 20)
            target_score = flat_logits.topk(k).values.mean()

        # 反向传播获取梯度
        target_score.backward()

        # 获取梯度和激活
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # Global Average Pooling on gradients -> channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # 加权求和
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU: 只保留正向贡献
        cam = F.relu(cam)

        # 上采样到输入图像尺寸
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        # 转为 numpy
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        return cam

    def release(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


def postprocess_cam(
    cam: np.ndarray,
    smooth_sigma_ratio: float = 0.02,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    对原始 CAM 进行后处理，使热力分布接近 heatmap.jpg 风格。
    
    处理步骤：
    1. percentile 裁切归一化（避免极端值拉伸）
    2. 大核高斯平滑（让热力扩散、过渡宽）
    3. 重新归一化
    4. gamma < 1 变换（抬升中低值，扩大绿/黄过渡区域面积）
    
    Args:
        cam: (H, W) 原始 Grad-CAM 热力图
        smooth_sigma_ratio: 高斯平滑核大小占图像尺寸的比例，越大越扩散
        gamma: gamma 变换指数，< 1 抬升中低值，让热力分布更"满"
        
    Returns:
        cam_final: (H, W) [0, 1] 后处理后的热力图
    """
    H, W = cam.shape

    # 1) percentile 裁切归一化：截断极端值，保留主体分布
    p_low, p_high = np.percentile(cam, [1, 99])
    cam = np.clip(cam, p_low, p_high)
    c_min, c_max = cam.min(), cam.max()
    if c_max - c_min > 1e-8:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)

    # 2) 轻量高斯平滑：仅用于展示，不强行塑形
    sigma = max(H, W) * smooth_sigma_ratio
    if sigma > 1e-6:
        cam = gaussian_filter(cam, sigma=sigma)

    # 3) 重新归一化
    c_min, c_max = cam.min(), cam.max()
    if c_max - c_min > 1e-8:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)

    # 4) gamma 变换：默认不压缩/抬升中低值，尽量保留原始响应形状
    cam = np.power(cam, gamma)

    return cam


def generate_heatmap_overlay(
    cam: np.ndarray,
    orig_np: np.ndarray,
    alpha: float = 0.65,
    saturation_scale: float = 1.3,
) -> np.ndarray:
    """
    生成与 heatmap.jpg 风格一致的叠加图像。
    
    关键风格特征：
    - jet colormap
    - 固定 alpha 叠加（整幅图均匀覆盖）
    - 原图纹理清晰透出
    - 色彩饱和鲜艳（HSV 饱和度增强）
    
    Args:
        cam: (H, W) 归一化的热力图 [0, 1]
        orig_np: (H, W, 3) uint8 原图
        alpha: 热力图叠加的不透明度（0.65 更接近 heatmap.jpg）
        saturation_scale: 饱和度增强倍数（>1 更鲜艳）
        
    Returns:
        blended: (H, W, 3) uint8 叠加图像
    """
    H, W = orig_np.shape[:2]

    # 确保 cam 尺寸匹配
    if cam.shape != (H, W):
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    # 将 cam 转为 uint8
    cam_uint8 = np.uint8(255 * cam)

    # 使用 OpenCV 的 JET colormap（与 heatmap.jpg 一致）
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 固定 alpha 叠加：blended = alpha * heatmap + (1 - alpha) * orig
    orig_float = orig_np.astype(np.float32)
    heatmap_float = heatmap_rgb.astype(np.float32)

    blended = alpha * heatmap_float + (1.0 - alpha) * orig_float
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # 色彩饱和度增强：在 HSV 空间提升 S 通道，让颜色更鲜艳
    if saturation_scale != 1.0:
        blended_hsv = cv2.cvtColor(blended, cv2.COLOR_RGB2HSV).astype(np.float32)
        blended_hsv[:, :, 1] = np.clip(blended_hsv[:, :, 1] * saturation_scale, 0, 255)
        blended = cv2.cvtColor(blended_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return blended


def run_gradcam(
    checkpoint: str,
    image_path: str,
    output_dir: str,
    mask_path: Optional[str] = None,
    img_size: int = 224,
    dino_pretrained: bool = True,
    alpha: float = 0.45,
    target_layer_name: str = "reduce4",
    smooth_sigma_ratio: float = 0.02,
    gamma: float = 1.0,
    saturation_scale: float = 1.3,
) -> None:
    """
    对单张图像生成 Grad-CAM 热力图，输出风格与 heatmap.jpg 一致。
    
    输出文件:
    - original_{name}.png: 原图
    - gradcam_map_{name}.png: 纯 Grad-CAM 热力图（经后处理）
    - gradcam_overlay_{name}.png: 原图 + Grad-CAM 叠加图（heatmap.jpg 风格）
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 加载图像
    img_tensor, orig_np = load_image(image_path, img_size, device)
    print(f"Loaded image: {image_path} -> ({img_size}, {img_size})")

    # 2) 加载可选 mask
    region_mask: Optional[np.ndarray] = None
    if mask_path is not None and os.path.isfile(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((img_size, img_size), resample=Image.NEAREST)
        mask_np = np.array(mask)
        region_mask = (mask_np > mask_np.mean()).astype(np.float32)
        print(f"Loaded mask: {mask_path}")

    # 3) 构建模型
    model = build_model(checkpoint, device, dino_pretrained=dino_pretrained)
    model.eval()

    # 4) 选择目标层
    # DINOv3_S_UNet 的解码器层: up1, up2, up3, up4
    # reduce4 更偏语义关注，避免默认落在过于贴近输出的层
    target_layer = getattr(model, target_layer_name, None)
    resolved_target_layer_name = target_layer_name
    if target_layer is None:
        resolved_target_layer_name = "reduce4"
        print(f"Warning: target_layer '{target_layer_name}' not found, using 'reduce4'")
        target_layer = model.reduce4

    print(f"Using target layer: {resolved_target_layer_name}")

    # 5) 计算 Grad-CAM
    img_tensor.requires_grad_(True)

    gradcam = GradCAM(model, target_layer)
    cam_raw = gradcam.generate(img_tensor, target_mask=region_mask)
    gradcam.release()

    print(f"Grad-CAM raw computed, shape: {cam_raw.shape}, "
          f"range: [{cam_raw.min():.4f}, {cam_raw.max():.4f}]")

    # 6) 后处理：高斯平滑 + gamma 变换，使热力风格匹配 heatmap.jpg
    cam = postprocess_cam(
        cam_raw,
        smooth_sigma_ratio=smooth_sigma_ratio,
        gamma=gamma,
    )
    print(f"Postprocessed CAM, range: [{cam.min():.4f}, {cam.max():.4f}], "
          f"sigma_ratio={smooth_sigma_ratio}, gamma={gamma}")

    # 7) 生成输出图像
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # --- 保存原图 ---
    original_path = os.path.join(output_dir, f"original_{image_name}.png")
    Image.fromarray(orig_np).save(original_path)

    # --- 保存纯 Grad-CAM 热力图（经后处理） ---
    cam_uint8 = np.uint8(255 * cam)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    gradcam_map_path = os.path.join(output_dir, f"gradcam_map_{image_name}.png")
    Image.fromarray(heatmap_rgb).save(gradcam_map_path)

    # --- 生成叠加图（heatmap.jpg 风格）---
    blended = generate_heatmap_overlay(
        cam, orig_np, alpha=alpha, saturation_scale=saturation_scale,
    )
    overlay_path = os.path.join(output_dir, f"gradcam_overlay_{image_name}.png")
    Image.fromarray(blended).save(overlay_path)

    # --- 如果有 mask，额外保存带 GT 轮廓的叠加图 ---
    if region_mask is not None:
        contours_img = blended.copy()
        mask_uint8 = (region_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 2)
        gt_overlay_path = os.path.join(output_dir, f"gradcam_overlay_gt_{image_name}.png")
        Image.fromarray(contours_img).save(gt_overlay_path)
        print(f"Saved overlay with GT contour: {gt_overlay_path}")

    print(f"Saved original: {original_path}")
    print(f"Saved Grad-CAM map: {gradcam_map_path}")
    print(f"Saved Grad-CAM overlay (heatmap.jpg style): {overlay_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对单张图像的 DINOv3_S_UNet 分割结果生成 Grad-CAM 热力图（heatmap.jpg 风格）。"
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
        help="可选：对应的 mask 路径，用作 Grad-CAM 目标区域。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gradcam_single_image_out",
        help="保存 Grad-CAM 可视化结果的目录。",
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
        "--alpha",
        type=float,
        default=0.45,
        help="热力图叠加不透明度（0~1）。0.45 原图纹理清晰透出。",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default="reduce4",
        choices=["up1", "up2", "up3", "up4", "reduce1", "reduce2", "reduce3", "reduce4"],
        help="Grad-CAM 目标层名称。reduce4（默认）更偏语义关注，避免过于贴近输出。",
    )
    parser.add_argument(
        "--smooth_sigma_ratio",
        type=float,
        default=0.02,
        help="高斯平滑核占图像尺寸的比例；值越小越接近原始 CAM。",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma 变换指数；1.0 表示尽量保持原始响应形状。",
    )
    parser.add_argument(
        "--saturation_scale",
        type=float,
        default=1.3,
        help="色彩饱和度增强倍数（>1 更鲜艳）。",
    )
    args = parser.parse_args()
    args.dino_pretrained = str(args.dino_pretrained).lower() in ("true", "1", "yes", "y")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_gradcam(
        checkpoint=args.checkpoint,
        image_path=args.image_path,
        output_dir=args.output_dir,
        mask_path=args.mask_path,
        img_size=args.img_size,
        dino_pretrained=args.dino_pretrained,
        alpha=args.alpha,
        target_layer_name=args.target_layer,
        smooth_sigma_ratio=args.smooth_sigma_ratio,
        gamma=args.gamma,
        saturation_scale=args.saturation_scale,
    )
