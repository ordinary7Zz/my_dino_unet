import importlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


def load_gradcam_module():
    fake_dino_unet = types.ModuleType("dino_unet")

    class FakeModel:
        pass

    fake_dino_unet.DINOv3_S_UNet = FakeModel

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.COLOR_BGR2RGB = 0
    fake_cv2.COLOR_RGB2HSV = 1
    fake_cv2.COLOR_HSV2RGB = 2
    fake_cv2.COLORMAP_JET = 3
    fake_cv2.INTER_LINEAR = 4
    fake_cv2.RETR_EXTERNAL = 5
    fake_cv2.CHAIN_APPROX_SIMPLE = 6
    fake_cv2.resize = lambda img, size, interpolation=None: img
    fake_cv2.applyColorMap = lambda img, cmap: img
    fake_cv2.cvtColor = lambda img, code: img
    fake_cv2.findContours = lambda *args, **kwargs: ([], None)
    fake_cv2.drawContours = lambda img, *args, **kwargs: img

    fake_scipy = types.ModuleType("scipy")
    fake_ndimage = types.ModuleType("scipy.ndimage")
    fake_ndimage.gaussian_filter = lambda arr, sigma=0: arr
    fake_scipy.ndimage = fake_ndimage

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch.device = str
    fake_torch.load = lambda *args, **kwargs: {}
    fake_torch.from_numpy = lambda array: array
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_torch_nn = types.ModuleType("torch.nn")
    fake_torch_nn.Module = object
    fake_torch.nn = fake_torch_nn

    fake_torch_functional = types.ModuleType("torch.nn.functional")
    fake_torch_functional.interpolate = lambda *args, **kwargs: args[0]
    fake_torch_functional.relu = lambda x: x

    fake_torchvision = types.ModuleType("torchvision")
    fake_transforms = types.ModuleType("torchvision.transforms")
    fake_transforms.Compose = lambda steps: (lambda image: image)
    fake_transforms.Resize = lambda size: (lambda image: image)
    fake_transforms.ToTensor = lambda: (lambda image: image)
    fake_torchvision.transforms = fake_transforms

    sys.modules.pop("gradcam_single_image_seg", None)
    with mock.patch.dict(
        sys.modules,
        {
            "dino_unet": fake_dino_unet,
            "cv2": fake_cv2,
            "scipy": fake_scipy,
            "scipy.ndimage": fake_ndimage,
            "torch": fake_torch,
            "torch.nn": fake_torch_nn,
            "torch.nn.functional": fake_torch_functional,
            "torchvision": fake_torchvision,
            "torchvision.transforms": fake_transforms,
        },
    ):
        return importlib.import_module("gradcam_single_image_seg")


class TestGradcamSingleImageSegHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_gradcam_module()

    def test_collect_files_by_stem_uses_supported_extensions(self):
        with TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir) / "images"
            image_dir.mkdir()
            (image_dir / "case_a.jpg").write_bytes(b"jpg")
            (image_dir / "case_b.png").write_bytes(b"png")
            (image_dir / "ignore.txt").write_text("ignore", encoding="utf-8")

            files = self.module.collect_files_by_stem(image_dir, self.module.IMAGE_EXTENSIONS)

            self.assertEqual(set(files.keys()), {"case_a", "case_b"})
            self.assertEqual(files["case_a"], image_dir / "case_a.jpg")
            self.assertEqual(files["case_b"], image_dir / "case_b.png")

    def test_collect_files_by_stem_rejects_duplicate_stems(self):
        with TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir) / "images"
            image_dir.mkdir()
            (image_dir / "case_a.jpg").write_bytes(b"jpg")
            (image_dir / "case_a.png").write_bytes(b"png")

            with self.assertRaisesRegex(ValueError, "重复"):
                self.module.collect_files_by_stem(image_dir, self.module.IMAGE_EXTENSIONS)

    def test_resolve_input_pairs_batch_mode_matches_by_stem(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            mask_dir = root / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            (image_dir / "case_a.jpg").write_bytes(b"jpg")
            (image_dir / "case_b.png").write_bytes(b"png")
            (mask_dir / "case_a.png").write_bytes(b"png")
            (mask_dir / "case_b.jpg").write_bytes(b"jpg")

            pairs = self.module.resolve_input_pairs(
                image_path=None,
                mask_path=None,
                image_dir=str(image_dir),
                mask_dir=str(mask_dir),
                output_types=("overlay",),
            )

            self.assertEqual(
                pairs,
                [
                    (str(image_dir / "case_a.jpg"), str(mask_dir / "case_a.png")),
                    (str(image_dir / "case_b.png"), str(mask_dir / "case_b.jpg")),
                ],
            )

    def test_resolve_input_pairs_requires_mask_for_overlay_gt(self):
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "case_a.jpg"
            image_path.write_bytes(b"jpg")

            with self.assertRaisesRegex(ValueError, "mask"):
                self.module.resolve_input_pairs(
                    image_path=str(image_path),
                    mask_path=None,
                    image_dir=None,
                    mask_dir=None,
                    output_types=("overlay_gt",),
                )

    def test_build_output_paths_preserve_original_filename(self):
        with TemporaryDirectory() as tmp_dir:
            output_paths = self.module.build_output_paths(
                output_dir=tmp_dir,
                image_filename="case_a.jpg",
                output_types=self.module.resolve_output_types("all"),
            )

            self.assertEqual(Path(output_paths["original"]), Path(tmp_dir) / "original" / "case_a.jpg")
            self.assertEqual(Path(output_paths["overlay"]), Path(tmp_dir) / "overlay" / "case_a.jpg")
            self.assertEqual(Path(output_paths["overlay_gt"]), Path(tmp_dir) / "overlay_gt" / "case_a.jpg")
            self.assertEqual(Path(output_paths["gradcam_map"]), Path(tmp_dir) / "gradcam_map" / "case_a.jpg")

    def test_batch_shell_script_uses_directory_arguments(self):
        script_path = Path("/Users/wangbd/sysu/my_dino_unet/scripts/gradcam/gradcam_single_image_seg_BM_batch.sh")

        self.assertTrue(script_path.is_file(), "批量调用脚本应已创建")
        content = script_path.read_text(encoding="utf-8")
        self.assertIn("--image_dir", content)
        self.assertIn("--mask_dir", content)
        self.assertIn("--output_type all", content)


if __name__ == "__main__":
    unittest.main()
