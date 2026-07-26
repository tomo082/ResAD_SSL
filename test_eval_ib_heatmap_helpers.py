from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from eval_ib_heatmap import (
    compute_display_range,
    normalize_dataset_name,
    num_images_to_save,
    resolve_class_names,
    validate_reference_files,
)


class EvalImageBindHeatmapHelperTest(unittest.TestCase):
    def test_dataset_aliases(self):
        self.assertEqual(normalize_dataset_name("mvtec_loco"), "mvtecloco")
        self.assertEqual(normalize_dataset_name("mvtec-loco"), "mvtecloco")
        self.assertEqual(normalize_dataset_name("mvtec_3d"), "mvtec3d")
        self.assertEqual(normalize_dataset_name("MVTec-3D"), "mvtec3d")

    def test_max_images_controls_only_save_count(self):
        self.assertEqual(num_images_to_save(12, -1), 12)
        self.assertEqual(num_images_to_save(12, 0), 0)
        self.assertEqual(num_images_to_save(12, 5), 5)
        self.assertEqual(num_images_to_save(3, 5), 3)
        with self.assertRaises(ValueError):
            num_images_to_save(12, -2)

    def test_invalid_class_lists_available_classes(self):
        args = SimpleNamespace(class_name="screw")
        registry = {"visa": {"class_names": ["candle", "capsules"]}}
        with self.assertRaisesRegex(ValueError, "candle, capsules"):
            resolve_class_names(args, "visa", registry)

    def test_display_range_is_finite_for_constant_map(self):
        vmin, vmax = compute_display_range(np.ones((2, 4, 4)))
        self.assertTrue(np.isfinite(vmin))
        self.assertTrue(np.isfinite(vmax))
        self.assertLess(vmin, vmax)

    def test_reference_validation_requires_all_four_layers(self):
        with TemporaryDirectory() as directory:
            class_dir = Path(directory) / "screw"
            class_dir.mkdir()
            for level in range(1, 4):
                np.save(
                    class_dir / f"layer{level}.npy",
                    np.zeros((1024, 1280), dtype=np.float32),
                )
            args = SimpleNamespace(
                test_ref_feature_dir=directory,
                total_ref_shot=4,
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "layer4.npy"
            ):
                validate_reference_files(args, "mvtec", ["screw"])

    def test_reference_validation_rejects_bad_shot_partition(self):
        with TemporaryDirectory() as directory:
            class_dir = Path(directory) / "screw"
            class_dir.mkdir()
            for level in range(1, 5):
                np.save(
                    class_dir / f"layer{level}.npy",
                    np.zeros((1023, 1280), dtype=np.float32),
                )
            args = SimpleNamespace(
                test_ref_feature_dir=directory,
                total_ref_shot=4,
            )
            with self.assertRaisesRegex(ValueError, "not divisible"):
                validate_reference_files(args, "mvtec", ["screw"])


if __name__ == "__main__":
    unittest.main()
