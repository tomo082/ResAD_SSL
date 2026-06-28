import argparse
import csv
import importlib
import json
import math
import os
import random
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
OPENAI_CLIP_BPE_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"


MVTec_CLASSES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

VISA_CLASSES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AdaCLIP text-reference anomaly maps without ResAD/Flow fusion."
    )
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--class_name", type=str, default="all")
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument("--ref_root", type=str, default="")
    parser.add_argument("--ref_selection", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--score_mode", type=str, default="text_ref",
                        choices=["text_ref", "cos_only", "residual_norm", "adaclip_text"])
    parser.add_argument("--image_score", type=str, default="topk", choices=["max", "topk", "mean"])
    parser.add_argument("--topk_ratio", type=float, default=0.01)
    parser.add_argument("--score_norm", type=str, default="raw", choices=["raw", "image_minmax", "both"])
    parser.add_argument("--gaussian_sigma", type=float, default=0.0)
    parser.add_argument("--chunk_size", type=int, default=8192)

    parser.add_argument("--clip_layer", type=int, default=24)
    parser.add_argument("--clip_image_size", type=int, default=336)
    parser.add_argument("--adaclip_repo_url", type=str, default="https://github.com/tomo082/AdaCLIP_res")
    parser.add_argument("--adaclip_repo_path", type=str, default="")
    parser.add_argument("--adaclip_checkpoint", type=str, default="")
    parser.add_argument("--adaclip_checkpoint_url", type=str, default="")
    parser.add_argument("--adaclip_cache_dir", type=str, default="~/.cache/adaclip_res")
    parser.add_argument("--adaclip_model", type=str, default="ViT-L-14-336")
    parser.add_argument("--adaclip_prompt_mode", type=str, default="hybrid",
                        choices=["hybrid", "static_only", "dynamic_only"])

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_visuals", action="store_true")
    parser.add_argument("--max_visuals_per_class", type=int, default=25)
    return parser.parse_args()


class ReferenceImageDataset(Dataset):
    def __init__(self, image_paths, image_size):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), str(self.image_paths[idx])


class AdaCLIPTextRefExtractor(torch.nn.Module):
    """Small AdaCLIP runtime for projected patch tokens and text features.

    The AdaCLIP checkpoint and repository are only used for feature extraction.
    ResAD constraintor, VQ, and flow modules are intentionally not constructed.
    """

    def __init__(
        self,
        repo_url,
        repo_path,
        checkpoint,
        checkpoint_url,
        cache_dir,
        model_name,
        layer,
        image_size,
        prompt_mode,
        device,
    ):
        super().__init__()
        self.repo_url = repo_url
        self.repo_path = repo_path
        self.checkpoint = checkpoint
        self.checkpoint_url = checkpoint_url
        self.cache_dir = Path(cache_dir).expanduser()
        self.model_name = model_name
        self.layer = layer
        self.image_size = image_size
        self.prompt_mode = prompt_mode
        self.device = torch.device(device)

        self._clip_mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        self._clip_std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self._imagenet_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._imagenet_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

        repo = self._resolve_repo_path()
        ckpt = self._resolve_checkpoint_path()
        self.trainer = self._build_trainer(repo, ckpt)
        self.model = self.trainer.clip_model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _resolve_repo_path(self):
        if self.repo_path:
            repo = Path(self.repo_path).expanduser().resolve()
            if not repo.exists():
                raise FileNotFoundError(f"AdaCLIP repo path not found: {repo}")
            return repo

        repo = self.cache_dir / "repos" / "AdaCLIP_res"
        if repo.exists():
            return repo.resolve()

        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"[AdaCLIP] cloning repo to {repo}")
        subprocess.run(["git", "clone", self.repo_url, str(repo)], check=True)
        return repo.resolve()

    def _resolve_checkpoint_path(self):
        if self.checkpoint:
            path = Path(self.checkpoint).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"AdaCLIP checkpoint not found: {path}")
            return path

        path = self.cache_dir / "adaclip_checkpoint.pth"
        if path.exists():
            return path.resolve()
        if not self.checkpoint_url:
            raise ValueError(
                "Provide --adaclip_checkpoint or --adaclip_checkpoint_url for AdaCLIP weights."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[AdaCLIP] downloading checkpoint to {path}")
        torch.hub.download_url_to_file(self.checkpoint_url, str(path), progress=True)
        return path.resolve()

    def _build_trainer(self, repo_path, checkpoint_path):
        self._ensure_tokenizer_assets(repo_path)
        sys.path.insert(0, str(repo_path))
        try:
            method_module = importlib.import_module("method")
            trainer_cls = getattr(method_module, "AdaCLIP_Trainer")
        except Exception as exc:
            raise ImportError(f"Failed to import AdaCLIP_Trainer from {repo_path}") from exc

        config = self._load_model_config(repo_path)
        trainer = trainer_cls(
            backbone=self.model_name,
            feat_list=[self.layer],
            input_dim=config["vision_cfg"]["width"],
            output_dim=config["embed_dim"],
            learning_rate=0.0,
            device=str(self.device),
            image_size=self.image_size,
            prompting_depth=4,
            prompting_length=5,
            prompting_branch="VL",
            prompting_type="SD",
            use_hsf=True,
            k_clusters=20,
        )
        trainer.load(str(checkpoint_path))
        trainer.clip_model.to(self.device)
        trainer.clip_model.eval()
        return trainer

    def _ensure_tokenizer_assets(self, repo_path):
        bpe_path = Path(repo_path) / "method" / "bpe_simple_vocab_16e6.txt.gz"
        if self._is_valid_gzip(bpe_path):
            return

        bpe_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = bpe_path.with_suffix(bpe_path.suffix + ".download")
        print(f"[AdaCLIP] tokenizer BPE missing or invalid; downloading to {bpe_path}")
        try:
            torch.hub.download_url_to_file(OPENAI_CLIP_BPE_URL, str(tmp_path), progress=True)
            if not self._is_valid_gzip(tmp_path):
                raise RuntimeError("downloaded BPE file is not a valid gzip archive")
            os.replace(tmp_path, bpe_path)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                "Failed to prepare AdaCLIP tokenizer BPE. If the cached repo contains a "
                "Git LFS pointer, install git-lfs and run `git lfs pull`, or manually "
                f"download {OPENAI_CLIP_BPE_URL} to {bpe_path}."
            ) from exc

    @staticmethod
    def _is_valid_gzip(path):
        path = Path(path)
        if not path.is_file():
            return False
        try:
            with path.open("rb") as handle:
                return handle.read(2) == b"\x1f\x8b"
        except OSError:
            return False

    def _load_model_config(self, repo_path):
        config_path = Path(repo_path) / "model_configs" / f"{self.model_name}.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"AdaCLIP model config does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _normalize_for_adaclip(self, images):
        mean = self._imagenet_mean.to(images.device, images.dtype)
        std = self._imagenet_std.to(images.device, images.dtype)
        clip_mean = self._clip_mean.to(images.device, images.dtype)
        clip_std = self._clip_std.to(images.device, images.dtype)
        images = images * std + mean
        return (images - clip_mean) / clip_std

    @staticmethod
    def _prompt_type_for_mode(original_prompt_type, prompt_mode):
        if prompt_mode == "hybrid":
            return original_prompt_type
        if prompt_mode == "static_only":
            if "S" not in original_prompt_type:
                raise ValueError("prompt_mode='static_only' requires static prompts in prompting_type.")
            return "S"
        if prompt_mode == "dynamic_only":
            if "D" not in original_prompt_type:
                raise ValueError("prompt_mode='dynamic_only' requires dynamic prompts in prompting_type.")
            return "D"
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    def _capture_prompt_state(self):
        state = {}
        for name in ("prompting_type",):
            if hasattr(self.model, name):
                state[name] = getattr(self.model, name)
        for name in ("text_prompter", "visual_prompter"):
            module = getattr(self.model, name, None)
            if module is not None and hasattr(module, "prompting_type"):
                state[f"{name}.prompting_type"] = module.prompting_type
        return state

    def _set_prompt_type(self, prompt_type):
        if hasattr(self.model, "prompting_type"):
            self.model.prompting_type = prompt_type
        for name in ("text_prompter", "visual_prompter"):
            module = getattr(self.model, name, None)
            if module is not None and hasattr(module, "prompting_type"):
                module.prompting_type = prompt_type

    def _restore_prompt_state(self, state):
        if "prompting_type" in state:
            self.model.prompting_type = state["prompting_type"]
        for name in ("text_prompter", "visual_prompter"):
            key = f"{name}.prompting_type"
            module = getattr(self.model, name, None)
            if key in state and module is not None and hasattr(module, "prompting_type"):
                module.prompting_type = state[key]

    @contextmanager
    def _prompt_mode_context(self):
        state = self._capture_prompt_state()
        original = state.get("prompting_type", getattr(self.model, "prompting_type", "SD"))
        prompt_type = self._prompt_type_for_mode(original, self.prompt_mode)
        self._set_prompt_type(prompt_type)
        try:
            yield
        finally:
            self._restore_prompt_state(state)

    @torch.no_grad()
    def extract(self, images, class_names):
        images = self._normalize_for_adaclip(images.to(self.device))
        if isinstance(class_names, str):
            class_names = [class_names] * images.shape[0]
        else:
            class_names = list(class_names)

        with self._prompt_mode_context():
            with torch.cuda.amp.autocast(enabled=images.is_cuda):
                _, proj_patch_tokens, text_features = self.model.extract_feat(images, class_names)

        tokens = self._select_layer_tokens(proj_patch_tokens)
        tokens = tokens.float()
        text_features = text_features.float()
        return tokens, text_features

    def _select_layer_tokens(self, proj_patch_tokens):
        if isinstance(proj_patch_tokens, (list, tuple)):
            if len(proj_patch_tokens) != 1:
                return proj_patch_tokens[-1]
            return proj_patch_tokens[0]
        return proj_patch_tokens


def list_image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def get_ref_image_paths(args, class_name):
    root = Path(args.ref_root or args.data_root)
    if args.dataset == "mvtec":
        candidates = list_image_files(root / class_name / "train" / "good")
    elif args.dataset == "visa":
        candidates = get_visa_train_normal_paths(root, class_name)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if len(candidates) < args.num_ref_shot:
        raise ValueError(
            f"{class_name}: requested {args.num_ref_shot} reference shots, found {len(candidates)}"
        )
    if args.ref_selection == "random":
        rng = random.Random(args.seed)
        candidates = rng.sample(candidates, args.num_ref_shot)
    else:
        candidates = candidates[:args.num_ref_shot]
    return candidates


def get_visa_train_normal_paths(root, class_name):
    csv_path = Path(root) / "split_csv" / "1cls.csv"
    if not csv_path.exists():
        # Few-shot folders are often exported in an MVTec-like layout.
        fallback = Path(root) / class_name / "train" / "good"
        return list_image_files(fallback)

    paths = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("object") != class_name:
                continue
            if row.get("split") != "train" or row.get("label") != "normal":
                continue
            image = row.get("image", "")
            image = image[1:] if image.startswith("/") else image
            paths.append(Path(root) / image)
    return sorted(paths)


def get_classes(dataset, class_name):
    classes = MVTec_CLASSES if dataset == "mvtec" else VISA_CLASSES
    if class_name == "all":
        return classes
    requested = [c.strip() for c in class_name.split(",") if c.strip()]
    unknown = [c for c in requested if c not in classes]
    if unknown:
        raise ValueError(f"Unknown class(es) for {dataset}: {unknown}")
    return requested


def build_test_dataset(args, class_name):
    kwargs = dict(
        root=args.data_root,
        class_name=class_name,
        train=False,
        normalize="w50",
        img_size=args.clip_image_size,
        crp_size=args.clip_image_size,
        msk_size=args.clip_image_size,
        msk_crp_size=args.clip_image_size,
    )
    if args.dataset == "mvtec":
        from datasets.mvtec import MVTEC
        return MVTEC(**kwargs)
    if args.dataset == "visa":
        from datasets.visa import VISA
        return VISA(**kwargs)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def infer_patch_grid(tokens):
    n = tokens.shape[1]
    side = int(math.sqrt(n))
    if side * side == n:
        return tokens, side, side

    side = int(math.sqrt(n - 1))
    if side * side == n - 1:
        return tokens[:, 1:, :], side, side

    raise ValueError(f"Cannot infer square patch grid from token count {n}")


def extract_text_pair(text_features, batch_size):
    if text_features.dim() == 2:
        text_features = text_features.unsqueeze(0).expand(batch_size, -1, -1)

    if text_features.dim() != 3:
        raise ValueError(f"Unsupported text_features shape: {tuple(text_features.shape)}")

    if text_features.shape[-1] == 2:
        t_n = text_features[:, :, 0]
        t_a = text_features[:, :, 1]
    elif text_features.shape[1] == 2:
        t_n = text_features[:, 0, :]
        t_a = text_features[:, 1, :]
    else:
        raise ValueError(f"Cannot identify normal/abnormal text features: {tuple(text_features.shape)}")

    if t_n.shape[0] == 1 and batch_size > 1:
        t_n = t_n.expand(batch_size, -1)
        t_a = t_a.expand(batch_size, -1)
    return t_n, t_a


def nearest_reference(query_tokens, memory_tokens, chunk_size):
    query_n = F.normalize(query_tokens, p=2, dim=1)
    memory_n = F.normalize(memory_tokens, p=2, dim=1)
    matched = []
    for start in range(0, query_tokens.shape[0], chunk_size):
        end = min(start + chunk_size, query_tokens.shape[0])
        sim = query_n[start:end] @ memory_n.T
        idx = torch.argmax(sim, dim=1)
        matched.append(memory_tokens[idx])
    return torch.cat(matched, dim=0)


def compute_patch_scores(tokens, text_features, memory_tokens, score_mode, chunk_size):
    tokens, grid_h, grid_w = infer_patch_grid(tokens)
    b, n, c = tokens.shape
    flat_q = tokens.reshape(-1, c)

    t_n, t_a = extract_text_pair(text_features, b)
    t_n = t_n.to(tokens.device, tokens.dtype)
    t_a = t_a.to(tokens.device, tokens.dtype)

    if score_mode == "adaclip_text":
        q_n = F.normalize(tokens, p=2, dim=-1)
        tn = F.normalize(t_n, p=2, dim=1).unsqueeze(1)
        ta = F.normalize(t_a, p=2, dim=1).unsqueeze(1)
        logits = torch.stack([
            (q_n * tn).sum(dim=-1),
            (q_n * ta).sum(dim=-1),
        ], dim=-1)
        scores = torch.softmax(logits, dim=-1)[..., 1]
        return scores.reshape(b, grid_h, grid_w)

    matched = nearest_reference(flat_q, memory_tokens.to(tokens.device, tokens.dtype), chunk_size)
    residual = flat_q - matched
    residual_norm = torch.linalg.norm(residual, dim=1)

    rt = (t_a - t_n).repeat_interleave(n, dim=0)
    cos = (F.normalize(residual, p=2, dim=1) * F.normalize(rt, p=2, dim=1)).sum(dim=1)
    cos = torch.relu(cos)

    if score_mode == "text_ref":
        scores = residual_norm * cos
    elif score_mode == "cos_only":
        scores = cos
    elif score_mode == "residual_norm":
        scores = residual_norm
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    return scores.reshape(b, grid_h, grid_w)


def upsample_maps(patch_maps, target_hw):
    maps = patch_maps.unsqueeze(1)
    maps = F.interpolate(maps, size=target_hw, mode="bilinear", align_corners=False)
    return maps[:, 0]


def normalize_imagewise(maps):
    out = maps.copy()
    for i in range(out.shape[0]):
        mn = float(out[i].min())
        mx = float(out[i].max())
        if mx > mn:
            out[i] = (out[i] - mn) / (mx - mn)
        else:
            out[i] = 0.0
    return out


def maybe_smooth(maps, sigma):
    if sigma <= 0:
        return maps
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise ImportError("--gaussian_sigma requires scipy") from exc
    return np.stack([gaussian_filter(m, sigma=sigma) for m in maps], axis=0)


def aggregate_image_scores(maps, mode, topk_ratio):
    flat = maps.reshape(maps.shape[0], -1)
    if mode == "max":
        return flat.max(axis=1)
    if mode == "mean":
        return flat.mean(axis=1)
    if mode == "topk":
        k = max(1, int(flat.shape[1] * topk_ratio))
        idx = np.argpartition(flat, -k, axis=1)[:, -k:]
        return np.take_along_axis(flat, idx, axis=1).mean(axis=1)
    raise ValueError(f"Unsupported image_score: {mode}")


def safe_roc_auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_average_precision(labels, scores):
    from sklearn.metrics import average_precision_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def compute_aupro(gt_masks, scores):
    if gt_masks.sum() <= 0 or float(scores.max()) <= float(scores.min()):
        return float("nan")
    try:
        from utils import calculate_aupro
        return float(calculate_aupro(gt_masks, scores))
    except Exception:
        return float("nan")


def compute_metrics(labels, gt_masks, maps, image_score, topk_ratio):
    labels = np.asarray(labels).astype(np.int64)
    gt_masks = np.asarray(gt_masks).astype(np.uint8)
    maps = np.asarray(maps).astype(np.float32)
    image_scores = aggregate_image_scores(maps, image_score, topk_ratio)

    return {
        "image_auc": safe_roc_auc(labels, image_scores),
        "image_ap": safe_average_precision(labels, image_scores),
        "pixel_auc": safe_roc_auc(gt_masks.reshape(-1), maps.reshape(-1)),
        "pixel_ap": safe_average_precision(gt_masks.reshape(-1), maps.reshape(-1)),
        "aupro": compute_aupro(gt_masks, maps),
    }


def denorm_imagenet(image_tensor):
    image = image_tensor.detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_visual(path, image_tensor, mask, score_map, title):
    import matplotlib.pyplot as plt

    image = denorm_imagenet(image_tensor)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(image)
    axes[0].set_title("input")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("gt")
    im = axes[2].imshow(score_map, cmap="jet")
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def unpack_batch(batch):
    images = batch[0]
    labels = batch[1]
    masks = batch[2]
    return images, labels, masks


@torch.no_grad()
def build_memory_bank(args, extractor, class_name):
    paths = get_ref_image_paths(args, class_name)
    dataset = ReferenceImageDataset(paths, args.clip_image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    chunks = []
    for images, _ in loader:
        images = images.to(args.device)
        class_names = [class_name] * images.shape[0]
        tokens, _ = extractor.extract(images, class_names)
        tokens, _, _ = infer_patch_grid(tokens)
        chunks.append(tokens.reshape(-1, tokens.shape[-1]).cpu())
    memory = torch.cat(chunks, dim=0).float()
    print(f"[Reference] {class_name}: {len(paths)} images, memory={tuple(memory.shape)}")
    return memory


@torch.no_grad()
def evaluate_class(args, extractor, class_name):
    save_dir = Path(args.save_dir)
    class_dir = save_dir / class_name
    maps_dir = class_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    memory = None
    if args.score_mode != "adaclip_text":
        memory = build_memory_bank(args, extractor, class_name)
    dataset = build_test_dataset(args, class_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_raw_maps = []
    all_labels = []
    all_masks = []
    all_paths = []
    visual_count = 0
    sample_idx = 0

    for batch in loader:
        images, labels, masks = unpack_batch(batch)
        images = images.to(args.device)
        masks = masks.float()
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        class_names = [class_name] * images.shape[0]

        tokens, text_features = extractor.extract(images, class_names)
        patch_maps = compute_patch_scores(
            tokens=tokens,
            text_features=text_features,
            memory_tokens=memory,
            score_mode=args.score_mode,
            chunk_size=args.chunk_size,
        )
        score_maps = upsample_maps(patch_maps, target_hw=masks.shape[-2:]).cpu().numpy()

        labels_np = labels.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        score_maps = maybe_smooth(score_maps.astype(np.float32), args.gaussian_sigma)

        for b in range(score_maps.shape[0]):
            image_path = getattr(dataset, "image_paths", [None] * len(dataset))[sample_idx]
            image_path = str(image_path) if image_path is not None else f"{class_name}_{sample_idx:06d}"
            stem = Path(image_path).stem
            np.save(maps_dir / f"{sample_idx:06d}_{stem}_raw.npy", score_maps[b])
            all_paths.append(image_path)
            if args.save_visuals and visual_count < args.max_visuals_per_class:
                save_visual(
                    class_dir / "visuals" / f"{sample_idx:06d}_{stem}_raw.png",
                    images[b],
                    masks_np[b],
                    score_maps[b],
                    args.score_mode,
                )
                visual_count += 1
            sample_idx += 1

        all_raw_maps.append(score_maps)
        all_labels.append(labels_np)
        all_masks.append(masks_np)

    raw_maps = np.concatenate(all_raw_maps, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    norm_variants = ["raw", "image_minmax"] if args.score_norm == "both" else [args.score_norm]
    rows = []
    for norm_name in norm_variants:
        maps = raw_maps if norm_name == "raw" else normalize_imagewise(raw_maps)
        if norm_name == "image_minmax":
            for idx, path in enumerate(all_paths):
                stem = Path(path).stem
                np.save(maps_dir / f"{idx:06d}_{stem}_image_minmax.npy", maps[idx])
        metrics = compute_metrics(labels, masks, maps, args.image_score, args.topk_ratio)
        row = {
            "class_name": class_name,
            "score_mode": args.score_mode,
            "score_norm": norm_name,
            "image_score": args.image_score,
            "topk_ratio": args.topk_ratio,
            **metrics,
        }
        rows.append(row)
        print(
            f"[Class {class_name} | {args.num_ref_shot}-shot | {args.score_mode} | {norm_name}] "
            f"Image AUROC: {metrics['image_auc']:.4f} | AP: {metrics['image_ap']:.4f} | "
            f"Pixel AUROC: {metrics['pixel_auc']:.4f} | AP: {metrics['pixel_ap']:.4f} | "
            f"AUPRO: {metrics['aupro']:.4f}"
        )

    with (class_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows


def save_metrics_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_name",
        "score_mode",
        "score_norm",
        "image_score",
        "topk_ratio",
        "image_auc",
        "image_ap",
        "pixel_auc",
        "pixel_ap",
        "aupro",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_average_rows(rows):
    average_rows = []
    keys = sorted(set((r["score_norm"], r["image_score"]) for r in rows))
    for score_norm, image_score in keys:
        subset = [r for r in rows if r["score_norm"] == score_norm and r["image_score"] == image_score]
        avg = {
            "class_name": "average",
            "score_mode": subset[0]["score_mode"],
            "score_norm": score_norm,
            "image_score": image_score,
            "topk_ratio": subset[0]["topk_ratio"],
        }
        for metric in ("image_auc", "image_ap", "pixel_auc", "pixel_ap", "aupro"):
            avg[metric] = float(np.nanmean([r[metric] for r in subset]))
        average_rows.append(avg)
        print(
            f"[Average | {avg['score_mode']} | {score_norm}] "
            f"Image AUROC: {avg['image_auc']:.4f} | AP: {avg['image_ap']:.4f} | "
            f"Pixel AUROC: {avg['pixel_auc']:.4f} | AP: {avg['pixel_ap']:.4f} | "
            f"AUPRO: {avg['aupro']:.4f}"
        )
    return average_rows


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    classes = get_classes(args.dataset, args.class_name)

    print("[TextRefMap] dataset:", args.dataset)
    print("[TextRefMap] classes:", classes)
    print("[TextRefMap] score_mode:", args.score_mode)
    print("[TextRefMap] image_score:", args.image_score)
    print("[TextRefMap] score_norm:", args.score_norm)
    print("[AdaCLIP] layer:", args.clip_layer)
    print("[AdaCLIP] prompt_mode:", args.adaclip_prompt_mode)

    extractor = AdaCLIPTextRefExtractor(
        repo_url=args.adaclip_repo_url,
        repo_path=args.adaclip_repo_path,
        checkpoint=args.adaclip_checkpoint,
        checkpoint_url=args.adaclip_checkpoint_url,
        cache_dir=args.adaclip_cache_dir,
        model_name=args.adaclip_model,
        layer=args.clip_layer,
        image_size=args.clip_image_size,
        prompt_mode=args.adaclip_prompt_mode,
        device=args.device,
    )

    all_rows = []
    for class_name in classes:
        all_rows.extend(evaluate_class(args, extractor, class_name))

    all_rows_with_avg = all_rows + add_average_rows(all_rows)
    save_metrics_csv(Path(args.save_dir) / "metrics.csv", all_rows_with_avg)
    with (Path(args.save_dir) / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_rows_with_avg, f, indent=2, ensure_ascii=False)
    print("[TextRefMap] saved:", Path(args.save_dir).resolve())


if __name__ == "__main__":
    main()
