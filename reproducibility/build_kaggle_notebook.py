"""Build the Kaggle notebook for BraTS-Africa inference with pre-trained models.

The notebook inlines the repo's exact model / preprocessing / metrics source
files so the Kaggle kernel is self-contained, then adds dataset discovery,
sliding-window inference, metrics tables and figure generation.

Run:  python build_kaggle_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "kaggle_kernel"
OUT_DIR.mkdir(exist_ok=True)


def src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = []

# ---------------------------------------------------------------- 0. header
CELLS.append(md("""# 3D Brain Tumor Segmentation — Pre-trained UNet3D vs UNETR on BraTS-Africa

End-to-end **inference-only** project by **Prakhar Gupta (ZekRock18)**,
with project teammate **Aditya Dubey**.

**Project repository:** [ZekRock18/brain-tumor-segmentation](https://github.com/ZekRock18/brain-tumor-segmentation)

Kaggle integration, data adaptation, evaluation and visual reporting build on the
original architectures and pretrained checkpoints from
[Brain_Tumor_Benchmark](https://github.com/MohanadMahran/Brain_Tumor_Benchmark).
No model training or fine-tuning was performed in this demonstration.

* Loads the **pre-trained** checkpoints (no training, no fine-tuning).
* Runs them on real tumour cases from the **BraTS-Africa** dataset.
* Reports **Dice** and **HD95** per region (ET / TC / WT).
* Renders axial slices: T1ce input, ground truth, and both model predictions.

| Section | What happens |
|---|---|
| 0. Setup | Install deps, pick device (GPU if available), seed |
| 1. Models | `UNet3D` and `UNETR` — inlined verbatim from the repo |
| 2. Preprocessing + Metrics | NIfTI Z-score pipeline and Dice/HD95 — inlined verbatim from the repo |
| 3. Weights | Load pre-trained checkpoints **strictly** from the attached Kaggle dataset |
| 4. Dataset | Discover BraTS-Africa cases (`t1n`/`t1c`/`t2w`/`t2f`/`seg`) |
| 5. Inference | Sliding-window inference (roi 128^3) for both models |
| 6. Results | Per-case metrics table, aggregate table, CSV |
| 7. Figures | Axial slice overlays: T1ce / GT / UNet3D / UNETR |

**Attached datasets**

| Dataset | Purpose |
|---|---|
| `zekrock/brats-weights` | Pre-trained checkpoints (UNet3D + UNETR) |
| `yakunokamizen/brats-africa` | BraTS-Africa cases (146 volumes with ground-truth segmentations) |
"""))

# ---------------------------------------------------------------- 1. setup
CELLS.append(code(r'''# ============================================================
# 0. SETUP — dependencies, device, seed
# ============================================================
# MONAI is optional; the tested fallback does not require a network install.
import importlib.util, subprocess, sys
if importlib.util.find_spec("monai") is None:
    install = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                              "--retries", "0", "--timeout", "15", "monai"],
                             capture_output=True, text=True)
    if install.returncode:
        print("MONAI could not be installed; using the tested built-in sliding window.")

import os, time, random, logging
import json, hashlib, platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REQUIRE_GPU = True
if REQUIRE_GPU and DEVICE.type != "cuda":
    raise RuntimeError("GPU requested but CUDA is unavailable. Verify Kaggle account access, select GPU T4 x2, and start a new session. Set REQUIRE_GPU=False only for an intentional CPU run.")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False


def set_global_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(42)
print("Device:", DEVICE, "| torch:", torch.__version__,
      "| cuda build:", torch.version.cuda if DEVICE.type == "cuda" else "N/A")
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("Input root:", Path("/kaggle/input"))
for p in sorted(Path("/kaggle/input").glob("*")):
    sub = sorted(q.name for q in p.iterdir()) if p.is_dir() else ["<file>"]
    print(f"  attached: {p.name} -> {sub[:12]}")
'''))

# ---------------------------------------------------------------- 2. models
CELLS.append(md("""## 1. Model architectures — inlined verbatim from the repo

These are the exact classes the pre-trained weights were saved against, so
`load_state_dict(..., strict=True)` succeeds in section 3."""))

CELLS.append(code(
    "# ============================================================\n"
    "# 1a. 3D U-Net — repo/models/unet3d.py (verbatim)\n"
    "# ============================================================\n"
    + src("repo/models/unet3d.py")
))

CELLS.append(code(
    "# ============================================================\n"
    "# 1b. UNETR — repo/models/unetr.py (verbatim)\n"
    "# ============================================================\n"
    + src("repo/models/unetr.py")
))

# ---------------------------------------------------------------- 3. prep + metrics
CELLS.append(md("""## 2. Preprocessing and metrics — inlined verbatim from the repo

Preprocessing: per-modality Z-score inside the brain mask, label map to
3 region channels with the `3 -> 4` remap (UPenn-GBM compatibility).

Metrics: 3D Dice and 95th-percentile Hausdorff distance, implemented from scratch."""))

CELLS.append(code(
    "# ============================================================\n"
    "# 2a. Preprocessing — repo/data/preprocessing.py (verbatim)\n"
    "# ============================================================\n"
    + src("repo/data/preprocessing.py")
))

CELLS.append(code(
    "# ============================================================\n"
    "# 2b. Metrics — repo/training/metrics.py (verbatim)\n"
    "# ============================================================\n"
    + src("repo/training/metrics.py")
))

# ---------------------------------------------------------------- 4. weights
CELLS.append(md("""## 3. Load the pre-trained checkpoints

Weights come from the attached `zekrock/brats-weights` dataset. Each `.bin` is a
`torch.save` dict `{"config": ..., "model_state_dict": ...}`; the model is rebuilt
from the saved config and loaded with `strict=True`."""))

CELLS.append(code(r'''# ============================================================
# 3. LOAD PRE-TRAINED WEIGHTS (strict) AND BUILD MODELS
# ============================================================
WEIGHT_FILES = {
    "unet3d": "unet3d_pytorch_model.bin",
    "unetr": "unetr_pytorch_model.bin",
}

INPUT_ROOT = Path("/kaggle/input")


def resolve_weight_path(filename: str) -> Optional[Path]:
    """Find the checkpoint in the attached dataset, else fall back to HF Hub."""
    hits = sorted(INPUT_ROOT.rglob(filename))
    if hits:
        return hits[0]
    try:
        from huggingface_hub import hf_hub_download
        repo = ("MohanadKombar/brain-tumor-unet3d-brats" if filename.startswith("unet3d")
                else "MohanadKombar/brain-tumor-unetr-brats")
        return Path(hf_hub_download(repo_id=repo, filename="pytorch_model.bin"))
    except Exception as e:
        logging.warning("Could not resolve %s (%s)", filename, e)
        return None


def load_unet3d(ckpt_path, device=DEVICE):
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = c.get("config", {})
    m = UNet3D(in_channels=cfg.get("in_channels", 4),
               out_channels=cfg.get("out_channels", 3),
               base_channels=cfg.get("base_channels", 60),
               groups=cfg.get("groups", 4),
               dropout_rates=cfg.get("dropout_rates", [0.0, 0.0, 0.0, 0.0]),
               drop_path_rate=cfg.get("drop_path_rate", 0.1))
    m.load_state_dict(c["model_state_dict"], strict=True)
    return m, c


def load_unetr(ckpt_path, device=DEVICE):
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = c.get("config", {})
    ps = cfg.get("patch_size", [128, 128, 128])
    input_size = ps[0] if isinstance(ps, (list, tuple)) else ps
    m = UNETR(in_channels=cfg.get("in_channels", 4),
              out_channels=cfg.get("out_channels", 3),
              input_size=input_size,
              patch_size=cfg.get("patch_size_tokens", 16),
              embedding_dim=cfg.get("embedding_dim", 384),
              num_layers=cfg.get("num_layers", 6),
              num_heads=cfg.get("num_heads", 8),
              mlp_ratio=cfg.get("mlp_ratio", 4),
              dropout=cfg.get("dropout", 0.0),
              drop_path_rate=cfg.get("drop_path_rate", 0.1),
              use_checkpoint=False)
    m.load_state_dict(c["model_state_dict"], strict=True)
    return m, c


MODEL, META = {}, {}
for name, fname in WEIGHT_FILES.items():
    path = resolve_weight_path(fname)
    if path is None:
        raise FileNotFoundError(f"Pre-trained weights for '{name}' not found")
    print(f"{name:7s} <- {path}  ({path.stat().st_size/1e6:.1f} MB)")
    MODEL[name], META[name] = (load_unet3d(path) if name == "unet3d" else load_unetr(path))
    with open(path, "rb") as weight_file:
        META[name]["sha256"] = hashlib.file_digest(weight_file, "sha256").hexdigest()

for name in MODEL:
    MODEL[name].to(DEVICE).eval()
    n_params = sum(p.numel() for p in MODEL[name].parameters())
    meta = META[name]
    print(f"{name:7s} params={n_params:,}  val_dice={meta.get('val_dice_mean', float('nan')):.4f}  "
          f"epoch={meta.get('epoch')}")

# Forward smoke test
with torch.no_grad():
    _x = torch.randn(1, 4, 128, 128, 128, device=DEVICE)
    for name in MODEL:
        _o = MODEL[name](_x)
        assert _o.shape == (1, 3, 128, 128, 128), (name, _o.shape)
print("Forward smoke test OK: (1,4,128,128,128) -> (1,3,128,128,128)")
del _x, _o
'''))

# ---------------------------------------------------------------- 5. dataset
CELLS.append(md("""## 4. Discover the BraTS-Africa cases

The dataset contains nested patient folders, with files named `BraTS-SSA-<id>-<modality>.nii` and
modality suffixes `t1n` (T1), `t1c` (T1ce), `t2w` (T2), `t2f` (FLAIR) and `seg`
(ground truth). The repo's own convention (`t1`, `t1ce`, `t2`, `flair`) is
restored before preprocessing."""))

CELLS.append(code(r'''# ============================================================
# 4. LOCATE BRAts-AFRICA CASES
# ============================================================
MODALITY_SUFFIX = {
    "t1": "-t1n",
    "t1ce": "-t1c",
    "t2": "-t2w",
    "flair": "-t2f",
    "label": "-seg",
}


def find_africa_root() -> Optional[Path]:
    """Find the BraTS-Africa dataset dir under /kaggle/input.

    Kaggle nests attached datasets under /kaggle/input/<owner>/<slug> or
    /kaggle/input/datasets/<owner>/<slug>, so search recursively.
    """
    for d in sorted(INPUT_ROOT.rglob("*")):
        if d.is_dir() and "africa" in d.name.lower():
            return d
    return None


AFRICA_ROOT = find_africa_root()
if AFRICA_ROOT is None:
    print("Could not auto-locate BraTS-Africa. /kaggle/input contains:")
    for d in sorted(INPUT_ROOT.rglob("*")):
        if d.is_dir():
            print("   ", d)
    raise FileNotFoundError(
        "BraTS-Africa dataset not attached. Add 'yakunokamizen/brats-africa' "
        "as an input to this notebook.")

seg_files = sorted(AFRICA_ROOT.rglob("*-seg.nii*"))
print(f"Dataset root : {AFRICA_ROOT}")
print(f"Ground-truth segmentations found: {len(seg_files)}")


def build_manifest(seg_files: List[Path]) -> pd.DataFrame:
    """Group NIfTI files into cases and find the cohort in ancestor folders."""
    records = []
    for seg in seg_files:
        case_id = seg.name.split("-seg")[0]
        cohort = next((p.name for p in seg.parents
                       if p.name in {"95_Glioma", "51_OtherNeoplasms"}), "unclassified")
        rec = {"case_id": case_id, "cohort": cohort, "label": str(seg)}
        for mod, suffix in MODALITY_SUFFIX.items():
            if mod == "label":
                continue
            hits = sorted(seg.parent.glob(f"{case_id}{suffix}.nii*"))
            rec[mod] = str(hits[0]) if hits else None
        records.append(rec)
    return pd.DataFrame(records)


MANIFEST = build_manifest(seg_files)
if MANIFEST.empty:
    raise RuntimeError("No segmentations found in the attached dataset")
complete = MANIFEST.dropna(subset=["t1", "t1ce", "t2", "flair"]).reset_index(drop=True)
if complete.empty or complete["case_id"].duplicated().any():
    raise RuntimeError("Expected nonempty, unique complete cases")
print(f"Cases with all 4 modalities + seg: {len(complete)}")
print(complete["cohort"].value_counts().to_string())
print()
print(complete.head(5)[["case_id", "cohort"]].to_string(index=False))
'''))

# ---------------------------------------------------------------- 6. loader
CELLS.append(code(r'''# ============================================================
# 5. LOAD + PREPROCESS A CASE
# ============================================================
def load_case(row) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one BraTS-Africa case.

    Returns:
        image: (4, H, W, D) float32, z-score normalised per modality
        label: (3, H, W, D) float32 binary regions [ET, TC, WT]
        seg_raw: (H, W, D) int32 raw label map
    """
    modality_paths = {m: row[m] for m in ("t1", "t1ce", "t2", "flair")}
    reference = nib.load(row["label"])
    for path in modality_paths.values():
        volume = nib.load(path)
        if volume.shape != reference.shape or not np.allclose(volume.affine, reference.affine, atol=1e-4):
            raise ValueError(f"MRI/label geometry mismatch: {path}")
    image, label = preprocess_volume(modality_paths, row["label"])
    seg_raw = load_nifti(row["label"]).astype(np.int32)
    if not set(np.unique(seg_raw)).issubset({0, 1, 2, 3, 4}):
        raise ValueError("Unsupported segmentation labels")
    if not np.isfinite(image).all():
        raise ValueError("Non-finite MRI intensities")
    return image, label, seg_raw


def describe_labels(seg_raw: np.ndarray) -> None:
    vals, counts = np.unique(seg_raw, return_counts=True)
    table = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
    print(f"    raw labels -> {table}")
    unexpected = sorted(set(int(v) for v in vals) - {0, 1, 2, 3, 4})
    if unexpected:
        print(f"    WARNING: unexpected label values {unexpected} "
              f"(3 is remapped to 4 by convert_labels_to_regions)")
    else:
        print("    supported labels: enhancing label 3 is mapped to 4; channels are ET, TC, WT")


# Verify the schema and the shape of one case before running anything heavy
_probe = complete.iloc[0]
_img, _lab, _seg = load_case(_probe)
print(f"Probe case   : {_probe['case_id']} ({_probe['cohort']})")
print(f"image shape  : {_img.shape}  dtype={_img.dtype}")
print(f"label shape  : {_lab.shape}")
describe_labels(_seg)
'''))

# ---------------------------------------------------------------- 7. inference
CELLS.append(md("""## 5. Sliding-window inference

Both models are evaluated exactly the way the repo's benchmark evaluator does it:
MONAI `sliding_window_inference` with a `128^3` ROI and 25% overlap, Gaussian
blending, and mixed precision on GPU."""))

CELLS.append(code(r'''# ============================================================
# 6. SLIDING-WINDOW INFERENCE (self-contained, MONAI optional)
# ============================================================
import warnings


def _gaussian_kernel(roi, sigma_frac: float = 0.125) -> np.ndarray:
    """3D Gaussian weight kernel for a given ROI (outer product of 1D curves)."""
    def _line(size):
        c = (size - 1) / 2.0
        s = size * sigma_frac
        return np.exp(-0.5 * ((np.arange(size) - c) / s) ** 2)
    gh, gw, gd = _line(roi[0]), _line(roi[1]), _line(roi[2])
    return np.maximum(np.einsum("i,j,k->ijk", gh, gw, gd), 1e-3)


def _starts(total: int, size: int, overlap: float):
    """Window start positions along one axis, clamped so windows fit."""
    if total <= size:
        return [0]
    step = max(1, int(size * (1 - overlap)))
    starts = list(range(0, total - size + 1, step))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


def predict_full_volume(model, image: np.ndarray, roi=(128, 128, 128),
                       overlap: float = 0.25, device=DEVICE) -> np.ndarray:
    """Sliding-window inference with Gaussian blending.

    Uses MONAI when available, otherwise a Gaussian-blended PyTorch fallback.

    Args:
        model: A loaded model in eval mode.
        image: (4, H, W, D) preprocessed input.

    Returns:
        (3, H, W, D) float32 probability map.
    """
    model.to(device).eval()
    t = torch.as_tensor(image, dtype=torch.float32, device=device)[None]  # (1,4,H,W,D)
    H, W, D = t.shape[2], t.shape[3], t.shape[4]

    use_monai = False
    try:
        from monai.inferers import sliding_window_inference
        use_monai = True
    except Exception:
        warnings.warn("MONAI not available - using built-in sliding window")
    if use_monai:
        predictor = lambda x: torch.sigmoid(model(x))
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=True):
                    out = sliding_window_inference(t, roi_size=roi, sw_batch_size=1,
                                                   predictor=predictor, overlap=overlap,
                                                   mode="gaussian", progress=False)
            else:
                out = sliding_window_inference(t, roi_size=roi, sw_batch_size=1,
                                               predictor=predictor, overlap=overlap,
                                               mode="gaussian", progress=False)
        return out[0].float().cpu().numpy()

    # ---- Pure-torch Gaussian-blended sliding window ----
    original_shape = (H, W, D)
    pads = [(max(0, r-s)//2, max(0, r-s)-max(0, r-s)//2)
            for s, r in zip(original_shape, roi)]
    t = F.pad(t, tuple(v for pair in reversed(pads) for v in pair))
    H, W, D = t.shape[2:]
    rh, rw, rd = roi
    ker = torch.as_tensor(_gaussian_kernel((rh, rw, rd)), dtype=torch.float32, device=device)
    sy, sx, sz = _starts(H, rh, overlap), _starts(W, rw, overlap), _starts(D, rd, overlap)
    acc = torch.zeros(1, 3, H, W, D, device=device)
    wsum = torch.zeros(1, 1, H, W, D, device=device)
    block = ker[None, None]                       # (1,1,rh,rw,rd)
    with torch.no_grad():
        for iy in sy:
            for ix in sx:
                for iz in sz:
                    patch = t[:, :, iy:iy + rh, ix:ix + rw, iz:iz + rd]
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        p = torch.sigmoid(model(patch)).float()
                    acc[:, :, iy:iy + rh, ix:ix + rw, iz:iz + rd] += p * block
                    wsum[:, :, iy:iy + rh, ix:ix + rw, iz:iz + rd] += block
    out = (acc / wsum.clamp_min(1e-8))[0].float().cpu().numpy()
    return out[:, pads[0][0]:pads[0][0]+original_shape[0],
               pads[1][0]:pads[1][0]+original_shape[1],
               pads[2][0]:pads[2][0]+original_shape[2]]


# The forward smoke test above has already warmed up both models.
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
print(f"Warm-up complete. Inference backend: "
      f"{'MONAI' if __import__('importlib').util.find_spec('monai') else 'built-in'} sliding window")
'''))

# ---------------------------------------------------------------- 8. visualization
CELLS.append(md("""## 6. Visualisation

Each panel shows the same axial slice, chosen automatically as the slice with the
most ground-truth tumour. Colours: **ET** red, **TC** green, **WT** blue."""))

CELLS.append(code(r'''# ============================================================
# 7. FIGURES — axial slice overlays
# ============================================================
import matplotlib.pyplot as plt

# ET red, TC green, WT blue — drawn widest first so ET ends up on top
REGION_COLORS = [(0.20, 0.45, 1.00),   # WT
                 (0.15, 0.90, 0.25),   # TC
                 (1.00, 0.15, 0.15)]   # ET


def to_gray(sl: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
    if hi <= lo:
        hi = lo + 1e-6
    gray = np.clip((sl - lo) / (hi - lo), 0, 1)
    # Preprocessing zero-fills outside the skull-stripped brain.
    gray[sl == 0] = 0
    return gray


def overlay(base: np.ndarray, regions: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the 3 region masks onto a grayscale slice.

    Args:
        base: 2D grayscale slice, shape (H, W).
        regions: 3 region masks for that slice, shape (3, H, W).
    """
    out = np.stack([to_gray(base)] * 3, axis=-1)
    for i, color in zip((2, 1, 0), REGION_COLORS):
        mask = regions[i] > 0.5
        if not mask.any():
            continue
        # Replace from the original grayscale to avoid mixing region colors.
        gray = np.stack([to_gray(base)] * 3, axis=-1)
        out[mask] = (1 - alpha) * gray[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 1)


def best_slice_index(gt: np.ndarray) -> int:
    """Axial slice index (along the D axis) with the most ground-truth tumour.

    Volumes are stored as (3, H, W, D), so the axial axis is the last one.
    """
    if gt is None:
        return 0
    wt = gt[2]
    if wt.sum() == 0:
        return wt.shape[-1] // 2
    per_slice = wt.sum(axis=(0, 1))
    return int(np.argmax(per_slice))


def show_case(image, gt, preds: Dict[str, np.ndarray], title: str, save_path=None):
    """Render T1ce / ground truth / each prediction for one axial slice."""
    z = best_slice_index(gt)
    t1c = image[1, :, :, z]          # T1ce, (H, W)
    panels = [("T1ce (input)", to_gray(t1c), None)]
    if gt is not None:
        panels.append(("Ground truth", overlay(t1c, gt[:, :, :, z]), None))
    for name, p in preds.items():
        panels.append((f"{name} prediction", overlay(t1c, p[:, :, :, z]), None))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8))
    if n == 1:
        axes = [axes]
    for ax, (label, arr, _) in zip(axes, panels):
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    fig.suptitle(title + "\nRed: ET | Green: TC outside ET | Blue: WT outside TC", fontsize=11)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    plt.close(fig)
'''))

# ---------------------------------------------------------------- 9. run
CELLS.append(md("""## 7. Run both pre-trained models on real tumour cases

`N_CASES` controls how many BraTS-Africa cases are scored (the dataset has 146).
`SHOW_FIGURES` controls how many get a rendered figure."""))

CELLS.append(code(r'''# ============================================================
# 8. RUN BOTH MODELS OVER THE SELECTED CASES
# ============================================================
N_CASES = 4
SHOW_FIGURES = 4
SEED = 42

# Fixed sample independent of GPU availability and model performance.
selected = complete.sort_values("case_id").sample(
    n=min(N_CASES, len(complete)), random_state=SEED
).sort_values("case_id").reset_index(drop=True)
assert len(selected) > 0
selected[["case_id", "cohort"]].to_csv("/kaggle/working/selected_cases.csv", index=False)
print(f"Evaluating {len(selected)} cases", flush=True)
print(selected[["case_id", "cohort"]].to_string(index=False))

results = []
pred_cache = {}

for i, row in selected.iterrows():
    print(f"[{i+1}/{len(selected)}] {row['case_id']}  ({row['cohort']})")
    t_case = time.time()
    image, gt, seg_raw = load_case(row)
    spacing = tuple(float(x) for x in nib.load(row["label"]).header.get_zooms()[:3])
    if i == 0:
        describe_labels(seg_raw)
    print(f"    loaded {image.shape} in {time.time()-t_case:.1f}s")

    rec = {"case_id": row["case_id"], "cohort": row["cohort"]}
    for name, model in MODEL.items():
        t0 = time.time()
        probs = predict_full_volume(model, image)
        dt = time.time() - t0
        m = compute_all_metrics(probs > 0.5, gt.astype(np.float32), voxel_spacing=spacing)
        rec.update({f"{k}_{name}": v for k, v in m.items()})
        rec[f"time_{name}_s"] = dt
        if i < SHOW_FIGURES:
            pred_cache.setdefault(row["case_id"], {})[name] = probs > 0.5
    if i < SHOW_FIGURES:
        pred_cache[row["case_id"]]["__image__"] = image
        pred_cache[row["case_id"]]["__gt__"] = gt
    results.append(rec)
    pd.DataFrame(results).to_csv("/kaggle/working/progress_results.csv", index=False)
    for name in MODEL:
        print(f"    {name:6s} dice_mean={rec[f'dice_mean_{name}']:.4f}  "
              f"(ET {rec[f'dice_ET_{name}']:.4f} / TC {rec[f'dice_TC_{name}']:.4f} / "
          f"WT {rec[f'dice_WT_{name}']:.4f})  {rec[f'time_{name}_s']:.1f}s", flush=True)

RESULTS = pd.DataFrame(results)
print("\nDone.")
'''))

# ---------------------------------------------------------------- 10. tables
CELLS.append(md("""## 8. Results

Per-case and aggregate Dice / HD95 for both pre-trained models."""))

CELLS.append(code(r'''# ============================================================
# 9. RESULTS TABLES
# ============================================================
pd.set_option("display.width", 200)

METRIC_KEYS = ["dice_ET", "dice_TC", "dice_WT", "dice_mean",
               "hd95_ET", "hd95_TC", "hd95_WT", "hd95_mean"]

print("=" * 100)
print("PER-CASE RESULTS (Dice / HD95 per model)")
print("=" * 100)
per_case_cols = ["case_id", "cohort"] + [f"{k}_{m}" for k in ["dice_mean", "hd95_mean"] for m in MODEL]
print(RESULTS[per_case_cols].round(4).to_string(index=False))

# Full per-case, per-region table, one row per (case, model)
long_rows = []
for _, r in RESULTS.iterrows():
    for m in MODEL:
        long_rows.append({"case_id": r["case_id"], "cohort": r["cohort"], "model": m,
                         **{k: r[f"{k}_{m}"] for k in METRIC_KEYS},
                         "inference_s": r[f"time_{m}_s"]})
LONG = pd.DataFrame(long_rows)

print()
print("=" * 100)
print("MODEL COMPARISON — means over the evaluated cases")
print("=" * 100)
summary = LONG.groupby("model")[[f"{k}" for k in METRIC_KEYS] + ["inference_s"]].mean()
print(summary.round(4).to_string())
from IPython.display import display
display(LONG.round(4))
display(summary.round(4))
LONG.groupby(["cohort", "model"])[METRIC_KEYS].mean().to_csv(
    "/kaggle/working/cohort_summary.csv")

LONG.to_csv("/kaggle/working/brats_africa_results.csv", index=False)
summary.round(4).to_csv("/kaggle/working/brats_africa_summary.csv")
run_metadata = {
    "device": str(DEVICE), "torch": torch.__version__, "python": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    "seed": SEED, "cases": selected["case_id"].tolist(), "training": False,
    "threshold": 0.5, "roi": [128,128,128], "overlap": 0.25,
    "label_mapping": "3 remapped to 4; ET=4, TC=1+4, WT=1+2+4 (repository convention)",
    "limitations": "Small illustrative sample; training-set overlap not independently excluded. Label semantics follow repository convention and require dataset documentation confirmation. Empty-mask HD95 uses repository fixed penalty.",
    "weights": {n: {"sha256": META[n]["sha256"],
                     "parameters": sum(p.numel() for p in MODEL[n].parameters())} for n in MODEL},
}
Path("/kaggle/working/run_metadata.json").write_text(json.dumps(run_metadata, indent=2))
print("\nSaved: /kaggle/working/brats_africa_results.csv (per case x model)")
print("Saved: /kaggle/working/brats_africa_summary.csv (model means)")
'''))

# ---------------------------------------------------------------- 11. figures
CELLS.append(code(r'''# ============================================================
# 10. FIGURES FOR THE SELECTED CASES
# ============================================================
import os
os.makedirs("/kaggle/working/figures", exist_ok=True)

for idx, (case_id, payload) in enumerate(pred_cache.items()):
    preds = {k: v for k, v in payload.items() if not k.startswith("__")}
    image = payload["__image__"]
    gt = payload["__gt__"]
    dice_txt = " | ".join(
        f"{n}: Dice {LONG[(LONG['case_id'] == case_id) & (LONG['model'] == n)]['dice_mean'].iloc[0]:.4f}"
        for n in preds
    )
    show_case(image, gt, preds,
              title=f"{case_id} — {dice_txt}",
              save_path=f"/kaggle/working/figures/{case_id}.png")
    print(f"figure saved: /kaggle/working/figures/{case_id}.png")

    # Four input modalities on the same tumor-containing slice.
    z = best_slice_index(gt)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for channel, (ax, name) in enumerate(zip(axes, ["T1", "T1ce", "T2", "FLAIR"])):
        ax.imshow(to_gray(image[channel, :, :, z]), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{case_id}: input MRI modalities, slice {z}")
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/figures/{case_id}_modalities.png", dpi=110)
    plt.show()
    plt.close(fig)

    # Animated axial traversal of the 3D volume (not a surface rendering).
    from PIL import Image, ImageDraw
    from IPython.display import Image as DisplayImage, display
    frames = []
    for z in range(0, image.shape[-1], 3):
        base = image[1, :, :, z]
        tiles = [("Ground truth", gt)] + list(preds.items())
        arrays = [(overlay(base, masks[:, :, :, z]) * 255).astype(np.uint8)
                  for _, masks in tiles]
        canvas = Image.new("RGB", (sum(a.shape[1] for a in arrays), arrays[0].shape[0]+32))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for (name, _), array in zip(tiles, arrays):
            canvas.paste(Image.fromarray(array), (x, 32))
            draw.text((x+4, 8), f"{name} | slice {z}", fill="white")
            x += array.shape[1]
        frames.append(canvas)
    gif_path = f"/kaggle/working/figures/{case_id}.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    display(DisplayImage(filename=gif_path))
    del frames
'''))

# ---------------------------------------------------------------- 12. bar chart
CELLS.append(code(r'''# ============================================================
# 11. SUMMARY CHART — Dice per region, both models
# ============================================================
import matplotlib.pyplot as plt

models = list(summary.index)
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(models))
width = 0.26
for j, region in enumerate(["ET", "TC", "WT"]):
    vals = [summary.loc[m, f"dice_{region}"] for m in models]
    bars = ax.bar(x + (j - 1) * width, vals, width, label=f"Dice {region}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Dice")
ax.set_ylim(0, 1.05)
ax.set_title(f"BraTS-Africa — pre-trained models, mean Dice over {len(LONG)//len(models)} cases")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/kaggle/working/dice_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
'''))

# ---------------------------------------------------------------- 13. notes
CELLS.append(md("""## 9. Notes

* **No training anywhere in this notebook** — only forward passes through the
  pre-trained checkpoints released with the
  [Brain_Tumor_Benchmark](https://github.com/MohanadMahran/Brain_Tumor_Benchmark) repo.
* **Pre-trained models**
  * 3D U-Net — [MohanadKombar/brain-tumor-unet3d-brats](https://huggingface.co/MohanadKombar/brain-tumor-unet3d-brats)
  * UNETR — [MohanadKombar/brain-tumor-unetr-brats](https://huggingface.co/MohanadKombar/brain-tumor-unetr-brats)
* **Regions** — ET = label 4, TC = labels 1+4, WT = labels 1+2+4; label 3 is
  automatically remapped to 4.
* **Inference protocol** — MONAI sliding window when available, otherwise the
  tested PyTorch fallback; 128^3 ROI, 25% overlap, Gaussian blending, AMP on GPU.
* **Dataset** — [yakunokamizen/brats-africa](https://www.kaggle.com/datasets/yakunokamizen/brats-africa)
  (BraTS-Africa, 146 cases with segmentations). This is a small real-data
  evaluation sample. Training-set overlap has not been independently excluded;
  these are illustrative sample results, not a reproduced independent benchmark.
  The author's validation scores (UNet3D 0.8694, UNETR 0.8362) are not our results.
* **Label semantics** follow the repository convention (3 mapped to 4).
  Observing numerical label values alone does not verify their biological meaning.
  Confirm this convention against the dataset annotation documentation before
  presenting ET/TC-specific results as validated.
* **HD95** uses NIfTI voxel spacing for nonempty masks. One-sided empty masks
  retain the repository's fixed diagonal penalty; both empty gives Dice=1, HD95=0.
* **Outputs written to `/kaggle/working`** — per-case CSV, summary CSV, and PNG
  figures for the visual comparison."""))

CELLS.append(md("""## 10. Export and completion check

The HTML report combines the measured tables, patient images and animations.
Only a completed run writes `SUCCESS.json`. Checkpoint validation scores are
not used as results in this report. The four patients are a demonstration
sample, not evidence of clinical validity or a reproduction of the author's
UPenn-GBM benchmark."""))
CELLS.append(code(r'''import html
from IPython.display import FileLink, display
output = Path("/kaggle/working")
assert len(LONG) == len(selected) * 2
assert np.isfinite(LONG[METRIC_KEYS].to_numpy()).all()
assert LONG[["dice_ET", "dice_TC", "dice_WT"]].ge(0).all().all()
assert LONG[["dice_ET", "dice_TC", "dice_WT"]].le(1).all().all()
parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Pretrained brain tumor segmentation</title>",
         "<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:20px;background:#f6f8fb;color:#172033}img{max-width:100%;background:white}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccd3df}h1,h2{color:#183d65}</style></head><body>",
         "<h1>Pretrained UNet3D vs UNETR on real BraTS-Africa MRI</h1>",
         f"<p>{len(selected)} seeded sample cases; no training or fine-tuning. Device: {DEVICE}. Both models use the same patients, preprocessing and threshold.</p>",
         "<h2>Measured results</h2>", summary.round(4).to_html(),
         "<p>Dice: higher is better. HD95: lower is better, in millimetres. Scores are measured on these cases, not copied from the model author's validation results.</p>",
         "<img src='dice_comparison.png' alt='Dice comparison'>",
         "<h2>Per-patient results</h2>", LONG.round(4).to_html(index=False)]
for case_id in pred_cache:
    parts += [f"<h2>{html.escape(case_id)}</h2>",
              f"<img src='figures/{case_id}_modalities.png' alt='Four MRI modalities'>",
              f"<img src='figures/{case_id}.png' alt='Ground truth and model overlays'>",
              f"<img src='figures/{case_id}.gif' alt='Animated axial slice comparison'>"]
parts += ["<h2>Scope and limitations</h2><p>Small seeded sample; not a clinical diagnostic tool. No independent training-data overlap audit was performed. Labels follow the repository mapping of 3 to 4 for enhancing tumor. One-sided empty masks receive the repository fixed HD95 penalty. Animated images traverse slices of a 3D scan; they are not surface reconstructions.</p>",
          "<h2>Sources</h2><p><a href='https://github.com/MohanadMahran/Brain_Tumor_Benchmark'>Original implementation</a> | <a href='https://www.kaggle.com/datasets/yakunokamizen/brats-africa'>Dataset</a> | <a href='https://pubs.rsna.org/doi/10.1148/ryai.240528'>BraTS-Africa dataset paper</a></p></body></html>"]
(output / "report.html").write_text("\n".join(parts), encoding="utf-8")
required = ["brats_africa_results.csv", "brats_africa_summary.csv", "selected_cases.csv",
            "cohort_summary.csv", "run_metadata.json", "report.html", "dice_comparison.png"]
required += [f"figures/{cid}{suffix}" for cid in pred_cache for suffix in [".png", ".gif", "_modalities.png"]]
assert all((output / name).is_file() and (output / name).stat().st_size > 0 for name in required)
(output / "SUCCESS.json").write_text(json.dumps({"cases": len(selected), "model_case_evaluations": len(LONG), "artifacts": required}, indent=2))
print("END-TO-END COMPLETE: real-case metrics, images, animations and report verified.")
display(FileLink(str(output / "report.html")))
'''))

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for cell_index, cell in enumerate(nb["cells"]):
    cell["id"] = f"brats-{cell_index:02d}"

out = OUT_DIR / "brain-tumor-seg-brats-africa.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out}  ({len(CELLS)} cells)")

metadata = {
    "id": "zekrock/brain-tumor-seg-brats-africa",
    "title": "brain-tumor-seg-brats-africa",
    "code_file": out.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": False,
    "enable_gpu": True,
    "machine_shape": "NvidiaTeslaT4",
    "enable_internet": True,
    "docker_image_pinning_type": "latest",
    "dataset_sources": [
        "zekrock/brats-weights",
        "yakunokamizen/brats-africa",
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Wrote {OUT_DIR / 'kernel-metadata.json'}")
