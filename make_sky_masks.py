#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("Please install onnxruntime (or onnxruntime-gpu).") from e

def list_images(img_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    return [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_session(model_path: str, use_gpu: bool):
    providers = ["CPUExecutionProvider"]
    if use_gpu:
        # If onnxruntime-gpu is installed, CUDAExecutionProvider is usually available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    return sess

def normalize_image(bgr: np.ndarray):
    # Convert BGR->RGB, float32 [0..1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Common ImageNet normalization (often used by U2Net-like models)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return rgb

def resize_keep_aspect(img: np.ndarray, target: int):
    h, w = img.shape[:2]
    if max(h, w) == target:
        return img, 1.0
    scale = target / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def pad_to_square(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    top = 0
    left = 0
    bottom = size - h
    right = size - w
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return padded, (top, bottom, left, right)

def unpad(mask: np.ndarray, pads):
    top, bottom, left, right = pads
    h, w = mask.shape[:2]
    return mask[top:h-bottom, left:w-right]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Path to input images folder")
    ap.add_argument("--out", required=True, help="Path to output masks folder")
    ap.add_argument("--model", default="", help="Path to skyseg.onnx (optional). If empty, expects you to place it manually.")
    ap.add_argument("--input-size", type=int, default=320, help="Model input size (square). 320 for this ONNX.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Mask threshold in [0..1]")
    ap.add_argument("--invert", action="store_true", help="Invert output mask (swap sky/non-sky)")
    ap.add_argument("--use-gpu", action="store_true", help="Use CUDAExecutionProvider if available (onnxruntime-gpu)")
    args = ap.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise SystemExit(f"Images dir not found: {img_dir}")

    model_path = args.model
    if not model_path:
        raise SystemExit(
            "Please provide --model /path/to/skyseg.onnx\n"
            "Tip: download ONNX from Hugging Face model 'JianyuanWang/skyseg'."
        )

    sess = load_session(model_path, use_gpu=args.use_gpu)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    images = list_images(img_dir)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    for p in tqdm(images, desc="SkySeg"):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        orig_h, orig_w = bgr.shape[:2]

        # Resize keep aspect then pad to square
        resized, scale = resize_keep_aspect(bgr, target=args.input_size)
        padded, pads = pad_to_square(resized, size=args.input_size)

        # Normalize & CHW
        x = normalize_image(padded)  # HWC
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)  # 1x3xHxW

        y = sess.run([output_name], {input_name: x})[0]

        # Output shape may be 1x1xHxW or 1xHxW; handle both
        if y.ndim == 4:
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        else:
            y = y.squeeze()

        # The model output is already a probability distribution (0..1)
        prob = y.astype(np.float32)

        # Unpad, resize back to original
        prob = unpad(prob, pads)
        prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        mask = (prob >= args.threshold).astype(np.uint8) * 255
        if args.invert:
            mask = 255 - mask

        out_path = out_dir / (p.stem + ".png")
        cv2.imwrite(str(out_path), mask)

if __name__ == "__main__":
    main()
