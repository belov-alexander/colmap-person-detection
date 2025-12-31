import argparse
import os
import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional

# Grounding DINO imports
from groundingdino.util.inference import load_model, load_image, predict

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
# Paths to model config and weights
CONFIG_PATH = os.path.join("groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")

# Detection settings
TEXT_PROMPT = "person"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# License plate detection settings
LICENSE_PLATE_PROMPTS = "license plate"
LICENSE_PLATE_BOX_THRESHOLD = 0.55
LICENSE_PLATE_TEXT_THRESHOLD = 0.45
PAD_RATIO = 0.02  # Expand bbox by 2% on each side to cover frames/bolts


def detect_objects(
    model, 
    image_tensor: torch.Tensor, 
    caption: str, 
    box_thm: float, 
    text_thm: float, 
    device: str
) -> Optional[torch.Tensor]:
    """
    Runs the GroundingDINO model to detect objects matching the caption.
    
    Args:
        model: Loaded GroundingDINO model.
        image_tensor: Preprocessed image tensor.
        caption: Text prompt for detection (e.g., "person").
        box_thm: Threshold for bounding box confidence.
        text_thm: Threshold for text confidence.
        device: Device to run inference on ('cpu' or 'cuda').

    Returns:
        boxes: Tensor of bounding boxes in normalized format [cx, cy, w, h],
               or None if no detections found.
    """
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=caption,
        box_threshold=box_thm,
        text_threshold=text_thm,
        device=device,
    )
    return boxes


def create_mask(
    image_shape: Tuple[int, int, int], 
    boxes: torch.Tensor, 
    output_path: str
) -> None:
    """
    Generates a binary mask where detected objects are black (0) and background is white (255).
    If a mask already exists at output_path, it loads the existing mask and adds new person
    detection boxes to it, preserving any manually created masks.
    Saved as a PNG file.

    Args:
        image_shape: Shape of the original image (H, W, C).
        boxes: Tensor of normalized bounding boxes [cx, cy, w, h].
        output_path: Path where the mask will be saved.
    """
    h, w, _ = image_shape
    
    # Check if mask already exists
    if os.path.exists(output_path):
        # Load existing mask
        mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load existing mask at {output_path}. Creating new mask.")
            mask = np.full((h, w), 255, dtype=np.uint8)
        else:
            print(f"Loaded existing mask from: {output_path}")
            # Ensure mask has correct dimensions
            if mask.shape != (h, w):
                print(f"Warning: Existing mask dimensions {mask.shape} don't match image dimensions ({h}, {w}). Creating new mask.")
                mask = np.full((h, w), 255, dtype=np.uint8)
    else:
        # Create white mask (255) single channel
        mask = np.full((h, w), 255, dtype=np.uint8)

    if boxes is not None and len(boxes) > 0:
        # Convert normalized boxes [cx, cy, w, h] to pixel coordinates
        boxes_px = (boxes * torch.tensor([w, h, w, h])).cpu().numpy()
        
        print(f"Adding {len(boxes_px)} person detection(s) to mask")
        
        for (cx, cy, bw, bh) in boxes_px:
            # Calculate top-left (x1, y1) and bottom-right (x2, y2) coordinates
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            # Add a small padding (8% of the largest dimension) to ensure full coverage
            # Matching the logic in censor_image
            pad = int(0.08 * max(bw, bh))
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)

            # Draw a filled black rectangle over the identified region (0)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)

    # Save the mask
    cv2.imwrite(output_path, mask)
    print(f"Saved mask to: {output_path}")


def censor_image(
    image_source: np.ndarray, 
    boxes: torch.Tensor, 
    output_path: str
) -> None:
    """
    Draws black rectangles over declared bounding boxes on the image and saves it.

    Args:
        image_source: Original image as a NumPy array (RGB).
        boxes: Tensor of normalized bounding boxes [cx, cy, w, h].
        output_path: Path where the censored image will be saved.
    """
    # Convert RGB image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

    if boxes is None or len(boxes) == 0:
        print(f"No objects detected. Saving original to {output_path}")
        cv2.imwrite(output_path, image_bgr)
        return

    h, w, _ = image_source.shape
    
    # Convert normalized boxes [cx, cy, w, h] to pixel coordinates
    # We multiply by [w, h, w, h] to scale from 0-1 range to pixel dimensions
    boxes_px = (boxes * torch.tensor([w, h, w, h])).cpu().numpy()

    print(f"Found {len(boxes_px)} objects. Censoring...")

    # Iterate over each detected box
    for (cx, cy, bw, bh) in boxes_px:
        # Calculate top-left (x1, y1) and bottom-right (x2, y2) coordinates
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        # Add a small padding (8% of the largest dimension) to ensure full coverage
        pad = int(0.08 * max(bw, bh))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)

        # Draw a filled black rectangle over the identified region
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    # Save the processed image
    cv2.imwrite(output_path, image_bgr)
    print(f"Saved censored image to: {output_path}")


def blur_license_plates(
    image_bgr: np.ndarray,
    model,
    image_tensor: torch.Tensor,
    device: str
) -> np.ndarray:
    """
    Detects and blurs license plates in the image.
    
    Args:
        image_bgr: Image in BGR format (OpenCV format).
        model: Loaded GroundingDINO model.
        image_tensor: Preprocessed image tensor for detection.
        device: Device to run inference on ('cpu' or 'cuda').
    
    Returns:
        Image with blurred license plates.
    """
    # Detect license plates
    boxes = detect_objects(
        model,
        image_tensor,
        LICENSE_PLATE_PROMPTS,
        LICENSE_PLATE_BOX_THRESHOLD,
        LICENSE_PLATE_TEXT_THRESHOLD,
        device
    )
    
    if boxes is None or len(boxes) == 0:
        print("  No license plates detected.")
        return image_bgr
    
    h, w, _ = image_bgr.shape
    boxes_px = (boxes * torch.tensor([w, h, w, h])).cpu().numpy()
    
    print(f"  Found {len(boxes_px)} license plate(s). Blurring...")
    
    # Blur each detected license plate
    for (cx, cy, bw, bh) in boxes_px:
        # Calculate top-left (x1, y1) and bottom-right (x2, y2) coordinates
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        
        # Expand bbox by PAD_RATIO to cover frames/bolts
        pad_w = int(bw * PAD_RATIO)
        pad_h = int(bh * PAD_RATIO)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w - 1, x2 + pad_w)
        y2 = min(h - 1, y2 + pad_h)
        
        # Extract the region of interest
        roi = image_bgr[y1:y2, x1:x2]
        
        # Apply Gaussian blur (kernel size should be odd)
        ksize = max(3, int(min(bw, bh) * 0.6))
        if ksize % 2 == 0:
            ksize += 1
        blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        
        # Replace the region with blurred version
        image_bgr[y1:y2, x1:x2] = blurred_roi
    
    return image_bgr


def main():
    """
    Main execution function.
    """
    # 0. Parse arguments
    parser = argparse.ArgumentParser(description="COLMAP Person Detection using GroundingDINO")
    parser.add_argument("-i", "--input", type=str, default="input", help="Path to input directory containing images")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory for processed images")
    parser.add_argument("-m", "--masks", type=str, default="masks", help="Path to output directory for masks")
    parser.add_argument("-c", "--carplateblur", action="store_true", help="Blur license plates on output images")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    masks_dir = args.masks

    # 1. Device detection
    # Check if CUDA (GPU) is available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 2. Check for existence of necessary files/directories
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create masks directory if it doesn't exist
    os.makedirs(masks_dir, exist_ok=True)

    # 3. Load Model (once)
    print("Loading model...")
    try:
        model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. Process all images in input directory
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process.")

    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Define mask filename: original filename + .png (e.g., img.jpg -> img.jpg.png)
        mask_filename = filename + ".png"
        mask_path = os.path.join(masks_dir, mask_filename)

        print(f"\nProcessing: {filename}")
        
        try:
            image_source, image_tensor = load_image(input_path)
            
            # Detect
            boxes = detect_objects(
                model, 
                image_tensor, 
                TEXT_PROMPT, 
                BOX_THRESHOLD, 
                TEXT_THRESHOLD, 
                device
            )
            
            # Censor and Save
            censor_image(image_source, boxes, output_path)
            
            # Blur license plates if flag is enabled
            if args.carplateblur:
                # Read the censored image
                censored_bgr = cv2.imread(output_path)
                # Reload image tensor for license plate detection
                _, image_tensor_for_plates = load_image(output_path)
                # Apply license plate blurring
                blurred_bgr = blur_license_plates(censored_bgr, model, image_tensor_for_plates, device)
                # Save the final result
                cv2.imwrite(output_path, blurred_bgr)
            
            # Create and Save Mask
            create_mask(image_source.shape, boxes, mask_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
