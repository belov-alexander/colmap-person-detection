
# COLMAP Person Detection

Python tool that uses the GroundingDINO model to automatically detect and censor people in images. This project is specifically designed for privacy protection in COLMAP photogrammetry workflows, 3D reconstruction, and general image anonymization tasks.

## Overview

This tool leverages the state-of-the-art GroundingDINO object detection model to:

- Detect people in images with high accuracy
- Generate binary masks for COLMAP and other 3D reconstruction tools
- Optionally blur license plates for additional privacy protection
- Process entire directories of images in batch mode

The tool creates three outputs for each processed image:

1. **Censored Image**: Original image with black rectangles over detected people
2. **Binary Mask**: PNG mask file (black=person, white=background) for use with COLMAP
3. **Enhanced Masks**: Supports loading and enhancing existing masks with new detections

## Features

- **Automatic Person Detection**: Uses GroundingDINO to find people in images with high accuracy
- **Privacy Censorship**: Automatically obscures detected people with black boxes
- **Mask Generation**: Creates binary masks compatible with COLMAP and other 3D reconstruction software
- **Mask Enhancement**: Loads existing masks and adds new person detections to them
- **Optional License Plate Blurring**: Detects and blurs license plates when enabled
- **Batch Processing**: Process entire directories of images automatically
- **Cross-Platform Support**: Works on both CPU and GPU (CUDA)
- **Configurable Thresholds**: Adjust detection sensitivity for different use cases

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for faster processing)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) model weights

## Installation

### 1. Clone or Download the Project

```bash
cd /path/to/your/projects
```

### 2. Create a Virtual Environment

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

Ensure you have the GroundingDINO weights and config files:

- **Config**: `groundingdino/config/GroundingDINO_SwinT_OGC.py`
- **Weights**: `weights/groundingdino_swint_ogc.pth`

You can download the weights from the [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO).

## Usage

### Basic Usage

Process all images in the `input` directory:

```bash
python colmap-person-detection.py
```

This will:

- Detect and censor people in all images from the `input` directory
- Save processed images to the `output` directory
- Generate binary masks in the `masks` directory

### With License Plate Blurring

To also blur license plates on the output images:

```bash
python colmap-person-detection.py -c
```

### Custom Directories

Specify custom input/output directories:

```bash
python colmap-person-detection.py -i /path/to/input -o /path/to/output -m /path/to/masks
```

### All Options

View all available options:

```bash
python colmap-person-detection.py --help
```

Available options:

- `-i, --input`: Path to input directory (default: `input`)
- `-o, --output`: Path to output directory (default: `output`)
- `-m, --masks`: Path to masks directory (default: `masks`)
- `-c, --carplateblur`: Enable license plate blurring on output images

## How It Works

1. **Image Loading**: The script reads all images from the input directory
2. **Person Detection**: GroundingDINO model detects people using the text prompt "person"
3. **Mask Generation/Enhancement**:
   - If a mask already exists, it loads and enhances it with new detections
   - If no mask exists, creates a new binary mask (black=person, white=background)
4. **Image Censoring**: Draws black rectangles over detected people in the output image
5. **License Plate Detection** (optional): Detects and blurs license plates if enabled
6. **Output Saving**: Saves both the censored image and the mask file

## Configuration

You can modify detection parameters in `colmap-person-detection.py`:

### Person Detection Settings

```python
TEXT_PROMPT = "person"           # Object to detect
BOX_THRESHOLD = 0.35            # Confidence threshold for bounding boxes
TEXT_THRESHOLD = 0.25           # Confidence threshold for text matching
```

### License Plate Detection Settings

```python
LICENSE_PLATE_PROMPTS = "license plate"
LICENSE_PLATE_BOX_THRESHOLD = 0.55
LICENSE_PLATE_TEXT_THRESHOLD = 0.45
PAD_RATIO = 0.02                # Padding ratio for bounding boxes
```

## Use Cases

### COLMAP Photogrammetry

Use the generated masks to exclude people from 3D reconstruction:

1. Place your images in the `input` directory
2. Run the script to generate masks
3. Use the `masks` directory with COLMAP's masking feature

### Privacy Protection

Automatically anonymize images containing people before sharing or publishing.

### Dataset Preparation

Create clean datasets by removing people from background scenes.

## Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory issues:

- The script will automatically fall back to CPU
- Consider processing images in smaller batches
- Reduce image resolution before processing

### No Detections

If people are not being detected:

- Lower the `BOX_THRESHOLD` value (try 0.25)
- Ensure images are clear and people are visible
- Check that model weights are properly loaded

## Project Structure

```text
colmap-person-detection/
├── colmap-person-detection.py          # Main script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── groundingdino/           # GroundingDINO config
│   └── config/
│       └── GroundingDINO_SwinT_OGC.py
├── weights/                 # Model weights
│   └── groundingdino_swint_ogc.pth
├── input/                   # Place input images here
├── output/                  # Censored images output
└── masks/                   # Binary masks output
```

## Dependencies

- `numpy` - Numerical operations
- `opencv-python` - Image processing
- `Pillow` - Image I/O
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `transformers` - Hugging Face transformers
- `timm` - PyTorch Image Models
- `supervision` - Computer vision utilities
- `groundingdino-py` - GroundingDINO model

## License

This project uses the GroundingDINO model. Please refer to the [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO) for licensing information.

## Acknowledgments

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Object detection model
- IDEA Research - For developing the GroundingDINO model

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

# colmap-person-detection
