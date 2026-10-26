# Rice Grain Analyzer

AI-powered rice grain counting and variety classification using YOLOv11 segmentation and ResNet50 classification.

![Rice Analysis Demo](scripts/inference_result.jpg)

## Features

- **Accurate Counting**: Handles dense piles of rice using tiled inference with global NMS
- **8 Variety Classification**: Arborio, Basmati, Ipsala, Jasmine, Karacadag, Jhili, HMT (Sona Masuri), Masuri
- **Robust Preprocessing**: Automatic background masking + white balance for color-tinted grains
- **Interactive UI**: Gradio-based web interface with adjustable parameters

## Project Structure

```
rice/
├── app.py                # Main Gradio application
├── models/
│   ├── best.pt           # YOLOv11-Seg model (grain detection)
│   └── rice_resnet50_best.pth  # ResNet50 classifier (variety)
├── scripts/
│   └── inference.py      # CLI inference script
├── notebooks/
│   ├── Rice_Grain_Counter_Colab.ipynb  # YOLO training notebook
│   └── rice_classifier.ipynb          # ResNet50 training notebook
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision ultralytics gradio opencv-python pillow numpy
```

### 2. Download Models

Place the following files in the `models/` directory:
- `best.pt` - YOLOv11 segmentation model
- `rice_resnet50_best.pth` - ResNet50 classifier

## Usage

### Web Interface (Recommended)

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

### Command Line

```bash
python scripts/inference.py --source path/to/image.jpg --output result.jpg
```

## How It Works

### Two-Stage Pipeline

1. **Stage 1 - Detection (YOLOv11-Seg)**
   - Image is split into overlapping tiles (640x640 default)
   - YOLO detects individual grains and generates segmentation masks
   - Global NMS merges detections across tiles

2. **Stage 2 - Classification (ResNet50)**
   - Each detected grain is cropped using its mask (black background)
   - White balance is applied to normalize color variations
   - ResNet50 classifies the grain into one of 8 varieties

### Preprocessing

- **Background Masking**: Uses YOLO's segmentation mask to isolate grains
- **White Balance**: Corrects for lighting/color tints using Gray World algorithm

## Training

### YOLO Model
See `Rice_Grain_Counter_Colab.ipynb` for training on a segmentation dataset.

### Classifier Model
See `rice_classifier.ipynb` for training on the Milled Rice Dataset (1.12M images, 8 classes).

## Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| YOLOv11-Seg | mAP@50 | >95% |
| ResNet50 Classifier | Val Accuracy | 99.62% |

## License

MIT License
