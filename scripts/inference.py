import argparse
import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

# advice.py lives one directory up, alongside app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advice import assess_quality, generate_consumption_advice

# 8 Classes from Mendeley "Milled Rice Grain" Dataset
VARIETY_CLASSES = [
    "Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag", 
    "Jhili", "HMT (Sona Masuri)", "Masuri"
]

class VarietyClassifier:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Classifier from {model_path} to {self.device}...")
        
        # Load ResNet50
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(VARIETY_CLASSES))
        
        # Load weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"⚠️ Classifier model not found at {model_path}. Random weights used for testing.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing (Match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def classify_batch(self, crop_images):
        """
        Classifies a batch of cropped grain images (BGR numpy arrays).
        """
        if not crop_images:
            return []
            
        # Convert to tensors
        tensors = []
        for img in crop_images:
            # BGR to RGB PIL
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensors.append(self.transform(pil_img))
            
        batch_tensor = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            _, preds = torch.max(outputs, 1)
            
        return [VARIETY_CLASSES[idx] for idx in preds.cpu().numpy()]


def get_slices(img_h, img_w, slice_size, overlap_ratio):
    slices = []
    stride = int(slice_size * (1 - overlap_ratio))
    y_min = 0
    while y_min < img_h:
        y_max = min(y_min + slice_size, img_h)
        x_min = 0
        while x_min < img_w:
            x_max = min(x_min + slice_size, img_w)
            slices.append((x_min, y_min, x_max, y_max))
            if x_max >= img_w: break
            x_min += stride
        if y_max >= img_h: break
        y_min += stride
    return slices

def run_pipeline(yolo_model, classifier, img_path, slice_size=640, overlap_ratio=0.2, conf_thresh=0.25):
    print(f"Processing image: {img_path}")
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"Image not found: {img_path}")
    img_h, img_w = img.shape[:2]
    
    # 1. Tiled YOLO Inference
    slices = get_slices(img_h, img_w, slice_size, overlap_ratio)
    all_boxes = []
    all_scores = []
    all_masks = [] # Polygons
    
    count = 0
    for (x1, y1, x2, y2) in slices:
        chip = img[y1:y2, x1:x2]
        results = yolo_model.predict(chip, conf=conf_thresh, verbose=False)
        result = results[0]
        
        if len(result.boxes) == 0: continue
        
        # Extract Masks (relative to chip)
        chip_masks = []
        if result.masks is not None:
            for poly in result.masks.xy:
                if len(poly) > 0:
                    gp = poly.copy()
                    gp[:, 0] += x1
                    gp[:, 1] += y1
                    chip_masks.append(gp)
                else:
                    chip_masks.append(None)
        else:
            chip_masks = [None] * len(result.boxes)

        # Extract Boxes
        for i, box in enumerate(result.boxes):
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            all_boxes.append([bx1+x1, by1+y1, bx2+x1, by2+y1])
            all_scores.append(float(box.conf[0].cpu().numpy()))
            all_masks.append(chip_masks[i])
            
    if not all_boxes:
        print("No grains detected.")
        return img, []

    # 2. Global NMS
    print(f"Running NMS on {len(all_boxes)} candidates...")
    cv2_boxes = [[int(x1), int(y1), int(x2-x1), int(y2-y1)] for x1, y1, x2, y2 in all_boxes]
    indices = cv2.dnn.NMSBoxes(cv2_boxes, all_scores, conf_thresh, 0.45) # 0.45 IoU
    
    final_grains = []
    
    if len(indices) > 0:
        valid_indices = indices.flatten()
        
        # Prepare crops for classification
        crops = []
        valid_box_indices = []
        
        for i in valid_indices:
            box = all_boxes[i]
            x1, y1, x2, y2 = map(int, box)
            
            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            crops.append(crop)
            valid_box_indices.append(i)
            
        # 3. Batch Classification (Stage 2)
        print(f"Classifying {len(crops)} grains...")
        if classifier:
            varieties = classifier.classify_batch(crops)
        else:
            varieties = ["Unknown"] * len(crops)
            
        # Compile final results
        for idx, variety in zip(valid_box_indices, varieties):
            final_grains.append({
                "box": all_boxes[idx],
                "score": all_scores[idx],
                "mask": all_masks[idx],
                "variety": variety
            })
            
    return img, final_grains

def visualize(img, grains, output_path):
    print("Visualizing...")
    # Generate colors for 8 classes
    class_colors = {
        "Arborio": (255, 0, 0),       # Blue
        "Basmati": (0, 255, 0),       # Green
        "Ipsala": (0, 0, 255),        # Red
        "Jasmine": (255, 255, 0),     # Cyan
        "Karacadag": (255, 0, 255),   # Magenta
        "Jhili": (0, 165, 255),       # Orange
        "HMT (Sona Masuri)": (128, 0, 128), # Purple
        "Masuri": (0, 128, 128)       # Olive
    }
    
    overlay = img.copy()
    
    for grain in grains:
        x1, y1, x2, y2 = map(int, grain['box'])
        variety = grain['variety']
        mask = grain['mask']
        
        color = class_colors.get(variety, (200, 200, 200))
        
        # Draw Mask
        if mask is not None:
             cv2.fillPoly(overlay, [np.int32(mask)], color)
             
        # Draw Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
    # Alpha blend
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    
    # Legend
    y_off = 30
    for var, col in class_colors.items():
        cv2.putText(img, var, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        y_off += 25
        
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--yolo", default="best.pt")
    parser.add_argument("--classifier", default="rice_resnet50_best.pth")
    parser.add_argument("--output", default="final_result.jpg")
    parser.add_argument("--slice-size", type=int, default=640)
    args = parser.parse_args()
    
    # Load Models
    try:
        yolo = YOLO(args.yolo)
    except Exception as e:
        print(f"Error loading YOLO: {e}")
        return
        
    classifier = None
    if os.path.exists(args.classifier):
        classifier = VarietyClassifier(args.classifier)
    else:
        print(f"⚠️ Classifier {args.classifier} not found. Running YOLO only (no class labels).")
        # Pseudo-classifier for testing logic without weights
        # classifier = VarietyClassifier(args.classifier) 

    # Run
    try:
        img, grains = run_pipeline(yolo, classifier, args.source, slice_size=args.slice_size)
        
        # Stats
        print("\n=== RESULTS ===")
        print(f"Total Grains: {len(grains)}")
        from collections import Counter
        var_counts = Counter([g['variety'] for g in grains])
        print("Breakdown:")
        for v, c in var_counts.items():
            print(f"  {v}: {c}")
        print("===============\n")

        # --- Consumption advice, driven off the same detections ---
        if grains:
            majority_variety = var_counts.most_common(1)[0][0]
            grain_records = [(i, g['variety']) for i, g in enumerate(grains)]
            all_boxes = [g['box'] for g in grains]
            all_masks = [g['mask'] for g in grains]

            quality_info = assess_quality(majority_variety, grain_records, all_boxes, all_masks)
            advice_text = generate_consumption_advice(majority_variety, quality_info)

            print("=== CONSUMPTION ADVICE ===")
            print(advice_text)
            print("===========================\n")
        
        visualize(img, grains, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
