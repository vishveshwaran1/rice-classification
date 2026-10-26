import gradio as gr
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

from advice import assess_quality, generate_consumption_advice

# --- CONFIG ---
# Models are stored in the models/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "models", "best.pt")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "rice_resnet50_best.pth")

VARIETY_CLASSES = [
    "Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag", 
    "Jhili", "HMT (Sona Masuri)", "Masuri"
]

# --- MODEL CLASSES (Re-implemented for standalone safety) ---
class VarietyClassifier:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Classifier from {model_path} to {self.device}...")
        
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(VARIETY_CLASSES))
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"⚠️ Classifier model not found at {model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def classify_batch(self, crop_images):
        if not crop_images: return []
        tensors = []
        for img in crop_images:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensors.append(self.transform(pil_img))
        
        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            _, preds = torch.max(outputs, 1)
        return [VARIETY_CLASSES[idx] for idx in preds.cpu().numpy()]

# --- INITIALIZATION ---
print("Initializing Models...")
try:
    print(f"Loading YOLO from {YOLO_PATH}")
    yolo_model = YOLO(YOLO_PATH)
    
    print(f"Loading Classifier from {CLASSIFIER_PATH}")
    classifier_model = VarietyClassifier(CLASSIFIER_PATH) if os.path.exists(CLASSIFIER_PATH) else None
except Exception as e:
    print(f"Error loading models: {e}")
    yolo_model = None
    classifier_model = None

# --- PIPELINE LOGIC ---
def get_slices(img_h, img_w, slice_size=640, overlap_ratio=0.2):
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

def analyze_rice(image, conf_thresh, slice_size):
    """
    Main function called by Gradio.
    Args:
        image: Numpy array (RGB) from Gradio
        conf_thresh: float
        slice_size: int
    Returns:
        Annotated Image (Numpy RGB) — only the majority variety's boxes drawn,
        JSON Counts (kept for backend/logging use, not shown in UI)
    """
    if yolo_model is None:
        return image, "Model failed to load", "", {"Error": "Model failed to load"}

    # Gradio passes RGB, OpenCV needs BGR for internal processing logic usually,
    # but let's stick to BGR for consistency with my previous script logic
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_bgr.shape[:2]
    
    slices = get_slices(img_h, img_w, slice_size=int(slice_size))
    
    all_boxes = []
    all_scores = []
    all_masks = []
    
    # 1. Detection
    for (x1, y1, x2, y2) in slices:
        chip = img_bgr[y1:y2, x1:x2]
        results = yolo_model.predict(chip, conf=conf_thresh, verbose=False)
        result = results[0]
        
        if len(result.boxes) == 0: continue
        
        chip_masks = []
        if result.masks is not None:
             for poly in result.masks.xy:
                if len(poly) > 0:
                    gp = poly.copy()
                    gp[:, 0] += x1
                    gp[:, 1] += y1
                    chip_masks.append(gp)
                else: chip_masks.append(None)
        else: chip_masks = [None] * len(result.boxes)
        
        for i, box in enumerate(result.boxes):
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            all_boxes.append([bx1+x1, by1+y1, bx2+x1, by2+y1])
            all_scores.append(float(box.conf[0].cpu().numpy()))
            all_masks.append(chip_masks[i])
            
    # 2. NMS
    cv2_boxes = [[int(x1), int(y1), int(x2-x1), int(y2-y1)] for x1, y1, x2, y2 in all_boxes]
    indices = cv2.dnn.NMSBoxes(cv2_boxes, all_scores, conf_thresh, 0.45)
    
    output_img = img_bgr.copy()
    overlay = output_img.copy()
    
    class_colors = {
        "Arborio": (255, 0, 0), "Basmati": (0, 255, 0), "Ipsala": (0, 0, 255),
        "Jasmine": (255, 255, 0), "Karacadag": (255, 0, 255), "Jhili": (0, 165, 255),
        "HMT (Sona Masuri)": (128, 0, 128), "Masuri": (0, 128, 128), "Detected": (0, 255, 0)
    }
    
    counts = {}
    grain_records = []  # (box_idx, variety) for every surviving grain, used only in backend
    majority_variety = None

    if len(indices) > 0:
        valid_indices = indices.flatten()
        crops = []
        valid_box_indices = []
        
        for i in valid_indices:
            box = all_boxes[i]
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            # Extract raw crop
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # --- APPLY MASKING FOR CLASSIFIER ---
            # The classifier expects a black background.
            # We use the segmentation mask found by YOLO to mask out the background.
            
            mask_poly = all_masks[i]
            
            if mask_poly is not None and len(mask_poly) > 0:
                # 1. Create a black mask of the same size as the crop
                h_crop, w_crop = crop.shape[:2]
                mask_img = np.zeros((h_crop, w_crop), dtype=np.uint8)
                
                # 2. Shift global polygon to local crop coordinates
                local_poly = mask_poly.copy()
                local_poly[:, 0] -= x1
                local_poly[:, 1] -= y1
                
                # 3. Fill the polygon area with white (255)
                cv2.fillPoly(mask_img, [np.int32(local_poly)], 255)
                
                # 4. Apply mask to crop (keep only the grain pixels)
                masked_crop = cv2.bitwise_and(crop, crop, mask=mask_img)
                
                # --- COLOR CORRECTION (White Balance) ---
                b, g, r = cv2.split(masked_crop)
                
                non_zero = (mask_img > 0)
                if np.count_nonzero(non_zero) > 0:
                    b_avg = np.mean(b[non_zero])
                    g_avg = np.mean(g[non_zero])
                    r_avg = np.mean(r[non_zero])
                    
                    b_avg = b_avg if b_avg > 0 else 128
                    g_avg = g_avg if g_avg > 0 else 128
                    r_avg = r_avg if r_avg > 0 else 128
                    
                    gray_avg = (b_avg + g_avg + r_avg) / 3
                    
                    b_scale = gray_avg / b_avg
                    g_scale = gray_avg / g_avg
                    r_scale = gray_avg / r_avg
                    
                    b = cv2.multiply(b, b_scale).astype(np.uint8)
                    g = cv2.multiply(g, g_scale).astype(np.uint8)
                    r = cv2.multiply(r, r_scale).astype(np.uint8)
                    
                    balanced_crop = cv2.merge([b, g, r])
                    balanced_crop = cv2.bitwise_and(balanced_crop, balanced_crop, mask=mask_img)
                    
                    crops.append(balanced_crop)
                else:
                    crops.append(masked_crop)
            else:
                # Fallback: if no mask (rare), use raw square crop
                crops.append(crop)

            valid_box_indices.append(i)
            
        # Classify every surviving grain (backend-only step — feeds counts + majority pick)
        varieties = classifier_model.classify_batch(crops) if classifier_model else ["Detected"] * len(crops)
        
        for idx, variety in zip(valid_box_indices, varieties):
            counts[variety] = counts.get(variety, 0) + 1
            grain_records.append((idx, variety))

        # --- Determine majority variety (backend logic) ---
        majority_variety = max(counts, key=counts.get) if counts else None

        # --- Visualize ONLY grains belonging to the majority variety ---
        for idx, variety in grain_records:
            if variety != majority_variety:
                continue

            mask = all_masks[idx]
            color = class_colors.get(variety, (200, 200, 200))  # BGR Color

            if mask is not None:
                cv2.fillPoly(overlay, [np.int32(mask)], color)

            x1, y1, x2, y2 = map(int, all_boxes[idx])
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            label = f"{variety}"
            cv2.putText(output_img, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)
    
    # Full breakdown stays available in backend/logs only — never shown on the frontend
    counts["TOTAL"] = len(grain_records)
    print(f"[backend] Grain counts by variety: {counts}")

    majority_label = f"Majority variety: {majority_variety}" if grain_records else "No grains detected"

    # --- Consumption advice, driven off the same detections (backend logic) ---
    quality_info = assess_quality(majority_variety, grain_records, all_boxes, all_masks)
    advice_text = generate_consumption_advice(majority_variety, quality_info)
    print(f"[backend] Quality assessment: {quality_info}")

    # Convert BGR back to RGB for Gradio
    final_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    return final_rgb, majority_label, advice_text, counts

# --- APP INTERFACE ---
with gr.Blocks(title="Rice Grain Analyzer") as demo:
    gr.Markdown("# 🍚 AI Rice Grain Analyzer")
    gr.Markdown("Upload an image of rice grains (piled or isolated). Only the boxes for the **majority variety** detected are highlighted.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.25, label="Confidence Threshold")
            slice_slider = gr.Slider(minimum=320, maximum=1280, step=32, value=640, label="Tile Size (Zoom)")
            btn = gr.Button("Analyze Rice", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Annotated Result (majority variety only)")
            output_majority = gr.Textbox(label="Majority Variety", interactive=False)
            output_advice = gr.Markdown(label="Consumption Advice")
            # Full per-variety counts are computed and printed server-side but not shown in the UI.
            output_stats = gr.JSON(label="Grain Counts (backend)", visible=False)
            
    btn.click(
        fn=analyze_rice,
        inputs=[input_image, conf_slider, slice_slider],
        outputs=[output_image, output_majority, output_advice, output_stats],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)