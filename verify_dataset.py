import zipfile
import os
import sys

def detect_format(zip_path):
    if not os.path.exists(zip_path):
        print(f"❌ File not found: {zip_path}")
        return

    print(f"🔍 Analyzing {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            files = z.namelist()
            
            # 1. Look for COCO JSONs
            json_files = [f for f in files if f.endswith('.json') and 'annotations' in f]
            if json_files:
                print(f"\n✅ COCO FORMAT DETECTED ({len(json_files)} json files found)")
                print("   -> This supports Segmentation naturally.")
                print("   -> Use 'Mask R-CNN' notebook OR 'Universal Loader' (v7) notebook.")
                # We can stop here or continue checking just in case
                return

            # 2. Look for YOLO TXT Labels
            txt_files = [f for f in files if f.endswith('.txt') and 'classes.txt' not in f and 'train' in f and 'labels' in f]
            
            # Fallback if 'labels' folder is not explicit or different structure
            if not txt_files:
                 txt_files = [f for f in files if f.endswith('.txt') and 'classes.txt' not in f and 'train' in f]

            if not txt_files:
                print("\n❌ NO LABELS FOUND. Zip might be empty or structure is wrong.")
                return

            print(f"\n📦 Found {len(txt_files)} label files. Checking content of the first 5...")
            
            cnt_seg = 0
            cnt_box = 0
            
            for i, txt_file in enumerate(txt_files[:5]):
                with z.open(txt_file) as f:
                    # Read first line
                    line = f.readline().decode('utf-8').strip().split()
                    if not line: continue
                    
                    num_values = len(line)
                    print(f"   📄 {os.path.basename(txt_file)} -> {num_values} values")
                    
                    if num_values > 5:
                        cnt_seg += 1
                    elif num_values == 5:
                        cnt_box += 1
            
            print("-" * 30)
            if cnt_seg > 0:
                print("✅ YOLO SEGMENTATION DETECTED")
                print("   -> The files contain Polygons (many points).")
                print("   -> Use 'Direct YOLO' notebook or 'Universal Loader'.")
            elif cnt_box > 0:
                print("⚠️ YOLO DETECTION DETECTED (Boxes only)")
                print("   -> This is NOT segmentation data.")
                print("   -> If you train a Segmentation model, it will fail (unless you want detection only).")
            else:
                print("❓ Unknown Format.")

    except zipfile.BadZipFile:
        print("❌ Invalid Zip File.")
    except Exception as e:
        print(f"❌ Error reading zip: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default search in current dir
        zips = [f for f in os.listdir('.') if f.endswith('.zip')]
        if zips:
            detect_format(zips[0])
        else:
            print("Usage: python verify_dataset.py <path_to_zip>")
    else:
        detect_format(sys.argv[1])
