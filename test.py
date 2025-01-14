import os
import cv2
import torch
import numpy as np
import pytesseract
from PIL import Image
from collections import defaultdict

# If Tesseract is not in your system PATH, set it manually
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# =====================
# 1) LOAD LICENSE PLATE DETECTION MODEL
# =====================
def load_detection_model(model_path=None):
    """
    Loads a YOLOv5 model from either a local checkpoint or
    from the Ultralytics hub if model_path is None.
    Make sure to install YOLOv5 dependencies:
        pip install git+https://github.com/ultralytics/yolov5.git
    """
    if model_path and os.path.exists(model_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    else:
        # Fallback to a generic YOLOv5 small model (not specialized for plates)
        # In practice, you want a specialized license plate detection model here.
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# =====================
# 2) DETECT & CROP LICENSE PLATE
# =====================
def detect_and_crop_plate(image, model, conf_threshold=0.5):
    """
    Takes a BGR image (OpenCV format), runs it through YOLOv5,
    and returns the cropped plate region if detected.
    If multiple plates are found, returns the one with highest confidence.
    """
    # Convert BGR (OpenCV) to RGB (for YOLO inference)
    results = model(image[..., ::-1])  # YOLO expects RGB
    # results.xyxy[0] -> [x1, y1, x2, y2, conf, class]
    detections = results.xyxy[0].cpu().numpy()  # shape: (N, 6)
    
    best_crop = None
    highest_conf = 0.0
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf >= conf_threshold:
            # If the model is truly specialized for license plates,
            # cls should correspond to the license plate class
            if conf > highest_conf:
                highest_conf = conf
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                best_crop = image[y1:y2, x1:x2]
    
    return best_crop

# =====================
# 3) EXTRACT TEXT FROM THE LICENSE PLATE
# =====================
def recognize_plate_text(cropped_plate):
    """
    Performs OCR on the cropped plate region using Tesseract
    and returns the recognized alphanumeric text (cleaned up).
    """
    if cropped_plate is None:
        return None
    
    # Convert to grayscale for better OCR results
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance text visibility
    gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Optionally resize to improve OCR performance
    gray = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    pil_img = Image.fromarray(gray)
    # psm 7: treat the image as a single text line
    text = pytesseract.image_to_string(pil_img, config="--psm 7")
    
    # Clean up text (remove spaces, newlines, etc.)
    text = ''.join(ch for ch in text if ch.isalnum())
    return text

# =====================
# 4) MAIN PIPELINE
# =====================
def main(
    images_folder,
    output_folder='output',
    model_path=None,
    conf_threshold=0.5
):
    """
    Processes train, valid, and test datasets for YOLOv5 training,
    while grouping images by detected license plates into separate folders.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    model = load_detection_model(model_path)
    
    datasets = ['train', 'valid', 'test']  # Specify dataset types
    plate_groups = defaultdict(list)  # {plate_text: [(image, filename), ...]}
    
    for dataset in datasets:
        dataset_images_folder = os.path.join(images_folder, dataset, 'images')
        if not os.path.exists(dataset_images_folder):
            print(f"Dataset folder {dataset_images_folder} not found!")
            continue

        print(f"Processing dataset: {dataset}")
        
        # Collect all valid image filenames in the specified folder
        image_paths = [
            os.path.join(dataset_images_folder, f)
            for f in os.listdir(dataset_images_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        print(f"Processing {len(image_paths)} images in {dataset_images_folder}...")

        for img_path in image_paths:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read {img_path}")
                continue
            
            # Detect & crop
            cropped_plate = detect_and_crop_plate(image, model, conf_threshold=conf_threshold)
            
            # OCR
            plate_text = recognize_plate_text(cropped_plate)
            if not plate_text:
                plate_text = "UNKNOWN"
            
            print(f"Detected: {plate_text} in image {os.path.basename(img_path)}")
            
            # Group image by plate_text
            plate_groups[plate_text].append((image, os.path.basename(img_path)))

    # Create folders and save grouped images based on detected plates
    for plate_text, group_items in plate_groups.items():
        # Each group is a list of tuples: (image, filename)
        plate_folder = os.path.join(output_folder, plate_text)
        os.makedirs(plate_folder, exist_ok=True)
        
        for (img, fname) in group_items:
            out_path = os.path.join(plate_folder, fname)
            cv2.imwrite(out_path, img)
    
    print("Processing complete! Grouped images saved to output folders.")

if __name__ == "__main__":
    # Example usage
    images_folder = "images"  # Folder containing train, valid, and test folders
    output_folder = "output"
    model_path = None  # Replace with your model path if available
    
    main(images_folder, output_folder, model_path, conf_threshold=0.5)
