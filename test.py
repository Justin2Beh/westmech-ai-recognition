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
            # For a generic YOLO model, we might need class filtering
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
    
    # Optional: thresholding or other preprocessing
    # gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
    
    pil_img = Image.fromarray(gray)
    text = pytesseract.image_to_string(pil_img, config="--psm 7")  # psm 7: treat the image as a single text line
    
    # Clean up text (remove spaces, newlines, etc.)
    text = ''.join(ch for ch in text if ch.isalnum())
    return text

# =====================
# 4) CREATE A COLLAGE FOR EACH PLATE GROUP
# =====================
def create_collage(images, collage_width=800):
    """
    Creates a simple horizontal collage from the given list of images (OpenCV BGR).
    Returns the collage image (OpenCV BGR).
    """
    if not images:
        return None
    
    # Resize each image so they have the same height
    heights = [img.shape[0] for img in images]
    min_height = min(heights)
    
    resized_imgs = []
    for img in images:
        ratio = min_height / img.shape[0]
        new_width = int(img.shape[1] * ratio)
        resized = cv2.resize(img, (new_width, min_height))
        resized_imgs.append(resized)
    
    # Concatenate horizontally
    collage = np.hstack(resized_imgs)
    
    # If it's wider than collage_width, scale it down
    if collage.shape[1] > collage_width:
        scale = collage_width / collage.shape[1]
        new_height = int(collage.shape[0] * scale)
        collage = cv2.resize(collage, (collage_width, new_height))
    
    return collage

# =====================
# 5) MAIN PIPELINE
# =====================
def main(
    images_folder,
    output_folder='output',
    model_path=None,
    conf_threshold=0.5
):
    """
    - Loads a YOLOv5 (license plate) model
    - For each image in images_folder:
        - Detect/crop plate
        - Perform OCR
        - Group images by recognized plate text
    - Finally, create a collage for each plate group and save results in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    model = load_detection_model(model_path)
    
    plate_groups = defaultdict(list)  # {plate_text: [ (img, filename), ... ]}
    image_paths = [
        os.path.join(images_folder, f)
        for f in os.listdir(images_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
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
        
        plate_groups[plate_text].append((image, os.path.basename(img_path)))
    
    # Now create a collage for each plate group
    for plate_text, group_items in plate_groups.items():
        # group_items is a list of tuples: (image, filename)
        images = [item[0] for item in group_items]
        
        # Create a collage
        collage = create_collage(images)
        if collage is not None:
            collage_filename = f"{plate_text}_collage.jpg"
            collage_path = os.path.join(output_folder, collage_filename)
            cv2.imwrite(collage_path, collage)
            print(f"Collage saved for plate [{plate_text}] -> {collage_path}")

        # Optionally, save each image to plate-specific subfolder
        plate_folder = os.path.join(output_folder, plate_text)
        os.makedirs(plate_folder, exist_ok=True)
        for i, (img, fname) in enumerate(group_items):
            out_path = os.path.join(plate_folder, fname)
            cv2.imwrite(out_path, img)
    
    print("Processing complete!")

if __name__ == "__main__":
    # Example usage
    # Update 'images_folder' to point to your set of images.
    images_folder = "images"  # folder containing the input images
    output_folder = "output"
    model_path = None  # or 'best.pt' if you have a specialized license-plate model
    
    main(images_folder, output_folder, model_path, conf_threshold=0.5)
