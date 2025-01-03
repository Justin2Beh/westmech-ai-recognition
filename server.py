import cv2
import easyocr
from flask import Flask, request, jsonify
import os

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Flask app for dynamic image processing
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the License Plate Detection API!"})

@app.route('/detect_license_plate', methods=['POST'])
def detect_license_plate():
    # Check if a file is provided in the POST request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    # Save the file temporarily
    temp_path = os.path.join("temp_images", file.filename)
    os.makedirs("temp_images", exist_ok=True)
    file.save(temp_path)
    
    # Read and process the image
    image = cv2.imread(temp_path)
    results = reader.readtext(image)
    
    # Initialize a response dictionary
    detected_plates = []
    
    for (bbox, text, prob) in results:
        detected_plates.append({
            "license_plate": text,
            "confidence": prob,
            "coordinates": bbox
        })
    
    # Cleanup temporary file
    os.remove(temp_path)
    
    return jsonify({"detected_plates": detected_plates})

if __name__ == '__main__':
    app.run(debug=True)
