import easyocr
import csv
import cv2

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)
character_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def preprocess_license_plate(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 17, 4)
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morphed

def read_license_plate(license_plate_crop):
    #license_plate_crop = preprocess_license_plate(license_plate_crop)
    detections = reader.readtext(license_plate_crop)
    if not detections:
        return None, None
    textArr = []
    total_prob = 0
    for (bbox, text, prob) in detections:
        if len(text) < 4 or prob < 0.1:
            continue
        text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
        text = "".join([c for c in text if c in character_list])
        total_prob += prob
        textArr.append([text, prob])

    combined_text = "".join([t[0] for t in textArr])
    average_prob = total_prob / len(textArr) if textArr else 0

    return combined_text, average_prob

def write_csv(results, output_file):
    """Write results to CSV."""
    header = [
        'frame_nmr', 'car_id', 'vehicle_bbox', 'vehicle_class', 'vehicle_confidence',
        'color_label', 'color_confidence', 'license_plate_text', 'license_plate_score',
        'license_plate_detection_score'
    ]

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Iterate through the results to write each row
        for frame_nmr, frame_results in results.items():
            for car_id, car_data in frame_results.items():
                vehicle = car_data['vehicle']
                color = car_data['color']
                license_plate = car_data['license_plate']
                
                # Write data for each vehicle (car_id) in the frame
                writer.writerow([
                    frame_nmr,
                    car_id,
                    vehicle['bbox'],  # Vehicle bounding box coordinates [xmin, ymin, xmax, ymax]
                    vehicle['class'],  # Vehicle class label (e.g., "car", "truck")
                    vehicle['confidence'],  # Vehicle detection confidence
                    color['label'],  # Color label (e.g., "red", "blue")
                    color['confidence'],  # Color detection confidence
                    license_plate['text'],  # License plate text (e.g., "ABC123")
                    license_plate['score'],  # License plate recognition score
                    license_plate['lp_score']  # License plate detection score
                ])
