from ultralytics import YOLO
import cv2
import csv
import easyocr

# EasyOCR and YOLO setup
reader = easyocr.Reader(['en'], gpu=True)
character_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

vehicle_detector = YOLO('yolov8s.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

def read_license_plate(license_plate_crop):
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
        'license_plate_bbox', 'license_plate_text', 'license_plate_score',
        'license_plate_detection_score'
    ]

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame_nmr, frame_results in results.items():
            for car_id, car_data in frame_results.items():
                vehicle = car_data['vehicle']
                license_plate = car_data['license_plate']

                writer.writerow([
                    frame_nmr,
                    car_id,
                    vehicle['bbox'],
                    vehicle['class'],
                    vehicle['confidence'],
                    license_plate['bbox'],
                    license_plate['text'],
                    license_plate['score'],
                    license_plate['lp_score']
                ])

config = {
    'vehicle_classes': {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
}

cap = cv2.VideoCapture("/content/drive/MyDrive/Unilactic/fortunerVideo.mp4")
results = {}
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('/content/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

frame_nmr = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_image = frame.copy()
    frame_nmr += 1
    results[frame_nmr] = {}

    # Vehicle detection
    vehicle_detections = vehicle_detector.track(frame, persist=True, tracker='bytetrack.yaml')[0]

    tracked_objects = []
    for detection in vehicle_detections.boxes.data.tolist():
        x1, y1, x2, y2, track_id, score, class_id = detection
        if score >= 0.5 and int(class_id) in config['vehicle_classes']:
            tracked_objects.append({
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'score': score,
                'class_id': int(class_id)
            })

    # License plate detection and tracking
    license_plate_detections = license_plate_detector(frame)[0]
    for obj in tracked_objects:
        car_id = obj['track_id']
        xcar1, ycar1, xcar2, ycar2 = map(int, obj['bbox'])
        car_class = obj['class_id']
        mmt_label = config['vehicle_classes'].get(car_class, "Unknown")
        mmt_score = obj['score']

        license_text, license_score, lp_bbox, lp_score = None, None, [None, None, None, None], None
        for lp_detection in license_plate_detections.boxes.data.tolist():
            x1, y1, x2, y2, lp_score_val, _ = lp_detection
            if xcar1 < x1 < xcar2 and ycar1 < y1 < ycar2 and lp_score_val >= 0.8:
                license_plate_crop = original_image[int(y1):int(y2), int(x1):int(x2)]
                license_text, license_score = read_license_plate(license_plate_crop)
                lp_bbox = [x1, y1, x2, y2]
                lp_score = lp_score_val
                break

        results[frame_nmr][car_id] = {
            'vehicle': {
                'bbox': [xcar1, ycar1, xcar2, ycar2],
                'class': mmt_label,
                'confidence': mmt_score
            },
            'license_plate': {
                'bbox': lp_bbox,
                'text': license_text or '',
                'score': license_score or '',
                'lp_score': lp_score
            }
        }

        # Visualization
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color, bg_color = (255, 255, 255), (0, 0, 0)
        if license_text:
            label = f"LP: {license_text}"
            label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(frame, (xcar1, ycar1 - 15), (xcar1 + label_size[0] + 5, ycar1), bg_color, -1)
            cv2.putText(frame, label, (xcar1 + 5, ycar1 - 5), font, font_scale, text_color, font_thickness)

        # Draw bounding boxes
        cv2.rectangle(frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 2)
        if lp_bbox != [None, None, None, None]:
            cv2.rectangle(frame, (int(lp_bbox[0]), int(lp_bbox[1])), (int(lp_bbox[2]), int(lp_bbox[3])), (0, 0, 255), 2)

    # Write annotated frame
    out.write(frame)

# Cleanup
cap.release()
out.release()
write_csv(results, './results.csv')


# ocr len and Confidence
# cap is opened or not
# frame instead of copy
# detection threshoulds vehicle and plate
