import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import csv

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
        #if len(text) < 4 or prob < 0.1:
        #    continue
        print("text", text)
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
        'frame_nmr', 'car_id', 'vehicle_bbox', 'vehicle_class', 'bbox_score', 'vehicle_confidence',
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
                    vehicle['bbox_score'],  # Vehicle detection score
                    vehicle['confidence'],  # Vehicle classification confidence
                    color['label'],  # Color label (e.g., "red", "blue")
                    color['confidence'],  # Color detection confidence
                    license_plate['text'],  # License plate text (e.g., "ABC123")
                    license_plate['score'],  # License plate recognition score
                    license_plate['lp_score']  # License plate detection score
                ])

# Configuration for model paths and thresholds
config = {
    'mmt_model_file': "/content/drive/MyDrive/Unilactic/MMRfiles/model-weights-spectrico-mmr-mobilenet-128x128-344FF72B.pb",
    'mmt_label_file': "/content/drive/MyDrive/Unilactic/MMRfiles/classifier_MMT.txt",
    'mmt_input_layer': "input_1",
    'mmt_output_layer': "softmax/Softmax",
    'mmt_input_size': (128, 128),
    'license_plate_model_path': "/content/drive/MyDrive/Unilactic/MMRfiles/license_plate_detector.pt",
    'color_model_path': "/content/drive/MyDrive/Unilactic/MMRfiles/VCoRPChen.pt",
    'yolo_model_path': "/content/drive/MyDrive/Unilactic/MMRfiles/yolov8s.pt",
    'vehicle_classes': {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},

    # Confidence thresholds for various models
    'detection_threshold': 0.5,  # For vehicle and license plate detection
    'recognition_threshold': 0.6,  # For vehicle recognition (Make/Model)
    'color_confidence_threshold': 0.25,  # For color detection
    'lp_confidence_threshold': 0.25  # For license plate detection
}

def main():

    # Load labels for Make/Model classifier
    with open(config['mmt_label_file'], 'r') as f:
        mmt_labels = f.read().splitlines()

    # Load YOLO models
    vehicle_detector = YOLO(config['yolo_model_path'])
    license_plate_detector = YOLO(config['license_plate_model_path'])
    color_model = YOLO(config['color_model_path'])


    # Load Make/Model classifier graph
    mmt_graph = tf.compat.v1.Graph()
    with tf.compat.v1.gfile.GFile(config['mmt_model_file'], 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        with mmt_graph.as_default():
            tf.import_graph_def(graph_def, name='')

    # Process video
    cap = cv2.VideoCapture("/content/drive/MyDrive/Unilactic/MMRfiles/fortunerVideo.mp4")
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()
    results = {}

    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('/content/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    with tf.compat.v1.Session(graph=mmt_graph) as mmt_sess:
        mmt_input_tensor = mmt_graph.get_tensor_by_name(f'{config["mmt_input_layer"]}:0')
        mmt_output_tensor = mmt_graph.get_tensor_by_name(f'{config["mmt_output_layer"]}:0')

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

            # Prepare detections for ByteTrack
            tracked_objects = []
            for detection in vehicle_detections.boxes.data.tolist(): 
                x1, y1, x2, y2, track_id, score, class_id = detection
                if score >= config['detection_threshold'] and int(class_id) in config['vehicle_classes']:
                    tracked_object = {
                        'track_id': int(track_id),
                        'bbox': [x1, y1, x2, y2],
                        'score': score,
                        'class_id': int(class_id)
                        }
                    tracked_objects.append(tracked_object)

            # License plate detection
            license_plate_detections = license_plate_detector.track(frame, persist=True, tracker='bytetrack.yaml')[0]
            for obj in tracked_objects:
                car_id = obj['track_id']
                xcar1, ycar1, xcar2, ycar2 = map(int, obj['bbox'])
                car_class = obj['class_id']
                score = obj['score']
                # Make/Model classification for cars
                mmt_label = config['vehicle_classes'].get(car_class, "Unknown")
                # Vehicle Make/Model classification for cars
                mmt_score = None
                car_crop = original_image[ycar1:ycar2, xcar1:xcar2, :]
                if car_class == 2:  # Car class
                    
                    resized_car = cv2.resize(car_crop, config['mmt_input_size']).astype(np.float32) / 255.0
                    resized_car = np.expand_dims(resized_car, axis=0)
                    mmt_predictions = mmt_sess.run(mmt_output_tensor, feed_dict={mmt_input_tensor: resized_car})
                    mmt_class_index = np.argmax(mmt_predictions)
                    mmt_label = mmt_labels[mmt_class_index]
                    mmt_score = np.max(mmt_predictions)

                # Color Classification using ColorCLIP YOLO Model
                color_results = color_model(car_crop)
                color_label_index = color_results[0].probs.top1  # Get top predicted class index
                color_label = color_results[0].names[color_label_index]
                color_confidence = color_results[0].probs.top1conf.item()

                # Attempt to match license plates
                license_plate = None

                lp_score = None
                for license_detection in license_plate_detections.boxes.data.tolist():
                    if len(license_detection) == 7:
                        x1, y1, x2, y2, plate_id, lp_score_val, _ = license_detection
                    elif len(detection) == 6:
                        x1, y1, x2, y2, lp_score_val, _ = detection
                    if xcar1 < x1 < xcar2 and ycar1 < y1 < ycar2 and lp_score_val >= config['lp_confidence_threshold']:
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                        #license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        #_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                        license_plate = read_license_plate(license_plate_crop)
                        lp_score = lp_score_val
                        break

                license_text, license_score = license_plate if license_plate else (None, None)
                print("license_text", license_text)

                # Combine Make/Model and Color Labels
                if car_class == 2:  # For cars, combine Make/Model and Color
                    combined_label = f"ID:{car_id} {mmt_label} {color_label}"
                else:  # For other vehicle classes, use only YOLO class label
                    combined_label = f"ID:{car_id} {config['vehicle_classes'].get(car_class, 'Unknown')} {color_label}"

                # Save results with confidence
                results[frame_nmr][car_id] = {
                    'vehicle': {
                        'bbox': [xcar1, ycar1, xcar2, ycar2],
                        'class': mmt_label,
                        'track_id': car_id,
                        'bbox_score': round(score, 4) if score is not None else None,
                        'confidence': round(mmt_score, 4) if mmt_score is not None else None  # Store make/model confidence
                    },
                    'color': {
                        'label': color_label,
                        'confidence': round(color_confidence, 4) if color_confidence is not None else None  # Store color confidence
                    },
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2] if license_plate else [None, None, None, None],
                        'text': license_text or '',
                        'score': round(license_score, 4) if license_score is not None else None,
                        'lp_score': round(lp_score, 4) if lp_score is not None else None  # Add YOLO LP detection score
                    }
                }

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1.5
                label_size, _ = cv2.getTextSize(combined_label, font, font_scale, font_thickness)
                text_size, _ = cv2.getTextSize(license_text if license_text else '', font, font_scale, font_thickness)
                label_position = (xcar1 + 5, ycar1 + 20)
                text_position = (x1 + 5, y1 - 20) if license_detection else None
                bg_color = (0, 0, 0)
                text_color = (255, 255, 255)

                # Visualization
                ### BG black
                cv2.rectangle(frame, (label_position[0] - 5, label_position[1] - 15),
                                  (label_position[0] + label_size[0] + 5, label_position[1] + 5), bg_color, -1)
                ###  Text White
                cv2.putText(frame, f"ID:{car_id} {combined_label}", label_position,
                            font, font_scale, text_color, font_thickness)
                ## Car BBox
                cv2.rectangle(frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 2)

                if license_text and text_position is not None:
                    #### Draw filled rectangle for text background
                    cv2.rectangle(frame, (int(text_position[0] - 5), int(text_position[1] + 15)),
                               (int(text_position[0] + text_size[0] + 5), int(text_position[1] -5)), bg_color, -1)  # Black background
                    #### Overlay text on top of the rectangle
                    cv2.putText(frame, license_text, text_position, font, font_scale, text_color, font_thickness)
                if license_detection:
                    #### Plate BBox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Write annotated frame
            out.write(frame)

    # Cleanup
    cap.release()
    out.release()
    # Write results to CSV
    write_csv(results, './results.csv')

if __name__ == "__main__":
    main()
