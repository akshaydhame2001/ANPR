import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import xyxy2xywh
from ultralytics.tracker.byte_tracker import BYTETracker
from utils import read_license_plate, write_csv

# Configuration for model paths and thresholds
config = {
    'mmt_model_file': "/path/to/model-weights-spectrico-mmr-mobilenet-128x128-344FF72B.pb",
    'mmt_label_file': "/path/to/classifier_MMT.txt",
    'mmt_input_layer': "input_1",
    'mmt_output_layer': "softmax/Softmax",
    'mmt_input_size': (128, 128),
    'license_plate_model_path': "license_plate_detector.pt",
    'color_model_path': "/path/to/ColorCLIP.pt",
    'yolo_model_path': "yolov8n.pt",
    'vehicle_classes': {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
    
    # Confidence thresholds for various models
    'detection_threshold': 0.5,  # For vehicle and license plate detection
    'recognition_threshold': 0.6,  # For vehicle recognition (Make/Model)
    'color_confidence_threshold': 0.7,  # For color detection
    'lp_confidence_threshold': 0.8  # For license plate detection
}

def main():
    # ByteTrack Tracker Configuration
    tracker_config = {
        'track_high_thresh': 0.6,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.7,
        'match_thresh': 0.8,
        'track_buffer': 30,
        'mot20': False
    }

    # Load labels for Make/Model classifier
    with open(config['mmt_label_file'], 'r') as f:
        mmt_labels = f.read().splitlines()

    # Load YOLO models
    vehicle_detector = YOLO(config['yolo_model_path'])
    license_plate_detector = YOLO(config['license_plate_model_path'])
    color_model = YOLO(config['color_model_path'])

    # Initialize ByteTrack Tracker
    byte_tracker = BYTETracker(**tracker_config)

    # Load Make/Model classifier graph
    mmt_graph = tf.compat.v1.Graph()
    with tf.compat.v1.gfile.GFile(config['mmt_model_file'], 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        with mmt_graph.as_default():
            tf.import_graph_def(graph_def, name='')

    # Process video
    cap = cv2.VideoCapture("/path/to/video.mp4")
    results = {}

    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

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
            vehicle_detections = vehicle_detector(frame)[0]

            # Prepare detections for ByteTrack
            filtered_detections = []
            for detection in vehicle_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if score >= config['detection_threshold'] and int(class_id) in config['vehicle_classes']:
                    x, y, w, h = xyxy2xywh((x1, y1, x2, y2))
                    filtered_detections.append([x, y, w, h, score, int(class_id)])

            detections_array = np.array(filtered_detections)

            # Track vehicles using ByteTrack
            tracked_objects = (
                byte_tracker.update(detections_array[:, :5], detections_array[:, 5], frame.shape[:2])
                if len(detections_array) > 0 else []
            )

            # License plate detection
            license_plate_detections = license_plate_detector(frame)[0]
            for obj in tracked_objects:
                car_id = obj.track_id
                xcar1, ycar1, xcar2, ycar2 = map(int, obj.bbox)
                car_class = int(obj.cls)
                mmt_label = config['vehicle_classes'].get(car_class, "Unknown")

                # Vehicle Make/Model classification for cars
                mmt_score = None
                if car_class == 2:  # Car class
                    car_crop = original_image[ycar1:ycar2, xcar1:xcar2]
                    resized_car = cv2.resize(car_crop, config['mmt_input_size']).astype(np.float32) / 255.0
                    resized_car = np.expand_dims(resized_car, axis=0)
                    mmt_predictions = mmt_sess.run(mmt_output_tensor, feed_dict={mmt_input_tensor: resized_car})
                    mmt_class_index = np.argmax(mmt_predictions)
                    mmt_label = mmt_labels[mmt_class_index]
                    mmt_score = np.max(mmt_predictions)

                # Color Classification using ColorCLIP YOLO Model
                color_results = color_model(original_image[ycar1:ycar2, xcar1:xcar2])
                color_label_index = color_results[0].probs.top1  # Get top predicted class index
                color_label = color_results[0].names[color_label_index]
                color_confidence = color_results[0].probs.max().item()

                # Combine Make/Model and Color Labels
                if car_class == 2:  # For cars, combine Make/Model and Color
                    combined_label = f"{mmt_label} {color_label}"
                else:  # For other vehicle classes, use only YOLO class label
                    combined_label = config['vehicle_classes'].get(car_class, "Unknown") + " " + {color_label}

                # Attempt to match license plates
                license_plate = None

                lp_score = None
                for license_detection in license_plate_detections.boxes.data.tolist():
                    x1, y1, x2, y2, lp_score_val, _ = license_detection
                    if xcar1 < x1 < xcar2 and ycar1 < y1 < ycar2 and lp_score_val >= config['lp_confidence_threshold']:
                        license_plate_crop = original_image[int(y1):int(y2), int(x1):int(x2)]
                        #license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        #_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                        license_plate = read_license_plate(license_plate_crop)
                        lp_score = lp_score_val
                        break

                license_text, license_score = license_plate if license_plate else (None, None)

                # Save results with confidence
                results[frame_nmr][car_id] = {
                    'vehicle': {
                        'bbox': [xcar1, ycar1, xcar2, ycar2],
                        'class': mmt_label,
                        'track_id': car_id,
                        'confidence': mmt_score  # Store make/model confidence
                    },
                    'color': {
                        'label': color_label,
                        'confidence': color_confidence  # Store color confidence
                    },
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2] if license_plate else [None, None, None, None],
                        'text': license_text or '',
                        'score': license_score or '',
                        'lp_score': lp_score  # Add YOLO LP detection score
                    }
                }

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                label_size, _ = cv2.getTextSize(combined_label, font, font_scale, font_thickness)
                label_position = (xcar1 + 5, ycar1 + 20)
                bg_color = (0, 0, 0)
                text_color = (255, 255, 255)

                # Visualization
                ## BG black Text White
                cv2.rectangle(frame, (label_position[0] - 5, label_position[1] - 15),
                                  (label_position[0] + label_size[0] + 5, label_position[1] + 5), bg_color, -1)
                ##  Text White
                cv2.putText(frame, f"ID:{car_id} {combined_label} LP:{license_text}", label_position,
                            font, font_scale, text_color, font_thickness)
                # Car BBox
                cv2.rectangle(frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 2)
                # Plate BBox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Write annotated frame
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Write results to CSV
    write_csv(results, './results.csv')

if __name__ == "__main__":
    main()
