import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import xyxy2xywh
from ultralytics.tracker.byte_tracker import BYTETracker
from utils import get_car, read_license_plate, write_csv

# Configuration
mmt_model_file = "/content/drive/MyDrive/Unilactic/MMRfiles/model-weights-spectrico-mmr-mobilenet-128x128-344FF72B.pb"
mmt_label_file = "/content/drive/MyDrive/Unilactic/MMRfiles/classifier_MMT.txt"
mmt_input_layer = "input_1"
mmt_output_layer = "softmax/Softmax"
mmt_input_size = (128, 128)
license_plate_model_path = "license_plate_detector.pt"
yolo_model_path = "yolov8n.pt"

# Supported vehicle classes
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def main():
    # ByteTrack Tracker Configuration
    tracker_config = {
        'track_high_thresh': 0.6,   # High confidence threshold for tracking
        'track_low_thresh': 0.1,    # Low confidence threshold for tracking
        'new_track_thresh': 0.7,    # Threshold for creating new tracks
        'match_thresh': 0.8,        # Matching threshold between detections
        'track_buffer': 30,         # Number of frames to keep lost tracks
        'mot20': False              # MOT20 benchmark mode
    }

    # Load labels for Make/Model classifier
    with open(mmt_label_file, 'r') as f:
        mmt_labels = f.read().splitlines()

    # Load YOLO models
    vehicle_detector = YOLO(yolo_model_path)
    license_plate_detector = YOLO(license_plate_model_path)

    # Initialize ByteTrack Tracker
    byte_tracker = BYTETracker(**tracker_config)

    # Load Make/Model classifier graph
    mmt_graph = tf.compat.v1.Graph()
    with tf.compat.v1.gfile.GFile(mmt_model_file, 'rb') as f:
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
    out = cv2.VideoWriter('output_video.mp4', 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          30, 
                          (frame_width, frame_height))

    with tf.compat.v1.Session(graph=mmt_graph) as mmt_sess:
        mmt_input_tensor = mmt_graph.get_tensor_by_name(f'{mmt_input_layer}:0')
        mmt_output_tensor = mmt_graph.get_tensor_by_name(f'{mmt_output_layer}:0')

        frame_nmr = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_nmr += 1
            results[frame_nmr] = {}

            # Vehicle detection
            vehicle_detections = vehicle_detector(frame)[0]
            
            # Prepare detections for ByteTrack
            filtered_detections = []
            for detection in vehicle_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in VEHICLE_CLASSES:
                    # Convert to ByteTrack format [x, y, w, h, score, class]
                    x, y, w, h = xyxy2xywh((x1, y1, x2, y2))
                    filtered_detections.append([x, y, w, h, score, int(class_id)])

            # Convert to numpy array
            detections_array = np.array(filtered_detections)

            # Track vehicles using ByteTrack
            if len(detections_array) > 0:
                tracked_objects = byte_tracker.update(
                    detections_array[:, :5],  # bbox + score
                    detections_array[:, 5],   # class
                    frame.shape[:2]           # image shape
                )
            else:
                tracked_objects = []

            # License plate detection
            license_plate_detections = license_plate_detector(frame)[0]
            for license_plate in license_plate_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, _ = license_plate

                # Find matching vehicle for license plate
                matched_vehicle = None
                for obj in tracked_objects:
                    # Check if license plate is within vehicle bounding box
                    if (obj.bbox[0] < x1 < obj.bbox[2] and 
                        obj.bbox[1] < y1 < obj.bbox[3]):
                        matched_vehicle = obj
                        break

                if matched_vehicle:
                    # Extract vehicle and license plate information
                    xcar1, ycar1, xcar2, ycar2 = map(int, matched_vehicle.bbox)
                    car_id = matched_vehicle.track_id
                    car_class = int(matched_vehicle.cls)

                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_text, license_plate_score = read_license_plate(license_plate_crop)

                    # Vehicle Make/Model classification for cars
                    if car_class == 2:  # Car class
                        car_crop = frame[ycar1:ycar2, xcar1:xcar2]
                        resized_car = cv2.resize(car_crop, mmt_input_size).astype(np.float32) / 255.0
                        resized_car = np.expand_dims(resized_car, axis=0)
                        mmt_predictions = mmt_sess.run(
                            mmt_output_tensor, feed_dict={mmt_input_tensor: resized_car}
                        )
                        mmt_class_index = np.argmax(mmt_predictions)
                        mmt_label = mmt_labels[mmt_class_index]
                    else:
                        mmt_label = VEHICLE_CLASSES.get(car_class, "Unknown")

                    # Save results
                    results[frame_nmr][car_id] = {
                        'vehicle': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2], 
                            'class': mmt_label,
                            'track_id': car_id
                        },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2], 
                            'text': license_plate_text,
                            'score': license_plate_score
                        },
                    }

                    # Visualize tracking and detection
                    cv2.rectangle(frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, 
                        f"ID:{car_id} {mmt_label}: {license_plate_text}", 
                        (xcar1, ycar1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )

            # Write annotated frame
            out.write(frame)

            # Display frame
            cv2.imshow("ANPR System with ByteTrack", frame)
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