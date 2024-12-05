import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
from utils import get_car, read_license_plate, write_csv

# Paths for classifiers and models
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
    # Load labels for Make/Model classifier
    with open(mmt_label_file, 'r') as f:
        mmt_labels = f.read().splitlines()

    # Load YOLO models
    vehicle_detector = YOLO(yolo_model_path)
    license_plate_detector = YOLO(license_plate_model_path)

    # Load Make/Model classifier graph
    mmt_graph = tf.compat.v1.Graph()
    with tf.compat.v1.gfile.GFile(mmt_model_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        with mmt_graph.as_default():
            tf.import_graph_def(graph_def, name='')

    # Initialize SORT tracker
    tracker = Sort()

    # Process video
    cap = cv2.VideoCapture("/path/to/video.mp4")
    results = {}

    # Video writer setup (optional)
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
            filtered_detections = []
            for detection in vehicle_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in VEHICLE_CLASSES:
                    filtered_detections.append([x1, y1, x2, y2, score])

            # Track vehicles
            track_ids = tracker.update(np.asarray(filtered_detections))

            # License plate detection
            license_plate_detections = license_plate_detector(frame)[0]
            for license_plate in license_plate_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, _ = license_plate

                # Assign license plate to tracked vehicle
                try:
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                except TypeError:
                    # Skip if no matching vehicle found
                    continue

                if car_id != -1:
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_text, license_plate_score = read_license_plate(license_plate_crop)

                    # Vehicle Make/Model classification for cars
                    if int(vehicle_detections.boxes.data.tolist()[car_id][5]) == 2:  # Car class
                        car_crop = frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2)]
                        resized_car = cv2.resize(car_crop, mmt_input_size).astype(np.float32) / 255.0
                        resized_car = np.expand_dims(resized_car, axis=0)
                        mmt_predictions = mmt_sess.run(
                            mmt_output_tensor, feed_dict={mmt_input_tensor: resized_car}
                        )
                        mmt_class_index = np.argmax(mmt_predictions)
                        mmt_label = mmt_labels[mmt_class_index]
                    else:
                        mmt_label = VEHICLE_CLASSES.get(int(vehicle_detections.boxes.data.tolist()[car_id][5]), "Unknown")

                    # Save results
                    results[frame_nmr][car_id] = {
                        'vehicle': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2], 
                            'class': mmt_label
                        },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2], 
                            'text': license_plate_text,
                            'score': license_plate_score
                        },
                    }

                    # Draw bounding boxes and labels
                    cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame, 
                        f"{mmt_label}: {license_plate_text}", 
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )

            # Write frame to output video
            out.write(frame)

            cv2.imshow("ANPR System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Write results to CSV
    write_csv(results, './results.csv')

if __name__ == "__main__":
    main()