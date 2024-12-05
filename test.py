import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO

# Paths for the classifiers
color_model_path = "/content/ColorCLIP.pt"  # YOLOv8 color classifier path
mmt_model_file = "/content/model-weights-spectrico-mmr-mobilenet-128x128-344FF72B.pb"  # MMT classifier path
mmt_label_file = "/content/classifier_MMT.txt"  # Labels for MMT classifier
input_layer = "input_1"  # Input layer of MMT model
output_layer = "softmax/Softmax"  # Output layer of MMT model
mmt_input_size = (128, 128)

# Load MMT labels
with open(mmt_label_file, 'r') as f:
    mmt_labels = f.read().splitlines()

# Load YOLO color classifier
color_model = YOLO(color_model_path)

# Load MMT classifier graph
graph = tf.compat.v1.Graph()
with tf.compat.v1.gfile.GFile(mmt_model_file, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

# Open TensorFlow session for MMT classifier
with tf.compat.v1.Session(graph=graph) as sess:
    input_tensor = graph.get_tensor_by_name(f'{input_layer}:0')
    output_tensor = graph.get_tensor_by_name(f'{output_layer}:0')

    # Load YOLOv8 vehicle detection model
    vehicle_model = YOLO('/content/yolov8s.pt')  # Replace with your YOLOv8 detection model path

    # Open video source
    cap = cv2.VideoCapture("/content/ANPR.mp4")
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Prepare video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(
        "/content/output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_image = frame.copy()

        # Detect vehicles using YOLOv8
        vehicle_results = vehicle_model(frame)

        for box in vehicle_results[0].boxes.data.cpu().numpy():
            xmin, ymin, xmax, ymax, conf, class_id = map(int, box[:6])

            # Crop vehicle image
            vehicle_image = original_image[ymin:ymax, xmin:xmax]

            # Prepare vehicle image for MMT classification
            mmt_image = cv2.resize(vehicle_image, mmt_input_size).astype(np.float32)
            mmt_image = np.expand_dims(mmt_image / 255.0, axis=0)

            # Predict MMT label
            mmt_predictions = sess.run(output_tensor, feed_dict={input_tensor: mmt_image})
            mmt_class_index = np.argmax(mmt_predictions)
            mmt_label = mmt_labels[mmt_class_index]

            # Predict color using YOLO color classifier
            color_results = color_model(vehicle_image)
            color_index = color_results[0].probs.top1
            color_label = color_results[0].names[color_index]

            # Combine color and MMT label
            combined_label = f"{mmt_label}-{color_label}"

            # Draw bounding box and label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            label_size, _ = cv2.getTextSize(combined_label, font, font_scale, font_thickness)
            label_position = (xmin + 5, ymin + 20)
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)

            # Draw background rectangle
            frame = cv2.rectangle(frame, (label_position[0] - 5, label_position[1] - 15),
                                  (label_position[0] + label_size[0] + 5, label_position[1] + 5), bg_color, -1)

            # Draw label text
            frame = cv2.putText(frame, combined_label, label_position, font, font_scale, text_color, font_thickness)

            # Draw bounding box
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Write frame to output video
        output_video.write(frame)

    cap.release()
    output_video.release()
    print("Processed video saved to /content/output_video.mp4.")
    cv2.destroyAllWindows()
