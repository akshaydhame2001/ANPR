import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO

# Paths for Make/Model Classifier
mmt_model_file = "/content/drive/MyDrive/Unilactic/MMRfiles/model-weights-spectrico-mmr-mobilenet-128x128-344FF72B.pb"
mmt_label_file = "/content/drive/MyDrive/Unilactic/MMRfiles/classifier_MMT.txt"
mmt_input_layer = "input_1"
mmt_output_layer = "softmax/Softmax"
mmt_input_size = (128, 128)

# Paths for Color Classifier
color_model_file = "/content/drive/MyDrive/Unilactic/MMRfiles/model-weights-spectrico-car-colors-mobilenet-224x224-052EAC82.pb"
color_label_file = "/content/drive/MyDrive/Unilactic/MMRfiles/color_labels.txt"
color_input_layer = "input_1"
color_output_layer = "softmax/Softmax"
color_input_size = (224, 224)

# Load labels for Make/Model and Color Classifiers
with open(mmt_label_file, 'r') as f:
    mmt_labels = f.read().splitlines()
with open(color_label_file, 'r') as f:
    color_labels = f.read().splitlines()

# Load Make/Model Classifier Graph
mmt_graph = tf.compat.v1.Graph()
with tf.compat.v1.gfile.GFile(mmt_model_file, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    with mmt_graph.as_default():
        tf.import_graph_def(graph_def, name='')

# Load Color Classifier Graph
color_graph = tf.compat.v1.Graph()
with tf.compat.v1.gfile.GFile(color_model_file, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    with color_graph.as_default():
        tf.import_graph_def(graph_def, name='')

# YOLOv8 Detection Model
yolo_model = YOLO('/content/drive/MyDrive/Unilactic/MMRfiles/yolov8s.pt')

# Start TensorFlow Sessions
with tf.compat.v1.Session(graph=mmt_graph) as mmt_sess, tf.compat.v1.Session(graph=color_graph) as color_sess:
    mmt_input_tensor = mmt_graph.get_tensor_by_name(f'{mmt_input_layer}:0')
    mmt_output_tensor = mmt_graph.get_tensor_by_name(f'{mmt_output_layer}:0')
    color_input_tensor = color_graph.get_tensor_by_name(f'{color_input_layer}:0')
    color_output_tensor = color_graph.get_tensor_by_name(f'{color_output_layer}:0')

    # Open video
    cap = cv2.VideoCapture("/content/drive/MyDrive/Unilactic/MMRfiles/cars2red.mp4")
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Output video writer
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
        results = yolo_model(frame)

        for box in results[0].boxes.data.cpu().numpy():
            xmin, ymin, xmax, ymax, conf, class_id = map(int, box[:6])

            # YOLO Classes: Cars, Motorcycles, Buses, Trucks
            yolo_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
            if class_id not in yolo_classes:
                continue

            cropped_image = original_image[ymin:ymax, xmin:xmax]

            # Process cars for Make/Model and Color classification
            if class_id == 2:  # Car
                # Make/Model Classification
                mmt_image = cv2.resize(cropped_image, mmt_input_size).astype(np.float32) / 255.0
                mmt_image = np.expand_dims(mmt_image, axis=0)
                mmt_predictions = mmt_sess.run(mmt_output_tensor, feed_dict={mmt_input_tensor: mmt_image})
                mmt_class_index = np.argmax(mmt_predictions)
                mmt_label = mmt_labels[mmt_class_index]

                # Color Classification
                color_image = cv2.resize(cropped_image, color_input_size).astype(np.float32) / 255.0
                color_image = np.expand_dims(color_image, axis=0)
                color_predictions = color_sess.run(color_output_tensor, feed_dict={color_input_tensor: color_image})
                color_class_index = np.argmax(color_predictions)
                color_label = color_labels[color_class_index]

                # Combine Make/Model and Color Labels
                combined_label = f"{mmt_label} {color_label}"

            else:
                # Use YOLO labels for non-car vehicles
                combined_label = yolo_classes[class_id]

            # Draw Bounding Box and Label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            label_size, _ = cv2.getTextSize(combined_label, font, font_scale, font_thickness)
            label_position = (xmin + 5, ymin + 20)

            # Draw Background and Text
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)
            frame = cv2.rectangle(frame, (label_position[0] - 5, label_position[1] - 15),
                                  (label_position[0] + label_size[0] + 5, label_position[1] + 5),
                                  bg_color, -1)
            frame = cv2.putText(frame, combined_label, label_position, font, font_scale, text_color, font_thickness)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        output_video.write(frame)

    cap.release()
    output_video.release()
    print("Processed video saved to /content/output_video.mp4.")
    cv2.destroyAllWindows()
