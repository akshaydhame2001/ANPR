import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO
from sort.sort import *
from typing import Dict, Any

class ANPRSystem:
    def __init__(self, 
                 yolo_detection_model: str = 'yolov8s.pt',
                 license_plate_model: str = 'license_plate_detector.pt',
                 mmt_model_file: str = "/content/drive/MyDrive/Unilactic/MMRfiles/model-weights-spectrico-mmr-mobilenet-128x128-344FF72B.pb",
                 mmt_label_file: str = "/content/drive/MyDrive/Unilactic/MMRfiles/classifier_MMT.txt",
                 color_model_path: str = "/content/drive/MyDrive/Unilactic/MMRfiles/ColorCLIP.pt"):
        """
        Initialize ANPR System with multiple models for comprehensive vehicle analysis
        """
        # YOLO Detection Model for vehicles
        self.yolo_model = YOLO(yolo_detection_model)
        
        # License Plate Detector
        self.license_plate_detector = YOLO(license_plate_model)
        
        # Vehicle Tracker
        self.mot_tracker = Sort()
        
        # Make/Model/Type Classifier Setup
        self.mmt_graph = self._load_mmt_graph(mmt_model_file)
        self.mmt_label_file = mmt_label_file
        self.mmt_labels = self._load_mmt_labels()
        
        # Color Classifier
        self.color_model = YOLO(color_model_path)
        
        # Model Configuration
        self.mmt_input_layer = "input_1"
        self.mmt_output_layer = "softmax/Softmax"
        self.mmt_input_size = (128, 128)
        
        # Trackable Vehicle Classes
        self.vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def _load_mmt_graph(self, model_file):
        """Load Tensorflow graph for Make/Model Classification"""
        graph = tf.compat.v1.Graph()
        with tf.compat.v1.gfile.GFile(model_file, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            with graph.as_default():
                tf.import_graph_def(graph_def, name='')
        return graph

    def _load_mmt_labels(self):
        """Load Make/Model Labels"""
        with open(self.mmt_label_file, 'r') as f:
            return f.read().splitlines()

    def process_video(self, input_video_path: str, output_video_path: str):
        """
        Process video for vehicle and license plate detection
        """
        results: Dict[int, Dict[int, Any]] = {}
        
        cap = cv2.VideoCapture(input_video_path)
        
        # Video Writer Setup
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
        
        frame_nmr = -1
        
        with tf.compat.v1.Session(graph=self.mmt_graph) as mmt_sess:
            mmt_input_tensor = self.mmt_graph.get_tensor_by_name(f'{self.mmt_input_layer}:0')
            mmt_output_tensor = self.mmt_graph.get_tensor_by_name(f'{self.mmt_output_layer}:0')
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_nmr += 1
                results[frame_nmr] = {}
                original_frame = frame.copy()
                
                # Detect vehicles
                vehicle_detections = self.yolo_model(frame)[0]
                vehicle_tracks = []
                
                for detection in vehicle_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    class_id = int(class_id)
                    
                    if class_id in self.vehicle_classes:
                        vehicle_tracks.append([x1, y1, x2, y2, score])
                
                # Track vehicles
                track_ids = self.mot_tracker.update(np.asarray(vehicle_tracks))
                
                # Detect license plates
                license_plates = self.license_plate_detector(frame)[0]
                
                for license_plate in license_plates.boxes.data.tolist():
                    lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = license_plate
                    
                    # Find associated vehicle
                    matched_vehicle = self._get_car(license_plate, track_ids)
                    
                    if matched_vehicle is not None:
                        xcar1, ycar1, xcar2, ycar2, car_id = matched_vehicle
                        
                        # Crop vehicle and license plate
                        vehicle_crop = original_frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2)]
                        lp_crop = original_frame[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2)]
                        
                        # Process vehicle classification (only for cars)
                        if int(class_id) == 2:  # Car
                            vehicle_info = self._classify_vehicle(vehicle_crop, mmt_sess, mmt_input_tensor, mmt_output_tensor)
                        else:
                            vehicle_info = self.vehicle_classes[class_id]
                        
                        # Process license plate
                        lp_text = self._read_license_plate(lp_crop)
                        
                        # Annotate frame
                        self._annotate_frame(frame, vehicle_info, lp_text, 
                                             (xcar1, ycar1, xcar2, ycar2),
                                             (lp_x1, lp_y1, lp_x2, lp_y2))
                        
                        # Store results
                        results[frame_nmr][car_id] = {
                            'vehicle': vehicle_info,
                            'license_plate': {
                                'text': lp_text,
                                'bbox': [lp_x1, lp_y1, lp_x2, lp_y2]
                            }
                        }
                
                # Write annotated frame
                out.write(frame)
        
        cap.release()
        out.release()
        
        return results

    def _get_car(self, license_plate, track_ids):
        """
        Find the vehicle associated with a license plate
        """
        x1, y1, x2, y2, _, _ = map(int, license_plate)
        lp_center_x = (x1 + x2) / 2
        lp_center_y = (y1 + y2) / 2
        
        for track in track_ids:
            car_x1, car_y1, car_x2, car_y2, car_id = track
            if (car_x1 < lp_center_x < car_x2 and 
                car_y1 < lp_center_y < car_y2):
                return [car_x1, car_y1, car_x2, car_y2, car_id]
        
        return None

    def _classify_vehicle(self, vehicle_crop, mmt_sess, input_tensor, output_tensor):
        """
        Classify vehicle make, model, and color
        """
        # Make/Model Classification
        mmt_image = cv2.resize(vehicle_crop, self.mmt_input_size).astype(np.float32) / 255.0
        mmt_image = np.expand_dims(mmt_image, axis=0)
        mmt_predictions = mmt_sess.run(output_tensor, feed_dict={input_tensor: mmt_image})
        mmt_class_index = np.argmax(mmt_predictions)
        mmt_label = self.mmt_labels[mmt_class_index]
        
        # Color Classification
        color_results = self.color_model(vehicle_crop)
        color_label_index = color_results[0].probs.top1
        color_label = color_results[0].names[color_label_index]
        
        return f"{mmt_label} {color_label}"

    def _read_license_plate(self, license_plate_crop):
        """
        Read license plate text (placeholder method, replace with actual OCR)
        """
        # Convert to grayscale
        lp_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, lp_thresh = cv2.threshold(lp_gray, 64, 255, cv2.THRESH_BINARY_INV)
        
        # Placeholder: Use Tesseract or custom OCR model
        # For now, just return a dummy text
        return "PLACEHOLDER_LP"

    def _annotate_frame(self, frame, vehicle_info, lp_text, 
                         vehicle_bbox, lp_bbox):
        """
        Add annotations to the frame
        """
        # Vehicle Bounding Box
        cv2.rectangle(frame, (int(vehicle_bbox[0]), int(vehicle_bbox[1])), 
                      (int(vehicle_bbox[2]), int(vehicle_bbox[3])), 
                      (0, 255, 0), 2)
        
        # License Plate Bounding Box
        cv2.rectangle(frame, (int(lp_bbox[0]), int(lp_bbox[1])), 
                      (int(lp_bbox[2]), int(lp_bbox[3])), 
                      (255, 0, 0), 2)
        
        # Vehicle Info Text
        cv2.putText(frame, str(vehicle_info), 
                    (int(vehicle_bbox[0]), int(vehicle_bbox[1])-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # License Plate Text
        cv2.putText(frame, str(lp_text), 
                    (int(lp_bbox[0]), int(lp_bbox[1])-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Example Usage
if __name__ == "__main__":
    anpr = ANPRSystem()
    results = anpr.process_video(
        "/content/drive/MyDrive/Unilactic/MMRfiles/CarsCV.mp4", 
        "/content/output_video.mp4"
    )
    print("Processed video. Results available.")