import cv2
from ultralytics import YOLO
import easyocr

# Load YOLOv8 model
model = YOLO('license_plate_detector.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Specify language

# Load the image
image_path = "white.png"  # Replace with your image path
frame = cv2.imread(image_path)

# Check if the image was loaded successfully
if frame is None:
    print("Error loading image")
    exit()

# Run YOLOv8 detection on the loaded image
results = model(frame)

# Loop through the detections in the image
for result in results[0].boxes:
    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get bounding box coordinates
    score = result.conf[0]  # Get confidence score
    class_id = result.cls[0]  # Get class ID
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display label with confidence
    label = f'{model.names[int(class_id)]}: {score:.2f}'
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Crop the license plate from the frame
    license_plate = frame[y1:y2, x1:x2]

    # Use EasyOCR to read the license plate text
    ocr_results = reader.readtext(license_plate)
    for (bbox, text, prob) in ocr_results:
        # Draw the recognized text on the frame
        cv2.putText(frame, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        print(f'Detected license plate text: {text}')

# Display the image with detections
cv2.imshow('YOLOv8 License Plate Detection', frame)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
