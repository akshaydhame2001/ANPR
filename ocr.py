import cv2
import easyocr

# Load the image
image_path = 'white.png'
img = cv2.imread(image_path)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Perform OCR and get the bounding boxes
results = reader.readtext(img, detail=1)

print(results)