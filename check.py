import cv2

# Load the image
image = cv2.imread("military2.png")

# Check if the image is loaded successfully
if image is None:
    print("Error: Image not loaded")
else:
    # Display the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()