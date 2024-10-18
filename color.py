import cv2
import numpy as np

# Define color ranges for different colors
color_ranges = {
    "yellow": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([30, 255, 255])
    },
    "red": {
        "lower_1": np.array([0, 100, 100]),
        "upper_1": np.array([10, 255, 255]),
        "lower_2": np.array([170, 100, 100]),
        "upper_2": np.array([180, 255, 255])
    },
    "green": {
        "lower": np.array([40, 100, 100]),
        "upper": np.array([80, 255, 255])
    },
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([180, 255, 30])
    },
    "white": {
        "lower": np.array([0, 0, 150]),
        "upper": np.array([180, 20, 255])
    },
    "blue": {
        "lower": np.array([100, 150, 0]),
        "upper": np.array([140, 255, 255])
    }
}

def detect_dominant_colors(image):
    # Load the image
    img = cv2.imread(image)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_pixel_count = {}

    # Check each color range
    for color, ranges in color_ranges.items():
        # Initialize count for this color
        color_pixel_count[color] = 0
        
        # For single lower/upper range
        if 'lower' in ranges and 'upper' in ranges:
            mask = cv2.inRange(hsv_img, ranges['lower'], ranges['upper'])
            color_pixel_count[color] += cv2.countNonZero(mask)
        
        # For red with two ranges
        elif 'lower_1' in ranges and 'upper_1' in ranges and 'lower_2' in ranges and 'upper_2' in ranges:
            mask1 = cv2.inRange(hsv_img, ranges['lower_1'], ranges['upper_1'])
            mask2 = cv2.inRange(hsv_img, ranges['lower_2'], ranges['upper_2'])
            color_pixel_count[color] += cv2.countNonZero(mask1) + cv2.countNonZero(mask2)

    # Sort colors by pixel count
    sorted_colors = sorted(color_pixel_count.items(), key=lambda item: item[1], reverse=True)

    # Get the dominant and second dominant color
    dominant_color = sorted_colors[0][0] if len(sorted_colors) > 0 else None
    second_dominant_color = sorted_colors[1][0] if len(sorted_colors) > 1 else None

    return dominant_color, second_dominant_color, color_pixel_count

# Example usage
dominant_color, second_dominant_color, pixel_counts = detect_dominant_colors('images/black.png')
print(f"Dominant Color: {dominant_color}")
print(f"Second Dominant Color: {second_dominant_color}")
print("Pixel Counts for Each Color:", pixel_counts)
