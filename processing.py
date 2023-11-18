
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def show_anns(anns):
    if len(anns) == 0:
        return

    # Sort annotations by area in descending order
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # Create an empty image
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3), dtype=np.uint8)

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.randint(0, 256, (3,)).tolist()  # Random color
        img[m] = color_mask

    # Display the image
    return img
    # cv2.imshow('Annotations', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def contour(box_segment_rgb):
    segment_image = cv2.imread('maskseg.png')  # Load your image
    gray_image = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)  # Create a binary image

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and select the largest contour (assuming it represents the polygon)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon with a smaller epsilon value
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)  # Adjust the epsilon value
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Extract x and y coordinates of the corners
    corner_points = [point[0] for point in approx_polygon]

    # Convert the OpenCV BGR image to RGB for displaying with matplotlib
    segment_image_rgb = cv2.cvtColor(segment_image, cv2.COLOR_BGR2RGB)

    # Plot the contour
    # plt.imshow(segment_image_rgb)
    # plt.plot(*zip(*corner_points), 'ro', markersize=2)  # Plot red dots at corner points
    # plt.axis('off')
    # plt.show()
    for point in corner_points:
        cv2.circle(box_segment_rgb, point, 2, (0, 0, 255), -1)  # (0, 0, 255) is the BGR color for red
    return box_segment_rgb

def process(image,mask_gen):
    masks = mask_gen.generate(image)
    return masks

def get_xy(maskofInt):
    # mask_seg=maskofInt["segmentation"]
    data = Image.fromarray(maskofInt)
    data.save('maskseg.png')

    # Load your image or mask and find the contour of the polygon
    segment_image = cv2.imread('maskseg.png')  # Load your image
    gray_image = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)  # Create a binary image

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and select the largest contour (assuming it represents the polygon)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Extract x and y coordinates of the corners
    corner_points = [(point[0][0], point[0][1]) for point in approx_polygon]

    # Calculate the center of the segment
    M = cv2.moments(largest_contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    # Convert the OpenCV BGR image to RGB for displaying with matplotlib
    segment_image_rgb = cv2.cvtColor(segment_image, cv2.COLOR_BGR2RGB)

    return corner_points, (center_x, center_y)

def euler_to_quaternion(yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

def normalize(data):
    min_range = 0
    max_range = 255
    # Calculate the minimum and maximum values in your data
    min_value = np.min(data)
    max_value = np.max(data)

    # Apply Min-Max scaling
    scaled_data = (data - min_value) / (max_value - min_value) * (max_range - min_range) + min_range
    return scaled_data
