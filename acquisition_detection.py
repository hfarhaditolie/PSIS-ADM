from brping import Ping360
from pathlib import Path
import csv
import re
from datetime import datetime, timedelta, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from skimage.measure import label, regionprops

# Helper functions for displaying masks, points, and boxes
def show_mask(mask, ax, random_color=False):
    """Displays a mask on a given axis with a specified color."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 30/255, 30/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Displays positive and negative points on a plot."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Displays bounding box on the image."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    print(box)
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def mask_process(mask):
    """Processes the mask by removing regions with small areas."""
    probs = regionprops(label(mask))
    for prob in probs:
        if prob.area <= 600:
            mask[prob.bbox[0]:prob.bbox[2], prob.bbox[1]:prob.bbox[3]] = 0
    return mask

def find_centroid(mask):
    """Finds centroids of regions in the mask."""
    probs = regionprops(label(mask))
    centroids = []
    for prob in probs:
        print(prob.centroid)
        centroids.append(prob.centroid[::-1])  # Reversing the order for (x, y)
    return centroids

# Importing the segmentation model
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Initialize the SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator_2 = SamPredictor(sam)

# Sonar-related calculation functions
def calculate_v_sound(T, S, D):
    """Calculates the speed of sound in water based on temperature (T), salinity (S), and depth (D)."""
    return 1410 + 4.21 * T - 0.037 * T ** 2 + 1.10 * S + 0.018 * D

def calculate_sample_distance(max_range, number_of_samples):
    """Calculates the distance between samples based on the sonar range and number of samples."""
    return max_range / number_of_samples

def calculate_sample_period(sample_distance, v_sound):
    """Calculates the sample period based on the distance and speed of sound."""
    return sample_distance / (v_sound * 12.5 * 10 ** (-9))

# Sonar configuration parameters
T = 16  # Temperature in °C
S = 0  # Salinity (optional)
D = 0.15  # Depth in meters
max_range = 1.6  # Maximum range in meters
number_of_samples = 700  # Number of samples for sonar
G = 1  # Gain setting

# Initialize Ping360 sonar instance
p = Ping360()

# Connect to the sonar via serial port (adjust based on your system)
# p.connect_serial("/dev/ttyUSB0", 115200)  # For Linux
p.connect_serial("COM4", 115200)  # For Windows

# Initialize sonar and set basic configurations
p.initialize()
p.set_number_of_samples(number_of_samples)

# Calculations based on sonar parameters
v_sound = calculate_v_sound(T, S, D)  # Speed of sound in water
sample_distance = calculate_sample_distance(max_range, number_of_samples)  # Sample distance
sample_period = calculate_sample_period(sample_distance, v_sound)  # Sample period
p.set_sample_period(int(sample_period))  # Set sample period in the sonar device
p.set_gain_setting(G)  # Set gain
p.set_transmit_duration(16)  # Set transmit duration
p.set_transmit_frequency(500)  # Set transmit frequency in kHz

# Prepare to display the polar map in real-time
plt.ion()  # Turn on interactive mode for live updating
fig, ax = plt.subplots()
polar_map_matrix = np.zeros((number_of_samples * 2, number_of_samples * 2))  # Initialize empty polar map matrix

# Main loop to collect and display sonar data
while True:
    data_vector = []  # List to store sonar data

    for x in range(150, 251):  # Scan over a full circle (adjust angle range as needed)
        response = p.transmitAngle(x)  # Transmit sonar signal at angle 'x'
        data = np.frombuffer(response.data, dtype=np.uint8)  # Convert response data to numpy array
        arr_list = data.tolist()
        arr_list.insert(0, x)  # Insert angle at the beginning of the list
        data = np.array(arr_list)

        # Precompute angles and their cos/sin values for vectorization
        angle = data[0]  # Current angle
        k_values = np.linspace(0, 99, 100) * 0.01  # Angle variation for fine adjustments
        r_values = np.arange(1, number_of_samples)  # Range values for each sample
        Degree_Angle = (angle - 100) * 0.9  # Adjusted degree angle

        # Calculate x and y coordinates for polar map plotting
        x_vals = (np.cos(np.radians(Degree_Angle - k_values[:, np.newaxis])) * r_values).astype(np.int32)
        y_vals = (np.sin(np.radians(Degree_Angle - k_values[:, np.newaxis])) * r_values).astype(np.int32)

        # Update the polar map matrix
        polar_map_matrix[np.abs(number_of_samples - y_vals), np.abs(number_of_samples + x_vals)] = data[1:-1]
        print(Degree_Angle)

    # Extract the image region of interest
    image = polar_map_matrix[:700, :]
    print(image.shape)
    
    # Create a copy of the image for processing
    I = image.copy()

    # Set region of interest based on your testing scenario
    I[:120, :] = 0
    I[550:, :] = 0
    I[:, :520] = 0
    I[:, 880:] = 0

    # Binary thresholding and mask processing
    th, bin_im = cv2.threshold(I, 150, 255, cv2.THRESH_BINARY)
    mask = mask_process(bin_im)
    
    # Find centroids of detected regions
    centroids = find_centroid(mask)
    input_point = np.array(centroids)
    input_label = np.ones(len(input_point))  # Label all centroids as positive points

    # Display centroids
    for c in centroids:
        print(f'Centroid: {c}')

    # Stack the image for processing
    image = np.stack((image,) * 3, axis=-1)

    # Set image for mask prediction and run SAM
    mask_generator_2.set_image(image.astype("uint8"))
    masks, scores, logits = mask_generator_2.predict(
        multimask_output=True,
        point_coords=input_point,
        point_labels=input_label,
    )

    # Display results
    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 12}
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Process only significant masks
        probs = regionprops(label(mask))
        plt.figure(figsize=(10, 10))
        plt.imshow(image / 255)
        show_mask(mask, plt.gca())
        
        # Annotate the mask with distance information
        for prob in probs:
            distance = round((mask.shape[0] - prob.bbox[2] - 4) / 4.3, 2) # 4.3 indicates that each centimeter is represented using approximately 4.3 pixels. refined this based on the max range as 4.3=number of rows/max range -> 4.3=700/1.6
            if prob.area < 200:
                continue
            x0, y0 = prob.bbox[1], prob.bbox[0]
            w, h = prob.bbox[3] - prob.bbox[1], prob.bbox[2] - prob.bbox[0]
            plt.text(x0 + w + 15, y0 + h, f'distance: {distance} cm', bbox=dict(facecolor='yellow', alpha=0.4))
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

    # Update the live polar map plot
    ax.imshow(polar_map_matrix[:700, :], cmap='gray')
    ax.set_title(f'Polar Map - Angle {Degree_Angle}')
    plt.pause(1)
    break  # Exit after the first loop (remove for continuous updates)
