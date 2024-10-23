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
import pandas as pd

outfile = "2024-03-13 16-01-29Gain_1_Results.csv" # Raw file path
data = pd.read_csv(outfile)
data=np.array(data)
data =data[1:]
data =[sample[0].split(';') for sample in data]

data = np.array(data)
number_of_samples = 1200
processed_data = data[5:96, :]
polar_map_matrix = np.zeros((number_of_samples * 2, number_of_samples * 2))  # Initialize empty polar map matrix

for data in processed_data:
    data=data.astype(np.int32)
    angle = data[0]  # Current angle
    k_values = np.linspace(0, 99, 100) * 0.01  # For angle variation
    r_values = np.arange(1, number_of_samples)
    Degree_Angle = (angle-100)*0.9
    x_vals = (np.cos(np.radians(Degree_Angle - k_values[:, np.newaxis])) * r_values).astype(np.int32)
    y_vals = (np.sin(np.radians(Degree_Angle - k_values[:, np.newaxis])) * r_values).astype(np.int32)

    polar_map_matrix[np.abs(number_of_samples - y_vals), np.abs(number_of_samples + x_vals)] = (data[1:-1])
    print(Degree_Angle)
# Extract the image region of interest
image = polar_map_matrix[:1200, :]
plt.imshow(image)
plt.show()

"""
Target Detection using SAM
"""

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
