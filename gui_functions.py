import cv2
import numpy as np
import csv
import pandas as pd
from io import StringIO


def create_csv_data(name, fieldnames, data):
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
    output.seek(0)
    return output


def create_image_with_mask(img, min_area, circularity_limit=0.5, color_tresh=50, color="b"):
    image = img
    # Remove fully white elements from the image
    image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(
        cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY)[1]))
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Split the image into color channels
    r, g, b = cv2.split(image)

    if color == "r":
        col = r
    elif color == "g":
        col = g
    else:
        col = b

    # Apply adaptive thresholding to segment the cells
    _, thresholded = cv2.threshold(col, color_tresh, 255, cv2.THRESH_BINARY)

    # Find contours of the cells
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create masks for each cell
    masks = []
    areas = []  # List to store the areas of each cell
    for contour in contours:
        # Calculate the area and perimeter of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Calculate circularity of the contour
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # Check if the contour is round and above the minimum area
        if circularity > circularity_limit and area > min_area:
            # Create a mask for the current cell
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            masks.append(mask)
            areas.append(area)  # Append the area to the list

    # Create a copy of the original image
    image_with_masks = image.copy()

    # Set the transparency level for the masks
    alpha = 0.5

    # Loop through each mask and overlay it on the image
    for i, mask in enumerate(masks):
        # Convert the mask to BGR format
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Blend the mask with the image using alpha blending
        image_with_masks = cv2.addWeighted(image_with_masks, 1, mask_bgr, alpha, 0)

    return image_with_masks, masks


def add_masks_to_image(image, masks):
    # Create a copy of the original image
    image_with_masks = image.copy()

    # Set the transparency level for the masks
    alpha = 0.5

    # Loop through each mask and overlay it on the image
    for i, mask in enumerate(masks):
        # Convert the mask to BGR format
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Blend the mask with the image using alpha blending
        image_with_masks = cv2.addWeighted(image_with_masks, 1, mask_bgr, alpha, 0)

    return image_with_masks, masks


def save_average_intensities_to_csv(average_intensities1, max_intensities1):
    # Create a list of dictionaries for each average intensity and area
    data = []
    for i in range(len(average_intensities1)):
        data.append({'Index': i, 'Average Intensity': average_intensities1[i], 'Max Intensity': max_intensities1[i]})

    # Define the field names for the CSV file
    fieldnames = ['Index', 'Average Intensity', 'Max Intensity']

    # Create the CSV file and write the data
    csv_data = create_csv_data("table_cell", fieldnames, data)
    csv_bytes = csv_data.getvalue().encode('utf-8')
    return csv_bytes


def calculate_average_intensity(image, masks, color):
    # Convert the color string to BGR format
    if color.lower() == 'red':
        color_bgr = [255, 0, 0]
    elif color.lower() == 'green':
        color_bgr = [0, 255, 0]
    elif color.lower() == 'blue':
        color_bgr = [0, 0, 255]
    else:
        raise ValueError("Invalid color. Please choose 'Blue', 'Green', or 'Red'.")

    # Calculate average intensity per mask for the given color
    average_intensities = []
    max_intensities=[]
    for mask in masks:
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Calculate the average intensity of the color within the mask
        average_intensity = np.mean(masked_image[:, :, color_bgr.index(max(color_bgr))])
        max_intensity = np.max(masked_image[:, :, color_bgr.index(max(color_bgr))])

        # Append the average intensity to the list
        average_intensities.append(average_intensity)
        max_intensities.append(max_intensity)

    return average_intensities, max_intensities
