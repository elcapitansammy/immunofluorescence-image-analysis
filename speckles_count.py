import cv2
import numpy as np
import os 
import csv


def create_masks(image_path, min_area): #the function takes as input an image and returns masks that are taken in function of the blue color. 
    # Load the image
    image = cv2.imread(image_path)
    # Remove fully white elements from the image
    image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY)[1]))
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Split the image into color channels
    b, g, r = cv2.split(image)

    # Apply adaptive thresholding to segment the cells
    _, thresholded = cv2.threshold(b, 60, 255, cv2.THRESH_BINARY)

    # Find contours of the cells
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create masks for each cell
    masks = []
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
        if circularity > 0.5 and area > min_area:
            # Create a mask for the current cell
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            masks.append(mask)

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

    # Display the image with masks
    cv2.namedWindow("Image with Masks", cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.imshow("Image with Masks", image_with_masks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return masks


def count_speckles(image, masks, threshold): #this function counts the speckles for every mask provided given a certain treshold
    # Convert the color string to BGR format
    color_bgr = [0, 255, 0]  # Green color

    # Count speckles per mask for the given color
    speckle_counts = []
    for mask in masks:
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert the masked image to grayscale
        masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to segment the speckles
        _, thresholded = cv2.threshold(masked_gray, threshold, 255, cv2.THRESH_BINARY)

        # Find contours of the speckles
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the speckles
        speckle_count = len(contours)

        # Append the speckle count to the list
        speckle_counts.append(speckle_count)

    return speckle_counts

import matplotlib.pyplot as plt

def slice_and_plot(image, masks): #this function slices the cell with multiple lines (in this case 3) equally distant from each other and then provides a graph for each mask that is the sum of the different graphs per line.
    # Convert the color string to BGR format
    color_bgr = [0, 255, 0]  # Green color
    intensity_values_list = []  # Create a list to store the intensity values for each cell

    # Loop through each mask
    for i, mask in enumerate(masks):
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Convert the masked image to grayscale
        masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Find the contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding box of the mask
        x, y, w, h = cv2.boundingRect(contours[0])

        array_sum = np.zeros(w)

        # Slice the cell with multiple lines
        num_lines = 3
        line_positions = np.linspace(y, y + h, num_lines, dtype=int, endpoint=False)
        for line_y in line_positions:
            line_start = (x, line_y)
            line_end = (x + w, line_y)
            cv2.line(image, line_start, line_end, color_bgr, 2)

            # Get the intensity values along the line
            intensity_values = []
            for j in range(w):
                intensity_values.append(masked_gray[line_y, x + j])
            
            print(len(intensity_values))
            array_sum += np.array(intensity_values)
        
        # # Create a figure and axes for the plot
        # fig, ax = plt.subplots()

        # # Plot the intensity values
        # ax.plot(array_sum/num_lines, label=f"Cell {i+1}")

        # # Set the plot title and labels
        # ax.set_title(f"Intensity of Green along the average line for Cell {i+1}")
        # ax.set_xlabel("Pixel Position")
        # ax.set_ylabel("Intensity")
        # ax.legend()
        
        # Append the intensity values to the list
        intensity_values_list.append(intensity_values)

    # Return the list of plots and intensity values
    return intensity_values_list

from scipy import interpolate
import pandas as pd

def stretch_and_average(arrays):
    # Find the maximum length among all arrays
    max_length = max(len(array) for array in arrays)

    # Initialize a list to store the stretched arrays
    stretched_arrays = []

    # Stretch the shorter arrays to match the maximum length
    for array in arrays:
        # Create an interpolation function for the array
        f = interpolate.interp1d(range(len(array)), array, fill_value="extrapolate")

        # Use the interpolation function to stretch the array
        stretched_array = f(np.linspace(0, len(array) - 1, max_length))
        
        # Add the stretched array to the list
        stretched_arrays.append(stretched_array)

    # Calculate the average of the stretched arrays
    average_array = np.mean(stretched_arrays, axis=0)

    return average_array


# Set the image path
image_path = r"CLK-IN red focused\CLK-IN\CLK-IN treated\Plasmid control\A1_Clk+_PC_x32_LM_red_Fig5_TIFF\A1_Clk+_PC_x32_LM_red_Fig5_"
# Set the minimum area for cell detection
min_area = 100
# Create masks for the cells in the image
masks = create_masks(image_path+"DAPI.tif", min_area)
# Load the image
image = cv2.imread((image_path+"AF488.tif").replace('red', 'green'))

values = slice_and_plot(image, masks)
averaged = stretch_and_average(values)


df = pd.DataFrame({'Intensity': averaged})
df.to_csv('A1_Clk+_PC_x32_LM_red_Fig1.csv'.replace("1", "5"), index=True)

# Create a figure and axes for the plot
fig, ax = plt.subplots()

# Plot the intensity values
ax.plot(averaged)

# Set the plot title and labels
ax.set_title(f"Intensity of Green along the average line for entire image")
ax.set_xlabel("Pixel Position")
ax.set_ylabel("Intensity")

# Display the plot
plt.show()

