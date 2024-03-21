import cv2
import numpy as np
import os 
import csv

def create_masks(image_path, min_area):
    # Load the image
    image = cv2.imread(image_path)
    # Remove fully white elements from the image
    image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY)[1]))
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Split the image into color channels
    b, g, r = cv2.split(image)

    # Apply adaptive thresholding to segment the cells
    _, thresholded = cv2.threshold(b, 50, 255, cv2.THRESH_BINARY)

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
        if circularity > 0.5 and area > min_area:
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

    # Display the image with masks
    cv2.namedWindow("Image with Masks", cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.imshow("Image with Masks", image_with_masks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the masks and areas
    return masks, areas


def calculate_average_intensity(image, masks, color):
    # Convert the color string to BGR format
    if color.lower() == 'blue':
        color_bgr = [255, 0, 0]
    elif color.lower() == 'green':
        color_bgr = [0, 255, 0]
    elif color.lower() == 'red':
        color_bgr = [0, 0, 255]
    else:
        raise ValueError("Invalid color. Please choose 'blue', 'green', or 'red'.")

    # Calculate average intensity per mask for the given color
    average_intensities = []
    for mask in masks:
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Calculate the average intensity of the color within the mask
        average_intensity = np.mean(masked_image[:, :, color_bgr.index(max(color_bgr))])
        max_intensity = np.max(masked_image[:, :, color_bgr.index(max(color_bgr))])
        print("Max intensity:", max_intensity)

        # Append the average intensity to the list
        average_intensities.append(average_intensity)

    return average_intensities

def save_average_intensities_to_csv(average_intensities1, average_intensities2, average_intensities3, areas, name):
    # Create a list of dictionaries for each average intensity and area
    data = []
    for i in range(len(average_intensities1)):
        if average_intensities2:
            data.append({'Index': i, 'Average Intensity red': average_intensities1[i], 'Average Intensity green': average_intensities2[i], 'Average Intensity blue': average_intensities3[i], 'Area': areas[i]})
        else:
            data.append({'Index': i, 'Average Intensity red': average_intensities1[i], 'Average Intensity green': 0, 'Average Intensity blue': average_intensities3[i], 'Area': areas[i]})

    # Define the field names for the CSV file
    fieldnames = ['Index', 'Average Intensity red', 'Average Intensity green', 'Average Intensity blue', 'Area']

    # Create the CSV file and write the data
    with open(f'{name}.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def calculate_and_save(image_path, name):
    blue_masks, areas = create_masks(image_path+'DAPI.tif', 200)
    image = cv2.imread(image_path+'AF647.tif')
    average_intensities_D_red = calculate_average_intensity(image, blue_masks, 'red')

    image = cv2.imread(image_path.replace('red', 'green')+'AF488.tif')
    average_intensities_D_green = calculate_average_intensity(image, blue_masks, 'green')
    
    image = cv2.imread(image_path+'DAPI.tif')
    average_intensities_D_blue = calculate_average_intensity(image, blue_masks, 'blue')
    save_average_intensities_to_csv(average_intensities_D_red, average_intensities_D_green, average_intensities_D_blue, areas,  name)


def calculate_and_save_no_green(image_path, name):
    blue_masks, areas = create_masks(image_path+'DAPI.tif', 200)
    image = cv2.imread(image_path+'AF647.tif')
    average_intensities_D_red = calculate_average_intensity(image, blue_masks, 'red')

    image = cv2.imread(image_path+'DAPI.tif')
    average_intensities_D_blue = calculate_average_intensity(image, blue_masks, 'blue')
    save_average_intensities_to_csv(average_intensities_D_red, [], average_intensities_D_blue,areas, name)


def remove_strings(string):
    strings_to_remove = ["AF488.tif", "AF647.tif", "DAPI.tif", "_TIFF"]
    for s in strings_to_remove:
        string = string.replace(s, "")
    return string

image_path = r'TRA2B & SC35\TRA2B\Plasmid control\B1_32x_TRAB2B_LM_Fig5_TIFF\B1_32x_TRAB2B_LM_Fig5_'
calculate_and_save_no_green(image_path, "B1_32x_TRAB2B_LM_Fig5_PC")
