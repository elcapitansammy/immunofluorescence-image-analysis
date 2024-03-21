import cv2
import numpy as np
import csv
import os

def count_cells(image_path, min_area, masks=None, color='blue'):
    # Load the image
    image = cv2.imread(image_path)
    # Remove fully white elements from the image
    image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY)[1]))
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Split the image into color channels
    b, g, r = cv2.split(image)

    # Apply adaptive thresholding to segment the cells of the specified color
    if color == 'blue':
        _, thresholded = cv2.threshold(b, 20, 255, cv2.THRESH_BINARY)
    elif color == 'green':
        _, thresholded = cv2.threshold(g, 20, 255, cv2.THRESH_BINARY)
    elif color == 'red':
        _, thresholded = cv2.threshold(r, 20, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("Invalid color specified")

    # Find contours of the cells
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create masks for each cell and calculate average intensity and area
    if masks is None:
        masks = []
    average_intensities = []
    cell_areas = []
    for contour in contours:
        # Calculate the area and perimeter of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Calculate circularity of the contour
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # Check if the contour is round, above the minimum area, and not too small
        if circularity > 0.5 and area > min_area:
            # Create a mask for the current cell
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            masks.append(mask)

            # Calculate the average intensity of the current cell
            average_intensity = np.mean(image[mask == 255])
            average_intensities.append(average_intensity)

            # Calculate the area of the current cell
            cell_area = np.sum(mask == 255)
            cell_areas.append(cell_area)

    # Overlay masks on the original image
    overlay = image.copy()
    for mask in masks:
        overlay[mask == 255] = (0, 255, 0)  # Green color for overlay

    # Create a CSV file to store the information
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_file = f'{image_name}_cell_information.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Cell ID', 'Area', 'Intensity'])
        for i, (area, intensity) in enumerate(zip(cell_areas, average_intensities)):
            writer.writerow([i+1, area, intensity])

    print(f"Cell information saved to {csv_file}")

    # Return the number of cells, masks, average intensities, and the overlay image
    return len(masks), masks, average_intensities, overlay

def count_cells_in_folder(folder_path, min_area,string, color, masks=None):
    total_masks = []
    mask_index = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tif") and string in file:
                image_path = os.path.join(root, file)
                num_cells, cell_masks, average_intensities, overlay = count_cells(image_path, min_area, masks[mask_index] if masks is not None else None, color)
                total_masks.append(cell_masks)
                mask_index += 1
                # # Display the overlay image
                # cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
                # cv2.imshow('Overlay', overlay)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    return total_masks

# Example usage
folder_path = 'CLK-IN red focused'
min_area = 200  # Minimum area for a cell to be considered
cell_masks= count_cells_in_folder(folder_path, min_area, "blue", "DAPI")
for mask in cell_masks:
    print(len(mask))

folder_path = 'CLK-IN red focused'
min_area = 200  # Minimum area for a cell to be considered
cell_masks= count_cells_in_folder(folder_path, min_area, "AF647","red", cell_masks)
for mask in cell_masks:
    print(len(mask))
