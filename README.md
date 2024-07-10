This is a basic analysis tool for analyzing immunofluorescent images of cells. Hope it is simpler to use than imageJ for basic tasks such as cell counting and color analysis. 

# Update: Graphical Interface added Added #
https://image-analysis-capitan.streamlit.app/

The graphical interface facilitates the visualization and creation of masks for images. These masks can be saved and applied to different images. Beneath the preview of the image with its mask, the average color within each masked area can be calculated. Additionally, a CSV file containing this data can be generated and downloaded from the website.

Note: The code for the GUI can be found in the master branch

### coming soon ###
Color graphical analysis for each image

# For developpers #
The most important files are the restarting.py file and the speckles_count.py. They contain the most important function that are used to calculate the colors and the masks  

restarting.py: 
- function to create masks (cells) based on the Blue color (DAPI) of a immunofluorescent image.
- function to calculate the average intensity of color in each mask (cell).

speckles_count.py:
- function to create masks (cells) based on the Blue color (DAPI) of a immunofluorescent image.
- function to create graphs to analyze the intensity of the green color in each cell. 

Note: this code has been developed in collaboration with github copilot
