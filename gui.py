import streamlit as st
from PIL import Image
import gui_functions
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("Image Analyzer")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg', 'tiff'])
st.sidebar.header("Masks Settings")
color_choice = st.sidebar.selectbox(label="Color to create masks", options=["Red", "Green", "Blue"])
color_dict = {
    "Red": "r",
    "Green": "g",
    "Blue": "b"
}
color_treshold = st.sidebar.slider(min_value=0, max_value=255, label="Color treshold", value=50)
circularity = st.sidebar.slider(min_value=0.0, max_value=1.0, label="Circularity", step=0.01, value=0.50)
color = color_dict[color_choice]

if img_file:
    img = Image.open(img_file)
    # Manipulate cropped image at will
    st.write("Preview")
    img = np.array(img)
    if st.session_state.saved_mask is None:
        image_with_masks = gui_functions.create_image_with_mask(img, 200, color=color, color_tresh=color_treshold,
                                                                circularity_limit=circularity)
    else:
        image_with_masks = gui_functions.add_masks_to_image(img, st.session_state.saved_mask)
    st.image(image_with_masks[0])
    st.caption("Cell count: " + str(len(image_with_masks[1])))

# Initialize session state if it doesn't exist
if 'saved_mask' not in st.session_state:
    st.session_state.saved_mask = None

# Button to save the mask using a lambda function
st.sidebar.button(label="Save Masks", on_click=lambda: st.session_state.update({'saved_mask': image_with_masks[1]}))
st.sidebar.button(label="Delete Masks", on_click=lambda: st.session_state.update({'saved_mask': None}))
# Display the saved mask if it exists
if st.session_state.saved_mask is not None:
    st.sidebar.write("Mask saved")
else:
    st.sidebar.write("No mask saved yet.")

st.header("Average Color per Mask analysis")
color_choice_average = st.radio(label="Average color to calculate per mask: ", options=["Red", "Green", "Blue"])
color_dict_avg = {
    "Red": "Red",
    "Green": "Green",
    "Blue": "Blue"
}
color_choice_final = color_dict_avg[color_choice_average]


def calculate():
    if st.session_state.saved_mask is None:
        st.session_state.saved_mask = image_with_masks[1]

    averages = gui_functions.calculate_average_intensity(img, st.session_state.saved_mask,
                                                         color_choice_final)
    csv_bytes = gui_functions.save_average_intensities_to_csv(averages[0], averages[1])

    # Provide download button
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"table_cell_{color_choice_final}.csv",
        mime='text/csv'
    )


st.button(label="Calculate", on_click=calculate)
