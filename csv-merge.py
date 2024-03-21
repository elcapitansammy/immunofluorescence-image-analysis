import glob
import pandas as pd
import os

folder_path = "data TRA2B and SC35"

# Create a dictionary to store the merged dataframes
merged_files = {}

# Iterate over the CSV files
for file in os.listdir(folder_path):
    # Check if the file starts with A, B, C, or D
    if file.startswith(('B1', 'B2', 'C1', 'C2')):
        # Read the CSV file into a DataFrame
        data = pd.read_csv(os.path.join(folder_path, file))
        
        # Get the first letter of the file name
        first_letter = file[0]+file[1]
        
        # Check if the first letter is already a key in the dictionary
        if first_letter in merged_files:
            # Append the data to the existing dataframe
            merged_files[first_letter] = merged_files[first_letter]._append(data, ignore_index=True)
        else:
            # Create a new dataframe for the first letter
            merged_files[first_letter] = data

# Save the merged dataframes to new CSV files
for letter, dataframe in merged_files.items():
    # Generate the new file name
    new_file_name = f"merged_{letter}.csv"
    
    # Save the dataframe to the new CSV file
    dataframe.to_csv(new_file_name, index=False)
