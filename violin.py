import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# List of CSV files
csv_files = ['merged_B2.csv', 'merged_C2.csv']
# Initialize the y-axis limits
y_min = float('inf')
y_max = float('-inf')

# Iterate over the CSV files
for csv_file in csv_files:
    # Create a figure and axes for each violin plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Update the y-axis limits based on the data range
    y_min = min(y_min, df['Average Intensity red'].min())
    y_max = max(y_max, df['Average Intensity red'].max())
    
    # Create a violin plot for the "intensity" column using seaborn
    sns.violinplot(data=df['Average Intensity red'], ax=ax)
    
    # Set the title for the plot
    ax.set_title(f'Violin Plot for Intensity - {csv_file}')
    
    # Save the plot as an image file
    plt.savefig(f'violin_plot_{csv_file[:-4]}.png')
    
    # Show the plot
    plt.show()

# Create a figure and axes for the final plot with same y-axis limits
fig, ax = plt.subplots(figsize=(8, 6))

# Iterate over the CSV files again to create the final plot
for csv_file in csv_files:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Create a violin plot for the "intensity" column using seaborn
    sns.violinplot(data=df['Average Intensity red'], ax=ax)
    
# Set the y-axis limits for all plots
plt.ylim(y_min, y_max)

# Add a legend
plt.legend(csv_files)

# Save the final plot as an image file
plt.savefig('violin_plot_same_scale.png')

# Show the final plot
plt.show()
# Calculate the average for each CSV file
averages = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    average = df['Average Intensity red'].mean()
    averages.append(average)

# Print the averages
for i, csv_file in enumerate(csv_files):
    print(f"Average for {csv_file}: {averages[i]}")
