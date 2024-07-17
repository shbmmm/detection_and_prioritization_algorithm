import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from scipy import ndimage

# Create synthetic SAR data (before and after images)
width, height = 100, 100

before_image = np.random.rand(width, height) * 100
after_image = before_image.copy()
after_image[30:70, 30:70] += np.random.rand(40, 40) * 50  # Simulate damage

# Save images to files
with rasterio.open('before.tif', 'w', driver='GTiff', height=height, width=width, count=1, dtype=before_image.dtype) as dst:
    dst.write(before_image, 1)

with rasterio.open('after.tif', 'w', driver='GTiff', height=height, width=width, count=1, dtype=after_image.dtype) as dst:
    dst.write(after_image, 1)

# Load images
before_image = rasterio.open('before.tif').read(1)
after_image = rasterio.open('after.tif').read(1)

# Change detection: compute absolute difference
change_map = np.abs(after_image - before_image)

#Thresholding 
threshold = 30
significant_changes = change_map > threshold

#Identify regions
labeled_changes, num_features = ndimage.label(significant_changes)

#Calculate the total change in each region
region_sums = ndimage.sum(change_map, labeled_changes, range(num_features + 1))

#Prioritize regions by total change
priority_areas = np.argsort(region_sums)[::-1]

# Plot images
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
ax1.set_title('Before Image')
show(before_image, ax=ax1)
ax2.set_title('After Image')
show(after_image, ax=ax2)
ax3.set_title('Change Map')
show(change_map, ax=ax3)
ax4.set_title('Significant Changes')
show(significant_changes, ax=ax4)
plt.show()

# Plot priority areas
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_title('Priority Area')
show(labeled_changes == priority_areas[1], ax=ax)  # Plot the highest priority area
plt.show()

