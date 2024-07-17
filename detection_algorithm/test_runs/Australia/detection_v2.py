import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from scipy import ndimage

def resize_image(image, shape):
    return image[:shape[0], :shape[1]]

# Load images
before_image = rasterio.open('before_bush_fire.tiff').read(1)
after_image = rasterio.open('after_bush_fire.tiff').read(1)

# Find the common shape
common_shape = (min(before_image.shape[0], after_image.shape[0]),
                min(before_image.shape[1], after_image.shape[1]))

# Resize images to the common shape
before_image = resize_image(before_image, common_shape)
after_image = resize_image(after_image, common_shape)

# Change detection: compute absolute difference
change_map = np.abs(after_image - before_image)

# Thresholding to highlight significant changes
threshold = 25
significant_changes = change_map > threshold

# Identify regions with significant changes
labeled_changes, num_features = ndimage.label(significant_changes)

# Calculate the total change in each region
region_sums = ndimage.sum(change_map, labeled_changes, range(num_features + 1))

# Prioritize regions by total change
priority_areas = np.argsort(region_sums)[::-1]

# Plot images
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
(ax1, ax2), (ax3, ax4) = axes

# Before Image
ax1.set_title('Before Image')
im1 = ax1.imshow(before_image, cmap='gray')
fig.colorbar(im1, ax=ax1, orientation='vertical')
ax1.axis('off')

# After Image
ax2.set_title('After Image')
im2 = ax2.imshow(after_image, cmap='gray')
fig.colorbar(im2, ax=ax2, orientation='vertical')
ax2.axis('off')

# Change Map
ax3.set_title('Change Map')
im3 = ax3.imshow(change_map, cmap='viridis')
fig.colorbar(im3, ax=ax3, orientation='vertical')
ax3.axis('off')

# Significant Changes
ax4.set_title('Significant Changes')
im4 = ax4.imshow(significant_changes, cmap='hot')
fig.colorbar(im4, ax=ax4, orientation='vertical')
ax4.axis('off')

# Save the figure
fig.savefig('change_detection_analysis.png')

# Plot priority areas
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_title('Highest Priority Change Area')
highest_priority_area = labeled_changes == priority_areas[1]
im5 = ax.imshow(highest_priority_area, cmap='hot')
fig.colorbar(im5, ax=ax, orientation='vertical')
ax.axis('off')

# Save the figure
fig.savefig('highest_priority_change_area.png')

plt.close('all')

