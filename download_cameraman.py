#!/usr/bin/python3
from skimage import data
from skimage.io import imsave

# Load the standard "cameraman" image
cameraman_image = data.camera()

# Save the image to a file
imsave('cameraman.tif', cameraman_image)
print("Cameraman image saved as 'cameraman.tif'")

