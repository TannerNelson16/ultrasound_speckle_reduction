#!/user/bin/python3
# Import necessary libraries
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2

# Define SRAD function
def srad(image, num_iterations=100, delta_t=0.25, q0_squared=0.05):
    image = image.astype(np.float32)
    for _ in range(num_iterations):
        nablaN = np.roll(image, -1, axis=0) - image
        nablaS = np.roll(image, 1, axis=0) - image
        nablaE = np.roll(image, -1, axis=1) - image
        nablaW = np.roll(image, 1, axis=1) - image

        mean_square_grad = (nablaN**2 + nablaS**2 + nablaE**2 + nablaW**2) / 4.0
        q_squared = mean_square_grad / (image**2 + 1e-10)

        diffusion_coefficient = np.exp(-(q_squared - q0_squared) / (q0_squared * (1 + q0_squared)))

        image += delta_t * (
            (np.roll(diffusion_coefficient * nablaN, 1, axis=0) - diffusion_coefficient * nablaN) +
            (np.roll(diffusion_coefficient * nablaS, -1, axis=0) - diffusion_coefficient * nablaS) +
            (np.roll(diffusion_coefficient * nablaE, 1, axis=1) - diffusion_coefficient * nablaE) +
            (np.roll(diffusion_coefficient * nablaW, -1, axis=1) - diffusion_coefficient * nablaW)
        )
    return image

# Define guided filter function
def convolve(image, kernel):
    return convolve2d(image, kernel, mode='same', boundary='symm')

def guided_filter(image, radius=5, eps=0.01):
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    mean_I = convolve(image, kernel)
    mean_II = convolve(image**2, kernel)

    var_I = mean_II - mean_I**2
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I

    mean_a = convolve(a, kernel)
    mean_b = convolve(b, kernel)

    return mean_a * image + mean_b

# Load the ultrasound image
ultrasound_image_path = 'speckled_cameraman.tif'
ultrasound_image = imread(ultrasound_image_path, as_gray=True)

# Normalize the input image
ultrasound_image = (ultrasound_image - ultrasound_image.min()) / (ultrasound_image.max() - ultrasound_image.min())

# Step 1: Apply SRAD filter
srad_filtered_ultrasound = srad(ultrasound_image, num_iterations=100, delta_t=0.1, q0_squared=0.04)

# Normalize SRAD output
srad_filtered_ultrasound = (srad_filtered_ultrasound - srad_filtered_ultrasound.min()) / (
    srad_filtered_ultrasound.max() - srad_filtered_ultrasound.min())

# Step 2: Logarithmic transformation
log_transformed_ultrasound = np.log1p(srad_filtered_ultrasound)

# Step 3: Guided filter
guided_filtered_ultrasound = guided_filter(log_transformed_ultrasound, radius=4, eps=0.01)

# Save the guided filtered image
output_path_ultrasound_guided = 'unspeckled.tif'
imsave(output_path_ultrasound_guided, (guided_filtered_ultrasound * 255).astype(np.uint8))

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Ultrasound")
plt.imshow(ultrasound_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("SRAD Filtered")
plt.imshow(srad_filtered_ultrasound, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Guided Filtered")
plt.imshow(guided_filtered_ultrasound, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

