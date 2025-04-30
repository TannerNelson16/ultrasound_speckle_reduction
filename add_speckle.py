from PIL import Image
import numpy as np

# Load the images
clean_image_path = 'camera_man_clean.png'  # Replace with the path to your clean image
noisy_image_path = 'Noisy_image.png'      # Replace with the path to your noisy image

clean_image = Image.open(clean_image_path).convert('L')
clean_ult_image = Image.open('Image01.jpg').convert('L')
noisy_image = Image.open(noisy_image_path).convert('L')

# Resize the clean image to match the noisy image's dimensions
noisy_image_resized = noisy_image.resize(clean_ult_image.size)
clean_image_resized = clean_image.resize(clean_ult_image.size)
noisy_image_array_resized = np.array(noisy_image_resized)

# Convert the noisy image to a numpy array
clean_image_array = np.array(clean_image_resized)

clean_ult_image_array = np.array(clean_ult_image)

# Calculate the noise by subtracting the clean image from the noisy image
noise = noisy_image_array_resized - clean_image_array

# Add the extracted noise back to the clean image
synthetic_noisy_image_array = clean_ult_image_array + noise

# Clip values to ensure they're within the valid range for image data (0-255)
synthetic_noisy_image_array = np.clip(synthetic_noisy_image_array, 0, 255).astype(np.uint8)

# Convert the resulting array back to an image
synthetic_noisy_image = Image.fromarray(synthetic_noisy_image_array)

# Save the synthetic noisy image
output_path = 'noisy_ultrasound.png'  # Replace with your desired output path
synthetic_noisy_image.save(output_path)

# Optionally, display the synthetic noisy image
synthetic_noisy_image.show()

print(f"Synthetic noisy image saved to: {output_path}")
