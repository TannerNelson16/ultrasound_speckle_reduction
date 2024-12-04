import numpy as np
import cv2

def add_speckle_noise(image, mean=0, var=0.04):
    """
    Add speckle noise to the image.
    """
    row, col = image.shape
    gauss = np.random.normal(mean, var ** 0.5, (row, col))
    noisy = image + image * gauss
    noisy = np.clip(noisy, 0, 255)  # Ensure pixel values are within [0, 255]
    return noisy.astype(np.uint8)

def main():
    input_image_path = 'cameraman.tif'
    output_image_path = 'speckled_cameraman.tif'

    # Load the image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Add speckle noise to the image
    speckled_image = add_speckle_noise(image)

    # Save the speckled image
    cv2.imwrite(output_image_path, speckled_image)

if __name__ == "__main__":
    main()
