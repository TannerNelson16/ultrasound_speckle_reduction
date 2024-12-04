import numpy as np
import cv2
from scipy.ndimage import laplace

def compute_gradient_and_laplacian(image):
    gradient = np.gradient(image)
    grad_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
    laplacian = laplace(image)
    return grad_magnitude, laplacian

def compute_icov(image, gradient, laplacian):
    num = 0.5 * (gradient**2) - (0.25 * laplacian**2)
    denom = (1 + 0.25 * laplacian)**2
    return np.sqrt(np.maximum(num / denom, 0))

def srad(image, num_iterations=100, delta_t=0.25, q0=1.0):
    smoothed_image = image.copy()
    for _ in range(num_iterations):
        gradient, laplacian = compute_gradient_and_laplacian(smoothed_image)
        q = compute_icov(smoothed_image, gradient, laplacian)
        c = 1 / (1 + ((q**2 - q0**2) / (q0**2 * (1 + q0**2))))
        smoothed_image += delta_t * (c * laplacian)
    return smoothed_image

def logarithmic_transformation(image):
    image = np.maximum(image, 1e-8)  # Avoid log of non-positive values
    return np.log1p(image)

def guided_filter(I, p, radius, eps):
    mean_I = cv2.boxFilter(I, -1, (radius, radius))
    mean_p = cv2.boxFilter(p, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    q = mean_a * I + mean_b
    return q

def exponential_transformation(image):
    return np.expm1(image)

def proposed_algorithm(image, srad_iterations=50, guided_radius=8, guided_eps=0.01):
    # Step 1: Apply SRAD
    srad_result = srad(image, num_iterations=srad_iterations)
    
    # Step 2: Logarithmic Transformation
    log_transformed = logarithmic_transformation(srad_result)
    
    # Step 3: Guided Filter
    guided_result = guided_filter(log_transformed, log_transformed, radius=guided_radius, eps=guided_eps)
    
    # Step 4: Exponential Transformation
    final_image = exponential_transformation(guided_result)
    
    return final_image

# Main Script
if __name__ == "__main__":
    # Load ultrasound image
    input_image = cv2.imread('speckled_cameraman.tif', cv2.IMREAD_GRAYSCALE)
    
    if input_image is None:
        print("Error: Could not load image. Please check the file path.")
        exit(1)
    
    # Normalize image to [0, 1]
    input_image = input_image / 255.0

    # Apply the proposed algorithm
    output_image = proposed_algorithm(input_image)

    # Normalize output image to [0, 1] and handle NaNs
    output_image = np.nan_to_num(output_image, nan=0.0)  # Replace NaNs with 0
    output_image = np.clip(output_image, 0, 1)          # Clamp to [0, 1]

    # Save the output
    output_path = 'denoised_image.jpg'
    cv2.imwrite(output_path, (output_image * 255).astype(np.uint8))
    print(f"Denoised image saved to {output_path}")
