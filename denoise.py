import numpy as np
import imageio
from matplotlib import pyplot as plt

def compute_icov(image):
    """
    Compute the Instantaneous Coefficient of Variation (ICOV) with clamping to avoid numerical instability.
    """
    img = image.astype(np.float32)
    gradient_squared = (
        (np.roll(img, -1, axis=0) - img) ** 2 +  # North gradient
        (np.roll(img, 1, axis=0) - img) ** 2 +   # South gradient
        (np.roll(img, -1, axis=1) - img) ** 2 +  # East gradient
        (np.roll(img, 1, axis=1) - img) ** 2     # West gradient
    )
    laplacian = (
        np.roll(img, -1, axis=0) + np.roll(img, 1, axis=0) +
        np.roll(img, -1, axis=1) + np.roll(img, 1, axis=1) - 4 * img
    )
    icov = np.sqrt(
        np.maximum(0.5 * (np.maximum(gradient_squared / (img ** 2 + 1e-8), 0) -
                          np.maximum((laplacian / (img + 1e-8)) ** 2, 0)), 0)
    )
    return np.nan_to_num(icov, nan=0.0, posinf=1.0, neginf=0.0)

def diffusion_coefficient(icov, q0):
    """
    Calculate diffusion coefficient c(q) with clamping to avoid instability.
    """
    return np.maximum(1.0 / (1.0 + np.maximum((icov ** 2 - q0 ** 2) / (q0 ** 2 + 1e-8), 0)), 0)

def srad_filter(image, num_iter, delta_t, q0):
    """
    Apply SRAD filter to the image using fixed ICOV and diffusion coefficient for stability.
    """
    img = image.astype(np.float32)
    for t in range(num_iter):
        icov = compute_icov(img)
        c = diffusion_coefficient(icov, q0)

        nablaN = np.roll(img, -1, axis=0) - img
        nablaS = np.roll(img, 1, axis=0) - img
        nablaE = np.roll(img, -1, axis=1) - img
        nablaW = np.roll(img, 1, axis=1) - img

        img += delta_t * (
            c * nablaN + c * nablaS + c * nablaE + c * nablaW
        )
    return img
def box_filter(img, r):
    """
    Apply a box filter to the image using numpy.
    """
    h, w = img.shape
    padded_img = np.pad(img, ((r, r), (r, r)), mode='reflect')
    cumsum = np.cumsum(np.cumsum(padded_img, axis=0), axis=1)
    result = cumsum[2*r:h+2*r, 2*r:w+2*r] + cumsum[:h, :w] - cumsum[2*r:h+2*r, :w] - cumsum[:h, 2*r:w+2*r]
    return result / ((2 * r + 1) ** 2)

def guided_filter(I, p, r, eps):
    """
    Apply guided filter to the image using numpy.
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    
    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(I * I, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * I + mean_b
    return q

def process_image(image_path):
    # Load the image
    image = imageio.v2.imread(image_path, mode="L")
    image = image / 255.0  # Normalize to [0, 1]

    # Step 1: Apply SRAD filter
    num_iter = 2000  # Reduced for faster testing
    #num_iter = 500  # ultrasound Image
    delta_t = 0.05 # Time step parameter (tune as needed)
    #delta_t = 0.07  # ultrasound Image
    q0 = 0.8  # Speckle scale parameter (tune as needed)
    #q0 = 0.001  # ultrasound Image
    print("Applying SRAD filter...")
    srad_image = srad_filter(image, num_iter, delta_t, q0)

    # Ensure all values are non-negative
    srad_image = np.clip(srad_image, 0, None)

    # Step 2: Apply logarithmic transformation
    print("Applying logarithmic transformation...")
    log_image = np.log1p(srad_image)

    # Step 3: Apply guided filter
    r = 20
    #r = 4  # ultrasound Image
    eps = 0.0000001
    #eps = 0.000001  # ultrasound Image
    print("Applying guided filter...")
    guided_image = guided_filter(log_image, log_image, r, eps)

    # Step 4: Apply exponential transformation
    print("Applying exponential transformation...")
    exp_image = np.expm1(guided_image)

    # Scale back to [0, 255] for saving
    print("Scaling back to [0, 255]...")
    exp_image = np.clip(exp_image * 255, 0, 255).astype(np.uint8)

    return exp_image

# Testing on the noisy image
noisy_image_path = "noisy_camera_man.png"
output_path = "denoised_camera_man.png"
#noisy_image_path = "Image01.jpg"
#output_path = "Denoised_Ultrasound.jpg"
noisy_image = imageio.v2.imread(noisy_image_path, mode="L")
denoised_image = process_image(noisy_image_path)
clean_image = imageio.v2.imread("camera_man_clean.png", mode="L")

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# Calculate PSNR and SSIM
psnr_value = psnr(clean_image, denoised_image, data_range=255)
ssim_value = ssim(clean_image, denoised_image, data_range=255)
print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.2f}")


# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Denoised Image")
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

