import numpy as np
import cv2

def srad_filter(image, num_iter, delta_t, kappa):
    """
    Apply SRAD filter to the image.
    """
    img = image.astype(np.float32)
    for t in range(num_iter):
        nablaN = np.roll(img, -1, axis=0) - img
        nablaS = np.roll(img, 1, axis=0) - img
        nablaE = np.roll(img, -1, axis=1) - img
        nablaW = np.roll(img, 1, axis=1) - img

        cN = np.exp(-(nablaN / kappa) ** 2)
        cS = np.exp(-(nablaS / kappa) ** 2)
        cE = np.exp(-(nablaE / kappa) ** 2)
        cW = np.exp(-(nablaW / kappa) ** 2)

        img += delta_t * (cN * nablaN + cS * nablaS + cE * nablaE + cW * nablaW)
    return img

def guided_filter(I, p, r, eps):
    """
    Apply guided filter to the image.
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    q = mean_a * I + mean_b
    return q

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Apply SRAD filter
    num_iter = 500
    delta_t = 0.15
    kappa = 17
    srad_image = srad_filter(image, num_iter, delta_t, kappa)

    # Ensure all values are non-negative
    srad_image = np.clip(srad_image, 0, None)

    # Step 2: Apply logarithmic transformation
    log_image = np.log1p(srad_image)

    # Step 3: Apply guided filter
    r = 8
    eps = 0.7
    guided_image = guided_filter(log_image, log_image, r, eps)

    # Step 4: Apply exponential transformation
    exp_image = np.expm1(guided_image)

    return exp_image

def main():
    input_image_path = 'Noisy_image.png'
    #input_image_path = 'Image01.jpg'
    #input_image_path = 'speckled_cameraman.tif'
    #output_image_path = 'denoised_ultrasound.png'
    #output_image_path = 'denoised_cameraman.png'
    output_image_path = 'denoised_Noisy_image.png'

    # Process the image
    output_image = process_image(input_image_path)

    # Save the output image
    cv2.imwrite(output_image_path, output_image)
    print(f"Processed image saved as {output_image_path}")

if __name__ == "__main__":
    main()