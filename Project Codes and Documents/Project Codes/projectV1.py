import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter

# Placeholder for Retinex decomposition
def estimate_illumination_map(image):
    # Use np.max with axis=2 to find the maximum across the color channels (R, G, B)
    illumination_map = np.max(image, axis=2)
    return illumination_map

def estimate_reflectance(image, illumination_map):
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        reflectance = image / illumination_map[:,:,None]
        reflectance[~np.isfinite(reflectance)] = 0  # Replace infinities and NaNs with 0
    return reflectance

# TV-ADMM denoising (using skimage's denoise_tv_chambolle as a placeholder)
def tv_admm_denoising(reflectance, weight=0.2):
    denoised = np.zeros_like(reflectance)
    for i in range(reflectance.shape[2]):  # Apply denoising channel-wise
        denoised[:, :, i] = denoise_tv_chambolle(reflectance[:, :, i], weight=weight, channel_axis=False)
    return denoised

# Tikhonov regularization for illumination enhancement
# def enhance_illumination(illumination, lambda_reg=1.0, num_iter=100, learning_rate=0.01):
#     enhanced_illumination = np.copy(illumination)

#     for _ in range(num_iter):
#         grad = 2 * (enhanced_illumination - illumination)  # Data fidelity term
#         grad += 2 * lambda_reg * gaussian_filter(enhanced_illumination, sigma=2, order=2)  # Regularization term
#         enhanced_illumination -= learning_rate * grad

#     return enhanced_illumination

def apply_gamma_correction(image, gamma=1.5):
    # Apply gamma correction
    gamma_corrected = np.power(image, gamma)
    return gamma_corrected


def enhance_illumination(image):
    # # Convert to float and normalize
    # image_float = image.astype(np.float32) / 255.0

    # # Estimate illumination map
    # illumination_map = estimate_illumination_map(image_float)

    # Apply histogram equalization to the illumination map
    # Convert to 8-bit for histogram equalization
    illumination_map_8bit = (image * 255).astype(np.uint8)
    # equalized_illumination = cv2.equalizeHist(illumination_map_8bit) / 255.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_illumination = clahe.apply(illumination_map_8bit) / 255.0

    # # Recombine with the reflectance component
    # enhanced_image = np.zeros_like(image_float)
    # for i in range(3):  # For each color channel
    #     enhanced_image[:, :, i] = image_float[:, :, i] / illumination_map * equalized_illumination

    return equalized_illumination



st.title('Image Illumination and Reflectance Enhancement')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # image = Image.open(uploaded_file)
    # image = np.array(image)
    # Load the image
    # Read the image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Display original image
    st.image(image_rgb, caption='Original Image', use_column_width=True)
    
    


    # Process and display the results
    if st.button('Enhance Image'):
        if image is None:
            print("Error loading image")
        else:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert image to float32 for more precision in calculations
            image_float = np.float32(image_rgb) / 255.0

            # Estimate the illumination map
            illumination = estimate_illumination_map(image_float)

            # Estimate the reflectance component
            reflectance = estimate_reflectance(image_float, illumination)

        # Enhance the reflectance component using TV-ADMM denoising
        enhanced_reflectance = tv_admm_denoising(reflectance, weight=0.25)

        enhanced_illumination = enhance_illumination(illumination.squeeze())
        # illumination_gamma_corrected = apply_gamma_correction(enhanced_illumination)
        # Recombine the components
        final_image = enhanced_reflectance * enhanced_illumination[:, :, np.newaxis]

        st.image(enhanced_illumination, caption='Enhanced Illumination Component', use_column_width=True)

        st.image(enhanced_reflectance, caption='Enhanced Reflectance Component', use_column_width=True)

        st.image(final_image, caption='Enhanced Image', use_column_width=True)

