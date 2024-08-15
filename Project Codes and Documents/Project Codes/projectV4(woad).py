from decimal import Clamped
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter
from scipy.sparse import eye, spdiags, kron, vstack, diags, csr_matrix
from scipy.sparse.linalg import factorized, splu
import scipy.sparse as sp


def solve_prox_tv_admm(X, Y, mD, paramLambda, paramRho=5, numIterations=100):
    """
    Solves the Prox of the Total Variation (TV) Norm using ADMM Method.
    Args:
    - X (ndarray): Optimization Vector (n x 1).
    - Y (ndarray): Measurements Vector (n x 1).
    - mD (ndarray): Model Matrix (m x n).
    - paramLambda (float): L1 Regularization parameter.
    - paramRho (float): The Rho Parameter for ADMM.
    - numIterations (int): Number of iterations.

    Returns:
    - X (ndarray): Output Vector (n x 1).
    - mX (ndarray): Matrix to store X at each iteration.
    """

    mX = np.zeros((Y.shape[0], numIterations))

    I = eye(Y.shape[0])
    mC = splu((I + paramRho * (mD.T @ mD)).tocsc()).solve  # Pre-factorize for efficiency

    Z = prox_l1(mD @ X, paramLambda / paramRho)
    vU = mD @ X - Z

    mX[:, 0] = X

    for ii in range(1, numIterations):
        X = mC(Y + (paramRho * mD.T @ (Z - vU)))
        Z = prox_l1(mD @ X + vU, paramLambda / paramRho)
        vU = vU + mD @ X - Z

        mX[:, ii] = X

    return X, mX


def prox_l1(v, lambda_param):
    """Proximal operator for L1 norm."""
    return np.sign(v) * np.maximum(np.abs(v) - lambda_param, 0)


def create_gradient_operator(num_rows, num_cols):
    # Horizontal gradient operator
    diagonals_h = np.array([-1 * np.ones(num_cols - 1), np.ones(num_cols - 1)])
    offsets_h = np.array([0, 1])
    mT_h = diags(diagonals_h, offsets_h, shape=(num_cols - 1, num_cols)).tocsr()
    mDh = kron(csr_matrix(np.eye(num_rows)), mT_h, format='csr')

    # Vertical gradient operator
    diagonals_v = np.array([-1 * np.ones(num_rows - 1), np.ones(num_rows - 1)])
    offsets_v = np.array([0, 1])
    mT_v = diags(diagonals_v, offsets_v, shape=(num_rows - 1, num_rows)).tocsr()
    mDv = kron(mT_v, csr_matrix(np.eye(num_cols)), format='csr')

    # Combine both operators
    mD = vstack([mDv, mDh])

    return mD

# Retinex decomposition
# def estimate_illumination_map(image):
#     # Use np.max with axis=2 to find the maximum across the color channels (R, G, B)
#     illumination_map = np.max(image, axis=2)
#     return illumination_map

def calculate_neighborhood_variance(image, x, y, neighborhood_size):
    """
    Calculate the variance of pixel values in the neighborhood of (x, y)
    """
    row_start = max(y - neighborhood_size, 0)
    row_end = min(y + neighborhood_size + 1, image.shape[0])
    col_start = max(x - neighborhood_size, 0)
    col_end = min(x + neighborhood_size + 1, image.shape[1])

    neighborhood = image[row_start:row_end, col_start:col_end]
    variance = np.var(neighborhood, axis=(0, 1))
    return variance

def construct_illumination_component(image, neighborhood_size, variance_threshold):
    """
    Construct the illumination component based on neighborhood variance
    """
    height, width, _ = image.shape
    illumination_component = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            variance = calculate_neighborhood_variance(image, x, y, neighborhood_size)
            if np.all(variance < variance_threshold):
                illumination_component[y, x] = np.max(image[y, x])
            else:
                # Calculate average value of the neighborhood
                row_start = max(y - neighborhood_size, 0)
                row_end = min(y + neighborhood_size + 1, height)
                col_start = max(x - neighborhood_size, 0)
                col_end = min(x + neighborhood_size + 1, width)
                neighborhood = image[row_start:row_end, col_start:col_end]
                illumination_component[y, x] = np.mean(neighborhood)

    return illumination_component


def estimate_reflectance(image, illumination_map):
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        reflectance = image / illumination_map[:, :, None]
        reflectance[~np.isfinite(reflectance)] = 0  # Replace infinities and NaNs with 0
    return reflectance

# TV-ADMM denoising (using skimage's denoise_tv_chambolle as a placeholder)
def tv_admm_denoising(reflectance, weight=0.2):
    denoised = np.zeros_like(reflectance)
    for i in range(reflectance.shape[2]):  # Apply denoising channel-wise
        denoised[:, :, i] = denoise_tv_chambolle(reflectance[:, :, i], weight=weight, channel_axis=False)
    return denoised

# Tikhonov regularization for illumination enhancement
def enhance_illumination_tikh(illumination, lambda_reg=1.0, num_iter=100, learning_rate=0.01):
    enhanced_illumination = np.copy(illumination)

    for _ in range(num_iter):
        grad = 2 * (enhanced_illumination - illumination)  # Data fidelity term
        grad += 2 * lambda_reg * gaussian_filter(enhanced_illumination, sigma=2, order=2)  # Regularization term
        enhanced_illumination -= learning_rate * grad

    return enhanced_illumination

def apply_gamma_correction(image, gamma=1.5):
    # Apply gamma correction
    gamma_corrected = np.power(image, gamma)
    return gamma_corrected


# def enhance_illumination(image):
#     # Adaptive histogram equalization to the illumination map
#     illumination_map_8bit = (image * 255).astype(np.uint8)
#     # equalized_illumination = cv2.equalizeHist(illumination_map_8bit) / 255.0
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized_illumination = clahe.apply(illumination_map_8bit) / 255.0

#     return equalized_illumination

def enhance_illumination(image):
    # Initialize an empty array to store the enhanced channels
    enhanced_channels = []

    # Loop through each channel in the image
    for i in range(3):  # Assuming image has 3 channels (RGB)
        # Extract the current channel
        channel = image[:, :, i]

        # Apply histogram equalization to the channel
        channel_8bit = (channel * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_channel = clahe.apply(channel_8bit) / 255.0

        # Append the enhanced channel to the list
        enhanced_channels.append(equalized_channel)

    # Stack the enhanced channels back into an image
    enhanced_image = np.stack(enhanced_channels, axis=-1)

    return enhanced_image



st.title('Image Illumination and Reflectance Enhancement')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_rgb = cv2.resize(image_rgb, (400, 600), interpolation = cv2.INTER_LINEAR)

    # Process and display the results
    if st.button('Enhance Image'):
        if image is None:
            print("Error loading image")
        else:
            # Convert image to float32 for more precision in calculations
            image_float = np.float32(image_rgb) / 255.0

            # Estimate the illumination map
            # illumination = estimate_illumination_map(image_float)
            neighborhood_size = 3  # Example size, can be adjusted
            variance_threshold = 150  # Example threshold, can be adjusted
            # illumination = estimate_illumination_map(image_float)

            illumination = construct_illumination_component(image_rgb, neighborhood_size, variance_threshold)

            # Estimate the reflectance component
            illumination = np.float32(illumination) / 255.0
            reflectance = estimate_reflectance(image_float, illumination)

        # Enhance the reflectance component using TV-ADMM denoising
        enhanced_reflectance = tv_admm_denoising(reflectance, weight=0.25)
        # enhanced_reflectance = np.clip(enhanced_reflectance, 0, 1)
        
        #Own implementation
        mI = reflectance

        # Add noise to each color channel
        noise_std = 10 / 255
        numRows, numCols, numChannels = mI.shape
        mY = mI + (noise_std * np.random.randn(numRows, numCols, numChannels))

        # Initialize variables for optimization
        delta = 0.025
        numIterations = 250

        # Denoising each color channel separately
        enhanced_reflectance_own = np.zeros_like(mI)
        for channel in range(numChannels):
            Y = mY[:, :, channel].flatten()
            XInit = np.zeros(numRows * numCols)
            div = create_gradient_operator(numRows, numCols)
            X_admm, mX = solve_prox_tv_admm(XInit, Y, div, delta, numIterations)
            enhanced_reflectance_own[:, :, channel] = X_admm.reshape(numRows, numCols)


        # enhanced_illumination = enhance_illumination(illumination.squeeze())
        enhanced_illumination = enhance_illumination_tikh(illumination, lambda_reg=0.25, num_iter=100, learning_rate=0.01)
        
        # illumination_gamma_corrected = apply_gamma_correction(enhanced_illumination)
        
        # Recombine the components
        final_image = enhanced_reflectance * enhanced_illumination[:, :, np.newaxis]
        final_image = enhance_illumination(final_image.squeeze())
        final_image = np.clip(final_image, 0, 1)
        
        #Own code
        final_image_own = enhanced_reflectance_own * enhanced_illumination[:, :, np.newaxis]
        final_image_own = enhance_illumination(final_image_own.squeeze())
        final_image_own = np.clip(final_image_own, 0, 1)
        enhanced_reflectance = np.clip(enhanced_reflectance, 0, 1)
        enhanced_reflectance_own = np.clip(enhanced_reflectance_own, 0, 1)
        enhanced_illumination = np.clip(enhanced_illumination, 0, 1)

        # Display images in a grid
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption='Original Image', use_column_width=True)
        with col2:
            st.image(final_image, caption='Enhanced Image', use_column_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.image(enhanced_reflectance, caption='Enhanced Reflectance Component', use_column_width=True)
        with col4:
            st.image(enhanced_illumination, caption='Enhanced Illumination Component', use_column_width=True)
        
        col5, col6 = st.columns(2)
        with col5:
            st.image(enhanced_reflectance_own, caption='Enhanced Reflectance Component (Own implementation)', use_column_width=True)
        with col6:
            st.image(final_image_own, caption='Enhanced Image (Own implementation)', use_column_width=True)

