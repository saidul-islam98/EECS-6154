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
from scipy.sparse import eye, spdiags, kron, vstack
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


# def prox_l1(X, lambdaFactor):
#     """
#     Soft thresholding operation for L1 norm.

#     Arguments:
#     X -- Input vector.
#     lambdaFactor -- Lambda factor for thresholding.

#     Returns:
#     X -- Thresholded vector.
#     """
#     return np.maximum(X - lambdaFactor, 0) + np.minimum(X + lambdaFactor, 0)

def create_gradient_operator(num_rows, num_cols):
    """
    Generates a convolution matrix for the 2D gradient operation.

    Arguments:
    num_rows -- Number of rows of the image to be convolved.
    num_cols -- Number of columns of the image to be convolved.

    Returns:
    div -- Convolution matrix for the 2D gradient operation.
    """
    diagonals_v = [np.ones(numRows - 1), -np.ones(numRows - 1)]
    mT_v = spdiags(diagonals_v, [0, 1], numRows - 1, numRows)
    mDv = kron(np.eye(numCols), mT_v)

    # Horizontal Operator - T(numCols)
    diagonals_h = [np.ones(numCols - 1), -np.ones(numCols - 1)]
    mT_h = spdiags(diagonals_h, [0, 1], numCols - 1, numCols)
    mDh = kron(mT_h, np.eye(numRows))

    # Combine both operators
    mD = vstack([mDv, mDh])

    # print(mD)
    return mD

# Retinex decomposition
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
    # st.image(image_rgb, caption='Original Image', use_column_width=True)
    
    


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

        enhanced_reflectance_own = np.clip(enhanced_reflectance_own, 0.0, 1.0)

        enhanced_illumination = enhance_illumination(illumination.squeeze())
        enhanced_illumination = enhance_illumination_tikh(enhanced_illumination, lambda_reg=1.0, num_iter=100, learning_rate=0.01)
        # illumination_gamma_corrected = apply_gamma_correction(enhanced_illumination)
        # Recombine the components
        final_image = enhanced_reflectance * enhanced_illumination[:, :, np.newaxis]
        #OWn code
        final_image_own = enhanced_reflectance_own * enhanced_illumination[:, :, np.newaxis]

        # st.image(enhanced_illumination, caption='Enhanced Illumination Component', use_column_width=True)

        # st.image(enhanced_reflectance, caption='Enhanced Reflectance Component', use_column_width=True)

        # st.image(final_image, caption='Enhanced Image', use_column_width=True)

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

