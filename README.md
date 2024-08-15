# Enhancing Low-Light Images Using Retinex Theory - An EECS 6154 Course Project

## Overview

This project focuses on enhancing images captured in low-light conditions using a Retinex-based approach. Low-light images often suffer from noise and reduced visibility, which can impair the performance of various computer vision algorithms. The project introduces a method to improve the quality of such images through illumination map estimation and reflectance estimation, leading to better visual quality and clarity.

## Methodology

### 1. **Illumination Map Estimation**

The initial illumination map is estimated by analyzing the local variance of image neighborhoods. The process involves:
- Calculating the variance within a neighborhood for each pixel.
- Determining neighborhood size based on the variance threshold.
- Estimating the illumination component by using the maximum intensity value for smooth regions and the mean intensity for regions with significant variation.

### 2. **Reflectance Estimation**

Reflectance is estimated using Total Variation (TV) denoising combined with bilateral filter weights. The steps include:
- Defining a discrete gradient operator for the image.
- Applying weighted TV denoising to retain edge information and reduce noise.
- Refining the reflectance component by solving an optimization problem that balances image fidelity and smoothness.

### 3. **Contrast Enhancement**

The final step involves enhancing the contrast of the image using Contrast Limited Adaptive Histogram Equalization (CLAHE), which boosts overall image contrast without amplifying noise.

## Experimental Results

The experimental results demonstrate that the proposed method significantly improves the quality of low-light images. The illumination map estimated through neighborhood variance analysis is more accurate and smoother compared to existing methods. The use of weighted TV in reflectance estimation provides better edge preservation and overall image clarity.

## Key Contributions

- A novel approach to estimate the initial illumination map using pixel neighborhood variance analysis.
- An improved method for reflectance estimation using Total Variation denoising with bilateral filter weights, offering better control over the denoising process.

## References

1. [Etta D Pisano et al., "Contrast limited adaptive histogram equalization image processing to improve the detection of simulated spiculations in dense mammograms," Journal of Digital imaging, 1998.](https://doi.org/10.1007/BF03168750)
2. [Xiaojie Guo, "LIME: A method for low-light image enhancement," 2016.](https://arxiv.org/abs/1511.06079)

For a detailed explanation of the methodology and experimental setup, please refer to the [full project report](path/to/report.pdf).
