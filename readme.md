# Image Compression Using Wavelet Transform

An image compression project that applies 2D Discrete Wavelet Transform (DWT) to evaluate and compare different wavelet types and threshold values for compressing grayscale images.

## Overview

This project implements wavelet-based image compression and quality evaluation. It tests multiple wavelet families and threshold levels to analyze the trade-off between compression ratio (sparsity) and image quality (PSNR). The results are visualized as trade-off plots for each test image.

## Features

- Converts color images to grayscale for processing
- Applies 2D DWT using three wavelet types: **Haar**, **Daubechies-2 (db2)**, and **Symlet-4 (sym4)**
- Compresses images by zeroing wavelet coefficients below a threshold
- Reconstructs compressed images using the Inverse DWT (IDWT)
- Evaluates compression quality using:
  - **MSE** (Mean Squared Error)
  - **PSNR** (Peak Signal-to-Noise Ratio, in dB)
  - **Sparsity** (percentage of zeroed coefficients)
- Plots quality vs. compression trade-off curves per image
- Visualizes wavelet subbands: LL (Approximation), LH (Horizontal), HL (Vertical), HH (Diagonal)

## Project Structure

```
itsc_wavelet_project/
├── wavelet_utils.py        # Main script with all functions and entry point
├── Group_7_Report.pdf      # Academic project report
├── digital_checkered.png   # Test image
├── satellite_image.jpg     # Test image
├── normal.jpg              # Test image
├── pattern.jpg             # Test image
├── forest.jpg              # Test image
└── aybu.jpg                # Test image (used for subband visualization)
```

## Requirements

- Python 3.x
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyWavelets](https://pywavelets.readthedocs.io/)

Install all dependencies with:

```bash
pip install numpy matplotlib PyWavelets
```

## Usage

Run the main script from the project directory:

```bash
python wavelet_utils.py
```

This will:
1. Process four test images: `digital_checkered.png`, `satellite_image.jpg`, `normal.jpg`, `pattern.jpg`
2. For each image, apply three wavelet types (Haar, db2, sym4) across seven threshold values (0.05 to 0.35)
3. Display trade-off plots showing PSNR (quality) vs. Sparsity (compression) for each image
4. Display wavelet subband decomposition for `aybu.jpg` using the Haar wavelet

## How It Works

### Pipeline

1. **Preprocessing** (`prep_img`): Reads an image, converts it to grayscale using luminance weights, and normalizes pixel values to `[0, 1]`.

2. **Wavelet Transform** (`perform_wavelet`): Applies a single-level 2D DWT using `pywt.dwt2`, producing one approximation subband (LL) and three detail subbands (LH, HL, HH).

3. **Thresholding** (`apply_threshold`): Zeroes out all coefficients with absolute values below the threshold `T`, increasing sparsity (compression).

4. **Reconstruction** (`reconstruct`): Applies the Inverse DWT (`pywt.idwt2`) to the thresholded coefficients to recover the compressed image.

5. **Quality Evaluation** (`calculate_mse_psnr`): Computes MSE and PSNR between the original and reconstructed image.

6. **Visualization** (`print_tradeoff_tables`, `show_subbands_of_wavelet`): Plots the PSNR vs. Sparsity trade-off for each wavelet and displays the four wavelet subbands.

### Threshold Values

| Threshold | Effect |
|-----------|--------|
| 0.05 | Low compression, high quality |
| 0.35 | High compression, lower quality |

The yellow horizontal line in trade-off plots marks **30 dB PSNR**, a commonly accepted threshold for acceptable image quality.

## Wavelet Types Compared

| Wavelet | Name | Characteristics |
|---------|------|-----------------|
| `haar` | Haar | Simplest wavelet; sharp transitions |
| `db2` | Daubechies-2 | Smoother than Haar; better frequency localization |
| `sym4` | Symlet-4 | Near-symmetric; good for natural images |

## Academic Report

See `Group_7_Report.pdf` for the full project report, including methodology, experimental results, and analysis.

