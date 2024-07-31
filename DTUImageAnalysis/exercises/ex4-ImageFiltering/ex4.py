import os
from typing import reveal_type
import numpy as np
from scipy.ndimage import correlate, gaussian_filter
from skimage import io
import matplotlib.pyplot as plt
import argparse

from skimage.filters import gaussian
from skimage.util.dtype import img_as_uint

dataDir = (
    "/home/madsrichardt/sem4_summer/DTUImageAnalysis/exercises/ex4-ImageFiltering/data/"
)


def f1():
    input_img = np.arange(25).reshape(5, 5)
    print(f"Input image\n{input_img}")
    weights = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    res_img = correlate(input_img, weights)
    print(f"\nImage after correlation\n{res_img}")

    # Ex1
    val = res_img[3][3]
    print(f"\nValue at pos (3,3) = {val}")

    # Ex2
    res_img = correlate(input_img, weights, mode="constant", cval=10)
    print(f"\nImage after correlation, mode=constant\n{res_img}")

    res_img = correlate(input_img, weights, mode="reflect")
    print(f"\nImage after correlation, mode=reflect\n{res_img}")

    # Ex3
    name = "Gaussian.png"
    imgPath = os.path.join(dataDir, name)
    print(f"Full path to the image: {imgPath}")

    img = io.imread(imgPath, as_gray=True)

    # Create a figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Two-dimensional filter filled with 1
    size = 5
    weights = np.ones((size, size))
    # Normalize weights
    weights = weights / np.sum(weights)

    # Apply mean filter
    imgMeanFilter = correlate(img, weights)

    # Display the filtered image
    ax[1].imshow(imgMeanFilter, cmap="gray")
    ax[1].set_title("Filtered Image (Mean Filter)")
    ax[1].axis("off")

    # Show the plots
    plt.show()


# Edge detection
def f8():
    from skimage.filters import prewitt_h
    from skimage.filters import prewitt_v
    from skimage.filters import prewitt

    imgName = "donald_1.png"
    imgPath = dataDir + imgName
    imgOrg = io.imread(imgPath, as_gray=True)
    imgPrewitt = prewitt(imgOrg)

    print(
        f"Max pixel value {max(imgPrewitt.flatten())}\nMin pixel value {min(imgPrewitt.flatten())}"
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(imgOrg, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(imgPrewitt, cmap="gray")
    ax[1].set_title("Image after prewitt filter")
    ax[1].axis("off")

    plt.show()


def f9():
    from skimage.filters import prewitt, gaussian, threshold_otsu, median
    from skimage import exposure
    from skimage.morphology import disk, square

    imgName = "ElbowCTSlice.png"
    imgPath = dataDir + imgName
    imgOrg = io.imread(imgPath, as_gray=True)

    imgEq = exposure.equalize_hist(imgOrg)
    imgFiltered = median(imgEq, disk(3))
    imgPrewitt = prewitt(imgFiltered)
    T_otzu = threshold_otsu(imgPrewitt)
    imgBin = imgPrewitt > T_otzu

    fig, ax = plt.subplots(2, 2, figsize=(24, 12))

    ax[0][0].imshow(imgOrg, cmap="gray")
    ax[0][0].set_title("Original Image")
    ax[0][0].axis("off")

    ax[0][1].imshow(imgFiltered, cmap="gray")
    ax[0][1].set_title("Image after equalize_hist and median filter")
    ax[0][1].axis("off")

    ax[1][0].imshow(imgPrewitt, cmap="gray")
    ax[1][0].set_title("Image after prewitt filter")
    ax[1][0].axis("off")

    ax[1][1].imshow(imgBin, cmap="gray")
    ax[1][1].set_title("Image after Otsu threshold on Prewitt image")
    ax[1][1].axis("off")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a specific function based on the argument."
    )
    parser.add_argument(
        "func_number", type=int, help="The function number to run (e.g., 1 for f1)."
    )
    args = parser.parse_args()

    func_name = f"f{args.func_number}"
    if func_name in globals():
        globals()[func_name]()
    else:
        print(f"No function named {func_name} found.")
