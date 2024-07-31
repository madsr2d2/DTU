from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk, square
import matplotlib.pyplot as plt
import argparse
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import disk, erosion, dilation, closing, opening
from skimage.exposure import equalize_adapthist
from skimage.segmentation import find_boundaries
import numpy as np

dataDir = "/home/madsrichardt/DTU/DTUImageAnalysis/exercises/ex4b-ImageMorphology/data/"


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(24, 12), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title("original")
    ax1.axis("off")
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis("off")
    io.show()


def plotImageHistogram(grayScaleImage):
    plt.figure()  # Create a new figure
    plt.hist(grayScaleImage.ravel(), bins=256, range=(0, 1), fc="k", ec="k")
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


def ex1():
    imgName = "lego_5.png"
    imgPath = dataDir + imgName
    imgOrg = io.imread(imgPath, as_gray=True)
    otzuThreshold = threshold_otsu(imgOrg)
    imgBin = imgOrg < otzuThreshold
    plot_comparison(imgOrg, imgBin, "Binary image")


def ex2():
    footprint = disk(7)
    print(footprint)

    # Load image
    imgName = "lego_5.png"
    imgPath = dataDir + imgName
    imgOrg = io.imread(imgPath, as_gray=True)

    # Equalize image
    # imgOrgEq = equalize_adapthist(imgOrg)
    # plotImageHistogram(imgOrg)
    # plotImageHistogram(imgOrgEq)

    # Find threshold and convert image to binary
    # The Histogram equalization does not improve the result so don't use
    otzuThreshold = threshold_otsu(imgOrg)
    imgBin = imgOrg < otzuThreshold

    # Close image
    imgClose = closing(imgBin, footprint)

    # Open image
    imgOpen = opening(imgClose, footprint)

    # plot
    plot_comparison(imgOrg, imgOpen, "Processed image")


def ex9():
    footprint = disk(12)
    print(footprint)

    # Load image
    imgName = "puzzle_pieces.png"
    imgPath = dataDir + imgName
    imgOrg = io.imread(imgPath, as_gray=True)

    # Equalize image
    imgOrgEq = equalize_adapthist(imgOrg)
    plotImageHistogram(imgOrg)
    plotImageHistogram(imgOrgEq)

    # Find threshold and convert image to binary
    # The Histogram equalization does not improve the result so don't use
    otzuThreshold = threshold_otsu(imgOrgEq)
    imgBin = imgOrgEq < otzuThreshold

    # Close image
    imgClose = closing(imgBin, footprint)

    # Open image
    imgOpen = opening(imgClose, footprint)

    # Get boundries
    boundries = find_boundaries(imgOpen)

    # plot
    plot_comparison(imgBin, imgOpen, "Processed image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a specific function based on the argument."
    )
    parser.add_argument(
        "func_number", type=int, help="The function number to run (e.g., 1 for f1)."
    )
    args = parser.parse_args()

    func_name = f"ex{args.func_number}"
    if func_name in globals():
        globals()[func_name]()
    else:
        print(f"No function named {func_name} found.")
