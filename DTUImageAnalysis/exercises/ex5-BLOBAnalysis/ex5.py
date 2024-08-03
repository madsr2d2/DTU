from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import lab2rgb, label2rgb, rgb2gray
from skimage.morphology import disk, opening, closing
import argparse


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis("off")
    plt.show()


dataDir = "/home/madsrichardt/DTU/DTUImageAnalysis/exercises/ex5-BLOBAnalysis/data/"


def ex1():
    imgName = "lego_4_small.png"
    imgPath = dataDir + imgName
    imgOrg = io.imread(imgPath)
    imgGray = rgb2gray(imgOrg)  # Use rgb2gray to convert to grayscale
    imgGray = img_as_float(imgGray)  # Convert to float32
    otsuThreshold = threshold_otsu(imgGray)
    imgBin = imgGray < otsuThreshold

    # Ex 2
    imgBin = segmentation.clear_border(imgBin)

    # Ex 3
    footprint = disk(5)
    imgBin = closing(imgBin, footprint)
    imgBin = opening(imgBin, footprint)

    # Ex 4,5
    imgBinLables = measure.label(imgBin)
    # imgBinLables = label2rgb(imgBinLables)
    # nLables = imgBinLables.max()
    # print(f"Number of lables: {nLables}")

    # Ex 6

    region_props = measure.regionprops(imgBinLables)
    print([prop for prop in region_props[1]])
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas, bins=50)
    plt.show()

    # show_comparison(imgOrg, imgBinLables, "Binary image")


def ex2():
    in_dir = dataDir
    img_org = io.imread(in_dir + "Sample E2 - U2OS DAPI channel.tiff")
    # slice to extract smaller image
    img_small = img_org[700:1200, 900:1400]
    img_gray = img_as_ubyte(img_small)
    io.imshow(img_gray, vmin=0, vmax=150)
    plt.title("DAPI Stained U2OS cell nuclei")
    io.show()


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
