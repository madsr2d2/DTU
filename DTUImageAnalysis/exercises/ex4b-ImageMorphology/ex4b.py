from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
import matplotlib.pyplot as plt
import argparse
from skimage import io
from skimage.filters import threshold_otsu

dataDir = "/home/madsrichardt/DTU/DTUImageAnalysis/exercises/ex4b-ImageMorphology/data/"


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title("original")
    ax1.axis("off")
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis("off")
    io.show()


def ex1():
    imgName = "lego_5.png"
    imgPath = dataDir + imgName
    imgOrg = io.imread(imgPath, as_gray=True)
    otzuThreshold = threshold_otsu(imgOrg)
    imgBin = imgOrg > otzuThreshold
    plot_comparison(imgOrg, imgBin, "Binary image")


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
