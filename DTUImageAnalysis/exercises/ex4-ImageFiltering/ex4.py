from skimage import io
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import mean
from skimage.filters import (
    prewitt_h,
    prewitt_v,
    prewitt,
    threshold_otsu,
    gaussian,
    median,
)

from data.Ex4_VideoImageFiltering import capture_from_camera_and_show_images

# Constants for visualization
DATA_DIR = "data/"
FIG_WIDTH = 20
AXIS_TITLE_SIZE = 20
FIG_TITLE_SIZE = 30
LEGEND_SIZE = 14


def golden_ratio(width):
    """Calculate height using the golden ratio."""
    return width / ((1 + 5**0.5) / 2)


def filtering(img_name):

    img = io.imread(DATA_DIR + img_name, as_gray=True)
    # mean, median, gaussian with different footprint size
    disk_list = []
    for i in range(3):
        disk_list.append(disk((i + 1) * 3))
    fig, ax = plt.subplots(
        3, 3, figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )
    fig.suptitle("Image filtering", fontsize=FIG_TITLE_SIZE)
    for i in range(3):
        ax[0, i].imshow(mean(img, disk_list[i]), cmap="gray")
        ax[0, i].set_title(
            f"Mean filter with disk({(i+1)*3})", fontsize=AXIS_TITLE_SIZE
        )

        ax[1, i].imshow(median(img, disk_list[i]), cmap="gray")
        ax[1, i].set_title(
            f"Median filter with disk({(i+1)*3})", fontsize=AXIS_TITLE_SIZE
        )

        ax[2, i].imshow(gaussian(img, sigma=((i + 1) * 3)), cmap="gray")
        ax[2, i].set_title(
            f"Gaussian filter with sigma = {(i+1)*3}", fontsize=AXIS_TITLE_SIZE
        )

    plt.tight_layout()
    plt.show()


def edge_filtering(img_name):
    img = io.imread(DATA_DIR + img_name, as_gray=True)

    fig, ax = plt.subplots(
        2, 2, figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )
    fig.suptitle("Edge filtering", fontsize=FIG_TITLE_SIZE)
    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 0].set_title("Original image", fontsize=AXIS_TITLE_SIZE)

    ax[0, 1].imshow(prewitt_h(img), cmap="gray")
    ax[0, 1].set_title(r"Image filtered the prewitt_h filter", fontsize=AXIS_TITLE_SIZE)

    ax[1, 0].imshow(prewitt_v(img), cmap="gray")
    ax[1, 0].set_title(r"Image filtered the prewitt_v filter", fontsize=AXIS_TITLE_SIZE)

    ax[1, 1].imshow(prewitt(img), cmap="gray")
    ax[1, 1].set_title(r"Image filtered the prewitt filter", fontsize=AXIS_TITLE_SIZE)

    plt.tight_layout()
    plt.show()


def edge_filtering1(img_name, filter_type=0):
    img = io.imread(DATA_DIR + img_name, as_gray=True)

    if filter_type == 0:
        img_filtered = median(img, disk(3))
    elif filter_type == 1:
        img_filtered = gaussian(img, sigma=2)
    else:
        img_filtered = gaussian(img, sigma=2)
        img_filtered = median(img_filtered, disk(3))

    img_gradients = prewitt(img_filtered)
    T = threshold_otsu(img_gradients)
    img_bin = img_gradients > T

    fig, ax = plt.subplots(
        2, 2, figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )
    fig.suptitle("Edge filtering", fontsize=FIG_TITLE_SIZE)
    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 0].set_title(f"Original image ({img_name})", fontsize=AXIS_TITLE_SIZE)

    ax[0, 1].imshow(img_filtered, cmap="gray")
    if filter_type == 0:
        ax[0, 1].set_title(r"Median", fontsize=AXIS_TITLE_SIZE)
    elif filter_type == 1:
        ax[0, 1].set_title(r"Gaussian", fontsize=AXIS_TITLE_SIZE)
    else:
        ax[0, 1].set_title(r"Gaussian + Median ", fontsize=AXIS_TITLE_SIZE)

    ax[1, 0].imshow(img_gradients, cmap="gray")
    if filter_type == 0:
        ax[1, 0].set_title(r"Median + Prewitt ", fontsize=AXIS_TITLE_SIZE)
    elif filter_type == 1:
        ax[1, 0].set_title(r"Gaussian + Prewitt ", fontsize=AXIS_TITLE_SIZE)
    else:
        ax[1, 0].set_title(r"Gaussian + Median + Prewitte ", fontsize=AXIS_TITLE_SIZE)

    ax[1, 1].imshow(img_bin, cmap="gray")
    if filter_type == 0:
        ax[1, 1].set_title(r"Median + Prewitt + Otsu", fontsize=AXIS_TITLE_SIZE)
    elif filter_type == 1:
        ax[1, 1].set_title(r"Gaussian + Prewitt + Otsu", fontsize=AXIS_TITLE_SIZE)
    else:
        ax[1, 1].set_title(
            r"Gaussian + Median + Prewitte + Otsu", fontsize=AXIS_TITLE_SIZE
        )

    plt.tight_layout()
    plt.show()


filtering("car.png")
edge_filtering("donald_1.png")
edge_filtering1("ElbowCTSlice.png", filter_type=0)
capture_from_camera_and_show_images()
