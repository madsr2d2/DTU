from skimage.util import img_as_ubyte
from skimage import io
import matplotlib.pyplot as plt
from skimage.morphology import disk, closing, opening
from skimage.filters.rank import mean
from skimage.filters import (
    prewitt,
    threshold_otsu,
    gaussian,
    median,
)

# Constants for visualization
DATA_DIR = "data/"
FIG_WIDTH = 20
AXIS_TITLE_SIZE = 20
FIG_TITLE_SIZE = 30
LEGEND_SIZE = 14


def golden_ratio(width):
    """Calculate height using the golden ratio."""
    return width / ((1 + 5**0.5) / 2)


def morphologyStuff(img_name, disk_size=3):
    closing_kernal = disk(2 * disk_size)
    opening_kernal = disk(disk_size)
    img_org = io.imread(DATA_DIR + img_name, as_gray=True)
    T = threshold_otsu(img_org)
    img_bin = img_org < T
    img_close = closing(img_bin, footprint=closing_kernal)
    img_close_open = opening(img_close, footprint=opening_kernal)
    fig, ax = plt.subplots(
        2, 2, figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )
    fig.suptitle("Morphology operations", fontsize=FIG_TITLE_SIZE)
    ax[0, 0].imshow(img_org, cmap="gray")
    ax[0, 0].set_title(f"Original image", fontsize=AXIS_TITLE_SIZE)

    ax[0, 1].imshow(img_bin, cmap="gray")
    ax[0, 1].set_title(f"Original image\n+ Otsu", fontsize=AXIS_TITLE_SIZE)

    ax[1, 0].imshow(img_close, cmap="gray")
    ax[1, 0].set_title(
        f"Original image\n+ Closing with disk({2*disk_size})", fontsize=AXIS_TITLE_SIZE
    )

    ax[1, 1].imshow(img_close_open, cmap="gray")
    ax[1, 1].set_title(
        f"Original image\n+ Closing with disk({2*disk_size})\n+ Opening with disk({disk_size})",
        fontsize=AXIS_TITLE_SIZE,
    )
    plt.tight_layout()
    plt.show()


morphologyStuff("lego_5.png", 5)
morphologyStuff("puzzle_pieces.png", 5)
