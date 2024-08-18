from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage import io, color
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.filters import threshold_otsu

# Constants for visualization
DATA_DIR = "data/"
FIG_WIDTH = 20
AXIS_TITLE_SIZE = 20
FIG_TITLE_SIZE = 30
LEGEND_SIZE = 14


def golden_ratio(width):
    """Calculate height using the golden ratio."""
    return width / ((1 + 5**0.5) / 2)


def pixelwiseOps(img_name):
    img_rgb = io.imread(DATA_DIR + img_name)

    img_hsv = color.rgb2hsv(img_rgb)
    img_gray = color.rgb2gray(img_rgb)
    img_gray = img_as_ubyte(img_gray)

    Tr = threshold_otsu(img_rgb[:, :, 0])
    Tg = threshold_otsu(img_rgb[:, :, 1])
    Tb = threshold_otsu(img_rgb[:, :, 2])

    mask_otsu = (img_rgb[:, :, 2]) < Tb & (img_rgb[:, :, 0] < Tr) & (
        img_rgb[:, :, 1] < Tg
    )
    h_comp = img_hsv[:, :, 0]
    s_comp = img_hsv[:, :, 1]

    # Adjusted ranges for blue detection
    sign_blue = (h_comp > 0.55) & (h_comp < 0.65) & (s_comp > 0.90) | (h_comp > 0.96)

    # Create subplots
    fig, ax = plt.subplots(
        2, 3, figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )
    fig.suptitle("Image Thresholding", fontsize=FIG_TITLE_SIZE)

    # Display the original image
    ax[0, 0].imshow(img_rgb)
    ax[0, 0].set_title("RGB Image", fontsize=AXIS_TITLE_SIZE)

    # Plot histogram of the color channel
    ax[0, 1].hist(
        img_rgb[:, :, 0].ravel(), bins=256, color="r", alpha=0.5, label="R channel"
    )
    ax[0, 1].axvline(
        x=Tr, color="r", linestyle="--", linewidth=3, label="R channel Otsu threshold"
    )

    ax[0, 1].hist(
        img_rgb[:, :, 1].ravel(), bins=256, color="g", alpha=0.5, label="G channel"
    )
    ax[0, 1].axvline(
        x=Tg, color="g", linestyle="--", linewidth=3, label="G channel Otsu threshold"
    )

    ax[0, 1].hist(
        img_rgb[:, :, 2].ravel(), bins=256, color="b", alpha=0.5, label="B channel"
    )
    ax[0, 1].set_title("RGB Image Histogram", fontsize=AXIS_TITLE_SIZE)
    ax[0, 1].axvline(
        x=Tb, color="b", linestyle="--", linewidth=3, label="B channel Otsu threshold"
    )
    ax[0, 1].legend(fontsize=LEGEND_SIZE)
    ax[0, 2].imshow(mask_otsu, cmap="gray")
    ax[0, 2].set_title(
        "R channel < Otsu threshold\nG channel < Otsu threshold\nB channel < Otsu threshold",
        fontsize=AXIS_TITLE_SIZE,
    )

    ax[1, 0].imshow(img_hsv, cmap="hsv")
    ax[1, 0].set_title("HSV Image", fontsize=AXIS_TITLE_SIZE)

    ax[1, 1].hist(
        img_hsv[:, :, 0].ravel(), bins=256, color="r", alpha=0.5, label="H channel"
    )

    ax[1, 1].hist(
        img_hsv[:, :, 1].ravel(), bins=256, color="g", alpha=0.5, label="S channel"
    )

    ax[1, 1].hist(
        img_hsv[:, :, 2].ravel(), bins=256, color="b", alpha=0.5, label="V channel"
    )
    ax[1, 1].set_title("HSV Image Histogram", fontsize=AXIS_TITLE_SIZE)
    ax[1, 1].legend(fontsize=LEGEND_SIZE)

    ax[1, 2].imshow(sign_blue, cmap="gray")
    ax[1, 2].set_title(
        f"Blue/Red  Mask\nH > 0.55 & H < 0.65 & S > 0.90 | h_comp > 0.96",
        fontsize=AXIS_TITLE_SIZE,
    )

    plt.tight_layout()
    plt.show()


pixelwiseOps("DTUSigns2.jpg")
