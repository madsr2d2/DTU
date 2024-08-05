from skimage import io, color
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.stats import norm
from scipy.spatial import distance


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.imshow(original, cmap="gray", vmin=-200, vmax=500)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis("off")
    io.show()


dataDir = "data/"
ct = dicom.read_file(dataDir + "Training.dcm")
img = ct.pixel_array
print(img.shape)
print(img.dtype)

io.imshow(img, vmin=0, vmax=150, cmap="gray")
io.show()

spleen_roi = io.imread(dataDir + "SpleenROI.png")
# convert to boolean image
spleen_mask = spleen_roi > 0
spleen_values = img[spleen_mask]
meanValus = spleen_values.mean()
spleen_valuesSD = spleen_values.std()
print(f"mean: {meanValus}, SD: {spleen_valuesSD}")
