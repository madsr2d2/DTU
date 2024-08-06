import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from skimage import io
import pydicom as dicom

# Load and display the CT image
dataDir = "data/"
ct = dicom.read_file(dataDir + "Training.dcm")
img = ct.pixel_array
print(img.shape)
print(img.dtype)

io.imshow(img, vmin=0, vmax=150, cmap="gray")
io.show()

# Load and process the spleen ROI
rois = ["BoneROI.png", "FatROI.png", "LiverROI.png", "KidneyROI.png"]

for elm in rois:
    roi = io.imread(dataDir + elm)
    mask = roi > 0
    values = img[mask]
    mean = values.mean()
    sd = values.std()
    print(f"mean: {mean}, SD: {sd}")

    # Plot the histogram with density and the PDF
    n, bins, patches = plt.hist(values.ravel(), bins=60, density=True)
    pdf_spleen = norm.pdf(bins, mean, sd)  # Corrected here
    plt.plot(bins, pdf_spleen)
    plt.xlabel("Hounsfield unit")
    plt.ylabel("Frequency")
    plt.title(elm)
    plt.show()


def addNumbers(*args):
    """This function takes any number of arguments and returns their sum."""
    return sum(args)
