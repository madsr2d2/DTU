import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, norm
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


def get_masks():
    mask_list = [
        ["BoneROI.png"],
        ["SpleenROI.png", "LiverROI.png", "KidneyROI.png"],
        ["FatROI.png"],
    ]

    return mask_list


def get_pdfs(mask_list):
    plt.figure(figsize=(15, 9))
    res = []
    for masks in mask_list:
        combined_mask = None
        for elm in masks:
            roi = io.imread(dataDir + elm)
            mask = roi > 0
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = np.logical_or(combined_mask, mask)

        values = img[combined_mask]
        mean = values.mean()
        sd = values.std()
        res.append((mean, sd))

        # Plot the histogram with density and the PDF
        n, bins, patches = plt.hist(
            values.ravel(), bins=60, density=True, alpha=0.6, label=" & ".join(masks)
        )
        pdf = norm.pdf(bins, mean, sd)
        plt.plot(bins, pdf, label=f'PDF of {" & ".join(masks)}')

    plt.xlabel("Hounsfield unit")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram and PDF of ROIs")
    plt.show()
    return res


# Load and process the spleen ROI
rois = ["BoneROI.png", "FatROI.png", "LiverROI.png", "KidneyROI.png"]
# plot_histograms_and_pdfs(rois)

# Get masks and PDFs
mask_list = get_masks()
get_pdfs(mask_list)
