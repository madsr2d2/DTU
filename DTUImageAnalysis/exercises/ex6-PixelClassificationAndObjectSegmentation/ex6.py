import matplotlib.pyplot as plt
from scipy.stats import norm
from skimage import io, exposure
import numpy as np
import pydicom as dicom
from matplotlib.colors import ListedColormap

# Load CT image
dataDir = "data/"
ct = dicom.dcmread(dataDir + "Training.dcm")
img = ct.pixel_array


def generate_colormap(num_classes):
    return [plt.get_cmap("tab10")(i) for i in range(num_classes)]


def combine_masks(mask_list, dataDir):
    combined_mask = np.zeros(io.imread(dataDir + mask_list[0]).shape, dtype=bool)
    for mask_name in mask_list:
        combined_mask |= io.imread(dataDir + mask_name) > 0
    return combined_mask


def overlay_rois_with_labels(mask_list, img, dataDir, colors):
    img_rgb = np.dstack([img] * 3)  # Convert grayscale to RGB
    img_rgb = exposure.rescale_intensity(img_rgb, out_range=(0, 1))

    labeled_img = np.zeros_like(img, dtype=int)
    for i, masks in enumerate(mask_list):
        labeled_img[combine_masks(masks, dataDir)] = i + 1

    for i, color in enumerate(colors[: len(mask_list)]):
        img_rgb[labeled_img == i + 1] = color[:3]  # Apply the color to each channel

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_rgb)
    ax.set_title("Image with ROIs")
    ax.axis("off")

    class_names = [
        "_".join(elm.split("ROI")[0] for elm in masks) for masks in mask_list
    ]
    legend_patches = [
        plt.Line2D([0], [0], color=color[:3], lw=4)
        for color in colors[: len(mask_list)]
    ]
    plt.legend(
        legend_patches,
        class_names,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Class Labels",
    )

    plt.show()


def label_and_visualize_image(img, lookup_table, colors):
    labeled_img = np.zeros_like(img, dtype=int)
    class_name_to_label = {
        name: idx + 1 for idx, name in enumerate(sorted(set(lookup_table.values())))
    }

    for val in np.unique(img):
        if val in lookup_table:
            labeled_img[img == val] = class_name_to_label[lookup_table[val]]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    unique_classes = np.unique(labeled_img)
    cmap = ListedColormap(colors[: len(unique_classes)])
    im = axes[1].imshow(labeled_img, cmap=cmap)
    axes[1].set_title("Labeled Image")
    axes[1].axis("off")

    class_names = sorted(set(lookup_table.values()))
    cbar = plt.colorbar(im, ax=axes[1], ticks=range(1, len(class_names) + 1))
    cbar.ax.set_yticklabels(class_names)

    plt.show()

    return labeled_img


def plot_class_data_and_distributions(mask_list, img, dataDir, colors):
    plt.figure(figsize=(15, 9))

    for i, masks in enumerate(mask_list):
        combined_mask = combine_masks(masks, dataDir)
        values = img[combined_mask]
        mean, sd = values.mean(), values.std()

        class_name = "_".join(elm.split("ROI")[0] for elm in masks)
        color = colors[i % len(colors)]
        plt.hist(
            values.ravel(),
            bins=60,
            density=True,
            alpha=0.6,
            color=color,
            label=f"Class: {class_name}",
        )

        x = np.linspace(values.min(), values.max(), 100)
        pdf = norm.pdf(x, mean, sd)
        plt.plot(x, pdf, color=color, linestyle="--")

    plt.xlabel("Hounsfield unit")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Histogram and Fitted Normal Distributions of Class Data")
    plt.show()


def minDistanceClassifier(classList, dataDir, img):
    meanLst, nameLst = [], []
    imgValues = img.ravel()
    HUmin, HUmax = imgValues.min(), imgValues.max()

    for masks in classList:
        combined_mask = combine_masks(masks, dataDir)
        values = img[combined_mask]
        meanLst.append(values.mean())
        nameLst.append("_".join(elm.split("ROI")[0] for elm in masks))

    lookup_table = {
        val: nameLst[np.argmin([abs(val - mean) for mean in meanLst])]
        for val in range(int(HUmin), int(HUmax) + 1)
    }
    return lookup_table


def parametricClassifier(classList, dataDir, img):
    meanLst, sdLst, nameLst = [], [], []
    imgValues = img.ravel()
    HUmin, HUmax = imgValues.min(), imgValues.max()

    for masks in classList:
        combined_mask = combine_masks(masks, dataDir)
        values = img[combined_mask]
        meanLst.append(values.mean())
        sdLst.append(values.std())
        nameLst.append("_".join(elm.split("ROI")[0] for elm in masks))

    lookup_table = {
        val: nameLst[
            np.argmax([norm.pdf(val, mean, sd) for mean, sd in zip(meanLst, sdLst)])
        ]
        for val in range(int(HUmin), int(HUmax) + 1)
    }
    return lookup_table


def bayesianClassifier(classList, dataDir, img):
    meanLst, sdLst, nameLst, priorLst = [], [], [], []
    imgValues = img.ravel()
    HUmin, HUmax = imgValues.min(), imgValues.max()

    total_pixels = img.size
    for masks in classList:
        combined_mask = combine_masks(masks, dataDir)
        values = img[combined_mask]
        meanLst.append(values.mean())
        sdLst.append(values.std())
        nameLst.append("_".join(elm.split("ROI")[0] for elm in masks))
        priorLst.append(values.size / total_pixels)

    lookup_table = {}
    for val in range(int(HUmin), int(HUmax) + 1):
        posteriors = [
            prior * norm.pdf(val, mean, sd)
            for mean, sd, prior in zip(meanLst, sdLst, priorLst)
        ]
        lookup_table[val] = nameLst[np.argmax(posteriors)]
    return lookup_table


# Example usage:
classList1 = [
    ["BoneROI.png"],
    ["KidneyROI.png"],
    ["SpleenROI.png"],
    ["LiverROI.png"],
    ["FatROI.png"],
    ["BackgroundROI.png"],
]

colors = generate_colormap(len(classList1))

overlay_rois_with_labels(classList1, img, dataDir, colors)
plot_class_data_and_distributions(classList1, img, dataDir, colors)
lookup_table = bayesianClassifier(classList1, dataDir, img)
label_and_visualize_image(img, lookup_table, colors)
