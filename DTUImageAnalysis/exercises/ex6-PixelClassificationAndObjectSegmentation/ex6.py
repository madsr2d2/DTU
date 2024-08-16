import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import norm
from skimage import io, exposure, measure
from skimage.morphology import disk, opening, closing
import numpy as np
import pydicom as dicom


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


def find_class_bounds(lookup_table, target_class):
    class_values = [val for val, cls in lookup_table.items() if cls == target_class]

    if not class_values:
        return None

    min_value = min(class_values)
    max_value = max(class_values)

    return min_value, max_value


def spleen_criteria(region):
    expected_area = 3457.0
    expected_orientation = -0.38
    expected_eccentricity = 0.91
    expected_solidity = 0.92
    orientation_std = 0.15
    eccentricity_std = 0.15
    area_std = expected_area * 0.15
    solidity_std = 0.15

    # Orientation score
    orientation_weight = 1.0
    orientation_deviation = np.abs(region.orientation - expected_orientation)
    orientation_score = (
        np.exp(-orientation_deviation / orientation_std) * orientation_weight
    )

    # Eccentricity score
    eccentricity_weight = 1.0
    eccentricity_deviation = np.abs(region.eccentricity - expected_eccentricity)
    eccentricity_score = (
        np.exp(-eccentricity_deviation / eccentricity_std) * eccentricity_weight
    )

    # Size score
    size_weight = 1.0
    size_deviation = np.abs(region.area - expected_area)
    size_score = np.exp(-size_deviation / area_std) * size_weight

    # Solidity score
    solidity_weight = 1.0
    solidity_deviation = np.abs(
        region.solidity - expected_solidity
    )  # Solidity should be close to 1 for compact regions
    solidity_score = np.exp(-solidity_deviation / solidity_std) * solidity_weight

    # Combine scores: higher scores are better
    total_score = orientation_score + eccentricity_score + solidity_score + size_score

    return total_score


def spleen_finder(img_list, lookup_table, disk_size=3):
    if not isinstance(img_list, list):
        img_list = [img_list]  # Convert to list if a single image is passed

    fig, axes = plt.subplots(1, len(img_list), figsize=(15, 10))

    if len(img_list) == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one image

    for i, img in enumerate(img_list):
        spleen_min, spleen_max = find_class_bounds(lookup_table, "Spleen")
        spleen_estimate = (img > spleen_min) & (img < spleen_max)

        footprint = disk(disk_size)
        closed = closing(spleen_estimate, footprint)
        opened = opening(closed, footprint)

        label_img = measure.label(opened)
        region_props = measure.regionprops(label_img)

        if not region_props:
            print(f"No regions found in image {i+1}.")
            continue

        spleen_blob = max(region_props, key=spleen_criteria)
        spleen_region = label_img == spleen_blob.label

        img_rgb = np.dstack([img] * 3)
        img_rgb = exposure.rescale_intensity(img_rgb, out_range=(0, 1))
        img_rgb[spleen_region] = [1, 0, 0]  # Red color for spleen region

        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Image {i+1}: Spleen Overlay")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    return spleen_region if len(img_list) == 1 else None


dataDir = "data/"
image_names = ["Training.dcm", "Validation1.dcm", "Validation2.dcm", "Validation3.dcm"]
img_list = [dicom.dcmread(dataDir + imgName).pixel_array for imgName in image_names]
img = img_list[0]

# Example usage:
classList = [
    ["BoneROI.png"],
    ["KidneyROI.png"],
    ["SpleenROI.png"],
    ["LiverROI.png"],
    ["FatROI.png"],
    ["TrabecROI.png"],
    ["BackgroundROI.png"],
]

colors = generate_colormap(len(classList))
overlay_rois_with_labels(classList, img, dataDir, colors)
plot_class_data_and_distributions(classList, img, dataDir, colors)
lookup_table = parametricClassifier(classList, dataDir, img)
label_and_visualize_image(img, lookup_table, colors)
spleen_region = spleen_finder(img_list, lookup_table, disk_size=3)
