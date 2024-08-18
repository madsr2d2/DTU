import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import norm
from skimage import io, exposure, measure, color
from skimage.morphology import disk, opening, closing
from skimage.filters import median, gaussian
import numpy as np
import pydicom as dicom

from matplotlib.lines import Line2D

FIG_WIDTH = 20
AXIS_TITLE_SIZE = 16
FIG_TITLE_SIZE = 30
LEGEND_SIZE = 14


def golden_ratio(width):
    """Calculate height using the golden ratio."""
    return width / ((1 + 5**0.5) / 2)


def generate_colormap(num_classes):
    return [plt.get_cmap("tab10")(i) for i in range(num_classes)]


def combine_masks(mask_list, dataDir):
    combined_mask = np.zeros(io.imread(dataDir + mask_list[0]).shape, dtype=bool)
    for mask_name in mask_list:
        combined_mask |= io.imread(dataDir + mask_name) > 0
    return combined_mask


def plot_rois_histograms_and_labels(mask_list, img, dataDir, colors, lookup_table):
    """
    Plot the image with overlaid ROIs, class histograms with fitted normal distributions,
    and the labeled image based on the lookup table.
    """
    # Convert grayscale image to RGB and rescale intensity
    img_rgb = np.dstack([img] * 3)
    img_rgb = exposure.rescale_intensity(img_rgb, out_range=(0, 1))

    # Create a labeled image for ROIs
    labeled_img = np.zeros_like(img, dtype=int)
    for i, masks in enumerate(mask_list):
        labeled_img[combine_masks(masks, dataDir)] = i + 1

    # Overlay the ROIs on the RGB image
    for i, color in enumerate(colors[: len(mask_list)]):
        img_rgb[labeled_img == i + 1] = color[:3]

    # Initialize figure with 1x3 subplots
    fig, ax = plt.subplots(
        1, 3, figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )

    # Plot the image with overlaid ROIs
    ax[0, 0].imshow(img_rgb)
    ax[0, 0].set_title("Image with ROIs", fontsize=AXIS_TITLE_SIZE)
    ax[0, 0].axis("off")

    # Create a legend for the ROIs
    class_names = [
        "_".join(elm.split("ROI")[0] for elm in masks) for masks in mask_list
    ]
    legend_patches = [
        Line2D([0], [0], color=color[:3], lw=4) for color in colors[: len(mask_list)]
    ]
    ax[0, 0].legend(
        handles=legend_patches,
        labels=class_names,
        loc="upper right",
        title="Class Labels",
        fontsize=LEGEND_SIZE,
    )

    # Plot histograms with fitted normal distributions
    for i, masks in enumerate(mask_list):
        combined_mask = combine_masks(masks, dataDir)
        values = img[combined_mask]
        mean, sd = values.mean(), values.std()

        class_name = "_".join(elm.split("ROI")[0] for elm in masks)
        color = colors[i % len(colors)]

        ax[0, 1].hist(
            values.ravel(),
            bins=60,
            density=True,
            alpha=0.6,
            color=color,
            label=f"Class: {class_name}",
        )

        x = np.linspace(values.min(), values.max(), 100)
        pdf = norm.pdf(x, mean, sd)
        ax[0, 1].plot(x, pdf, color=color, linestyle="--")

    ax[0, 1].set_xlabel("Hounsfield unit", fontsize=AXIS_TITLE_SIZE)
    ax[0, 1].set_ylabel("Density", fontsize=AXIS_TITLE_SIZE)
    ax[0, 1].legend(fontsize=LEGEND_SIZE)
    ax[0, 1].set_title(
        "Histogram and Fitted Normal Distributions", fontsize=AXIS_TITLE_SIZE
    )

    # Create and plot the labeled image based on the lookup table
    labeled_img_lookup = np.zeros_like(img, dtype=int)
    class_name_to_label = {
        name: idx + 1 for idx, name in enumerate(sorted(set(lookup_table.values())))
    }

    for val in np.unique(img):
        if val in lookup_table:
            labeled_img_lookup[img == val] = class_name_to_label[lookup_table[val]]

    unique_classes = np.unique(labeled_img_lookup)
    cmap = ListedColormap(colors[: len(unique_classes)])
    ax[0, 2].imshow(labeled_img_lookup, cmap=cmap)
    ax[0, 2].set_title("Labeled Image", fontsize=AXIS_TITLE_SIZE)

    # Create a legend for the labeled image
    legend_patches = [
        Line2D([0], [0], color=color[:3], lw=4)
        for color in colors[: len(unique_classes)]
    ]
    ax[0, 2].legend(
        handles=legend_patches,
        labels=sorted(set(lookup_table.values())),
        loc="upper right",
        title="Class Labels",
        fontsize=LEGEND_SIZE,
    )

    plt.tight_layout()
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

    min_value = min(class_values)
    max_value = max(class_values)

    return min_value, max_value


def spleen_objective_function(region):
    expected_area = 3457.0
    expected_orientation = -0.38
    expected_eccentricity = 0.91
    expected_solidity = 0.92
    orientation_std = 0.15
    eccentricity_std = 0.15
    area_std = expected_area * 0.15
    solidity_std = 0.15

    orientation_weight = 1.0
    orientation_deviation = np.abs(region.orientation - expected_orientation)
    orientation_score = (
        np.exp(-orientation_deviation / orientation_std) * orientation_weight
    )

    eccentricity_weight = 1.0
    eccentricity_deviation = np.abs(region.eccentricity - expected_eccentricity)
    eccentricity_score = (
        np.exp(-eccentricity_deviation / eccentricity_std) * eccentricity_weight
    )

    size_weight = 1.0
    size_deviation = np.abs(region.area - expected_area)
    size_score = np.exp(-size_deviation / area_std) * size_weight

    solidity_weight = 1.0
    solidity_deviation = np.abs(region.solidity - expected_solidity)
    solidity_score = np.exp(-solidity_deviation / solidity_std) * solidity_weight

    total_score = orientation_score + eccentricity_score + solidity_score + size_score

    return total_score


def generalized_objective_function(region, reference_region, weights=None):
    """
    Generalized objective function that scores a region based on its similarity to a reference region.

    Parameters:
    - region: The candidate region in the image (regionprops object).
    - reference_region: The reference region with desired attributes (regionprops object or dictionary).
    - weights: Optional dictionary of weights for each attribute to adjust their importance.

    Returns:
    - total_score: A score representing how similar the candidate region is to the reference region.
    """

    # Define a list of attributes to consider (these should exist in both regionprops objects)
    attributes = [
        "area",
        "orientation",
        "eccentricity",
        "solidity",
        "perimeter",
        "extent",
        "convex_area",
        "major_axis_length",
        "minor_axis_length",
    ]

    # Set default weights if none are provided
    if weights is None:
        weights = {attr: 1.0 for attr in attributes}

    total_score = 0.0

    for attr in attributes:
        if hasattr(region, attr) and hasattr(reference_region, attr):
            region_value = getattr(region, attr)
            reference_value = getattr(reference_region, attr)
            # Standard deviation can be based on a fraction of the reference value
            std_dev = 0.15 * reference_value if reference_value != 0 else 1.0

            # Calculate the deviation and score
            deviation = np.abs(region_value - reference_value)
            score = np.exp(-deviation / std_dev) * weights.get(attr, 1.0)
            total_score += score

    return total_score


def spleen_finder(img_list, lookup_table):
    footprint_closing = disk(3)
    footprint_opening = disk(10)

    fig, ax = plt.subplots(
        3, len(img_list), figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )
    best_spleen_blob = None
    highest_score = -np.inf  # Initialize with a very low score

    for i, img in enumerate(img_list):

        spleen_min, spleen_max = find_class_bounds(lookup_table, "Spleen")
        spleen_estimate = (img > spleen_min) & (img < spleen_max)

        closed = closing(spleen_estimate, footprint_closing)
        opened = opening(closed, footprint_opening)

        label_img = measure.label(opened)
        region_props = measure.regionprops(label_img)

        if not region_props:
            print(f"No regions found in image {i+1}.")
            continue

        # Find the region with the highest score based on the old objective function
        current_spleen_blob = max(region_props, key=spleen_objective_function)
        current_score = spleen_objective_function(current_spleen_blob)

        # Update the best spleen blob if the current one has a higher score
        if current_score > highest_score:
            best_spleen_blob = current_spleen_blob
            highest_score = current_score

        spleen_region = label_img == current_spleen_blob.label

        img_rgb = np.dstack([img] * 3)
        img_rgb = exposure.rescale_intensity(img_rgb, out_range=(0, 1))
        img_rgb[spleen_region] = [1, 0, 0]  # Red color for spleen region

        ax[0, i].imshow(spleen_estimate, cmap="gray")
        ax[0, i].set_title(
            f'Image {i+1}: "Spleen" pixels\n ({spleen_min} - {spleen_max})HU',
            fontsize=AXIS_TITLE_SIZE,
        )
        ax[1, i].imshow(color.label2rgb(label_img))
        ax[1, i].set_title(
            f"Image {i+1}: Preprocessed image\nClose + Open + Label",
            fontsize=AXIS_TITLE_SIZE,
        )

        ax[2, i].imshow(img_rgb)
        ax[2, i].set_title(f"Image {i+1}: Spleen Overlay", fontsize=AXIS_TITLE_SIZE)

    plt.tight_layout()
    plt.show()

    # Return the best spleen blob as the reference region
    return best_spleen_blob


# def spleen_finder(img_list, lookup_table, ref_spleen):
#     footprint_closing = disk(3)
#     footprint_opening = disk(10)
#
#     fig, ax = plt.subplots(
#         3, len(img_list), figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
#     )
#
#     for i, img in enumerate(img_list):
#
#         spleen_min, spleen_max = find_class_bounds(lookup_table, "Spleen")
#         spleen_estimate = (img > spleen_min) & (img < spleen_max)
#
#         closed = closing(spleen_estimate, footprint_closing)
#         opened = opening(closed, footprint_opening)
#
#         label_img = measure.label(opened)
#         region_props = measure.regionprops(label_img)
#
#         if not region_props:
#             print(f"No regions found in image {i+1}.")
#             continue
#
#         # Use the generalized objective function to find the best spleen region
#         spleen_blob = max(
#             region_props,
#             key=lambda region: generalized_objective_function(region, ref_spleen),
#         )
#         spleen_region = label_img == spleen_blob.label
#
#         img_rgb = np.dstack([img] * 3)
#         img_rgb = exposure.rescale_intensity(img_rgb, out_range=(0, 1))
#         img_rgb[spleen_region] = [1, 0, 0]  # Red color for spleen region
#
#         ax[0, i].imshow(spleen_estimate, cmap="gray")
#         ax[0, i].set_title(
#             f'Image {i+1}: "Spleen" pixels\n ({spleen_min} - {spleen_max})HU',
#             fontsize=AXIS_TITLE_SIZE,
#         )
#         ax[1, i].imshow(color.label2rgb(label_img))
#         ax[1, i].set_title(
#             f"Image {i+1}: Preprocessed image\nClose + Open + Label",
#             fontsize=AXIS_TITLE_SIZE,
#         )
#
#         ax[2, i].imshow(img_rgb)
#         ax[2, i].set_title(f"Image {i+1}: Spleen Overlay", fontsize=AXIS_TITLE_SIZE)
#
#     plt.tight_layout()
#     plt.show()


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
lookup_table = parametricClassifier(classList, dataDir, img)
plot_rois_histograms_and_labels(
    classList, img, dataDir, colors, lookup_table=lookup_table
)

# with open("spleen_ref.pkl", "rb") as f:
#     spleen_ref = pickle.load(f)


# spleen_finder(img_list, lookup_table, spleen_ref)
spleen_ref = spleen_finder(img_list, lookup_table)

# Save the spleen reference region to a file
# with open("spleen_ref.pkl", "wb") as f:
#     pickle.dump(spleen_ref, f)
