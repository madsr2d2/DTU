import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import (
    EuclideanTransform,
    SimilarityTransform,
    warp,
    swirl,
)
from skimage import io, img_as_float
from data.Ex7_VideoImageTransformations import capture_from_camera_and_show_images

from sklearn.decomposition import PCA
from scipy.spatial import procrustes

# Constants for visualization
DATA_DIR = "data/"
IMG_NAME = "NusaPenida.png"
FIG_WIDTH = 20
AXIS_TITLE_SIZE = 16
FIG_TITLE_SIZE = 30
LEGEND_SIZE = 14


def golden_ratio(width):
    """Calculate height using the golden ratio."""
    return width / ((1 + 5**0.5) / 2)


def apply_transformations(img):
    """Apply various transformations to an image and display the results."""
    img_height, img_width = img.shape[:2]
    rotation_angle = np.pi / 4
    translation_vector = [img_height / 4, img_width / 4]
    scale_factor = 0.6
    center = np.array([img_height / 2, img_width / 2])
    swirl_strength = 10
    swirl_radius = 300

    # Transformation steps
    transform_centered = EuclideanTransform(translation=-center)
    transform_rotated = EuclideanTransform(rotation=rotation_angle)
    transform_shifted_back = EuclideanTransform(translation=center)
    transform_translated = EuclideanTransform(translation=translation_vector)
    transform_scaled = SimilarityTransform(scale=scale_factor)

    # Apply transformations
    combined_rotation = transform_centered + transform_rotated + transform_shifted_back
    combined_rot_trans = combined_rotation + transform_translated
    combined_rot_trans_scale = combined_rot_trans + transform_scaled

    # Warp images with transformations
    img_rotated = warp(img, combined_rotation.inverse)
    img_rotated_translated = warp(img, combined_rot_trans.inverse)
    img_rot_trans_scaled = warp(img, combined_rot_trans_scale.inverse)
    img_swirled = swirl(img, strength=swirl_strength, radius=swirl_radius)

    # Display transformations
    fig, ax = plt.subplots(
        2, 3, figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )
    fig.suptitle("Image Transformations", fontsize=FIG_TITLE_SIZE)

    ax[0, 0].imshow(img)
    ax[0, 0].set_title("Original Image", fontsize=AXIS_TITLE_SIZE)

    ax[0, 1].imshow(img_rotated)
    ax[0, 1].set_title("Rotation\n(EuclideanTransform)", fontsize=AXIS_TITLE_SIZE)

    ax[0, 2].imshow(img_rotated_translated)
    ax[0, 2].set_title(
        "Rotation + Translation\n(EuclideanTransform)", fontsize=AXIS_TITLE_SIZE
    )

    ax[1, 0].imshow(img_rot_trans_scaled)
    ax[1, 0].set_title(
        "Rotation + Translation + Scaling\n(SimilarityTransform)",
        fontsize=AXIS_TITLE_SIZE,
    )

    ax[1, 1].imshow(img_swirled)
    ax[1, 1].set_title("Swirl Transformation", fontsize=AXIS_TITLE_SIZE)

    ax[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


def perform_registration():
    """Perform image registration and display the results."""
    src_img = io.imread(DATA_DIR + "Hand1.jpg")
    dst_img = io.imread(DATA_DIR + "Hand2.jpg")

    # Define landmarks in both images
    src_points = np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]])
    dst_points = np.array([[623, 295], [382, 163], [198, 271], [283, 446], [594, 448]])

    # Estimate the transformation
    tform = EuclideanTransform()
    tform.estimate(src_points, dst_points)
    initial_error = np.sum((src_points - dst_points) ** 2)

    # Transform the source points using the estimated transformation matrix
    transformed_src_points = tform(src_points)
    final_error = np.sum((transformed_src_points - dst_points) ** 2).astype(int)

    # Warp the source image using the estimated transformation
    warped_src_img = warp(src_img, tform.inverse)

    # Blend the warped source image with the destination image
    blended_img = 0.5 * img_as_float(dst_img) + 0.5 * img_as_float(warped_src_img)

    # Plotting results
    fig, ax = plt.subplots(
        2, 2, figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )
    fig.suptitle("Image Registration", fontsize=FIG_TITLE_SIZE)

    ax[0, 0].imshow(src_img)
    ax[0, 0].plot(
        src_points[:, 0],
        src_points[:, 1],
        ".r",
        markersize=12,
        label="Source Image Landmarks",
    )
    ax[0, 0].plot(
        dst_points[:, 0],
        dst_points[:, 1],
        ".b",
        markersize=12,
        label="Destination Image Landmarks",
    )
    ax[0, 0].set_title("Source Image", fontsize=AXIS_TITLE_SIZE)
    ax[0, 0].plot([], [], " ", label=f"Initial Error:\nF = {initial_error}")
    ax[0, 0].legend(fontsize=LEGEND_SIZE)

    ax[0, 1].imshow(dst_img)
    ax[0, 1].plot(
        dst_points[:, 0],
        dst_points[:, 1],
        ".b",
        markersize=12,
        label="Destination Image Landmarks",
    )
    ax[0, 1].set_title("Destination Image", fontsize=AXIS_TITLE_SIZE)
    ax[0, 1].legend(fontsize=LEGEND_SIZE)

    ax[1, 0].imshow(warped_src_img)
    ax[1, 0].plot(
        transformed_src_points[:, 0],
        transformed_src_points[:, 1],
        ".r",
        markersize=12,
        label="Transformed Source Landmarks",
    )
    ax[1, 0].plot(
        dst_points[:, 0],
        dst_points[:, 1],
        ".b",
        markersize=12,
        label="Destination Image Landmarks",
    )
    ax[1, 0].plot([], [], " ", label=f"Final Error:\nF = {final_error}")
    ax[1, 0].legend(fontsize=LEGEND_SIZE)
    ax[1, 0].set_title("Transformed Source Image", fontsize=AXIS_TITLE_SIZE)

    ax[1, 1].imshow(blended_img)
    ax[1, 1].set_title(
        "Blend of Transformed Source\nand Destination Image", fontsize=AXIS_TITLE_SIZE
    )

    plt.tight_layout()
    plt.show()


def generate_synthetic_triangles(num_shapes=1000, noise_scale=0.05):
    """Generate synthetic triangles with varying base length and height."""
    base_lengths = np.random.uniform(0.5, 2.0, size=num_shapes)
    heights = np.random.uniform(0.5, 2.0, size=num_shapes)
    synthetic_shapes = []

    for base, height in zip(base_lengths, heights):
        triangle = np.array(
            [
                [-base / 2, 0],  # Left corner
                [base / 2, 0],  # Right corner
                [0, height],  # Top corner
            ]
        )
        noise = np.random.normal(scale=noise_scale, size=triangle.shape)
        noisy_triangle = triangle + noise
        synthetic_shapes.append(noisy_triangle)

    return np.array(synthetic_shapes)


def align_shapes(shapes):
    """Align shapes using Procrustes analysis."""
    aligned_shapes = []
    reference_shape = shapes[0]
    for shape in shapes:
        _, aligned_shape, _ = procrustes(reference_shape, shape)
        aligned_shapes.append(aligned_shape)
    return np.array(aligned_shapes)


def perform_pca_on_shapes(aligned_shapes):
    """Perform PCA on aligned shapes, retaining all principal components."""
    n_shapes, n_landmarks, _ = aligned_shapes.shape
    data_matrix = aligned_shapes.reshape(n_shapes, -1)
    pca = PCA()  # Default is to keep all components
    pca.fit(data_matrix)
    return pca


def visualize_pca_results(pca, aligned_shapes):
    """Visualize the PCA results and shape variations."""
    # Reshape the mean shape and PCA components for easier handling
    mean_shape = pca.mean_.reshape(-1, 2)
    n_components = pca.n_components_
    components = pca.components_.reshape(-1, 2, n_components)

    # Determine the number of subplots needed
    n_cols = min(4, n_components + 1)  # Show up to 3 PCs plus the mean shape
    n_rows = int(np.ceil((n_components + 1) / n_cols))

    # Create a figure with the appropriate number of subplots
    fig, ax = plt.subplots(
        2, 2, figsize=(FIG_WIDTH, golden_ratio(FIG_WIDTH)), squeeze=False
    )
    fig.suptitle("Shape Analysis", fontsize=FIG_TITLE_SIZE)

    ax = ax.ravel()

    # Plot all shapes and the mean shape on the first subplot
    ax[0].scatter(
        aligned_shapes[:, :, 0],
        aligned_shapes[:, :, 1],
        color="gray",
        alpha=0.5,
        label="Aligned Shapes",
    )
    ax[0].scatter(mean_shape[:, 0], mean_shape[:, 1], color="red", label="Mean Shape")
    ax[0].set_title("Mean Shape with Aligned Shapes", fontsize=AXIS_TITLE_SIZE)
    ax[0].legend(fontsize=LEGEND_SIZE)

    # Plot the variations along each principal component
    for i in range(3):
        # Compute the positive and negative variations along the PC
        variation = np.sqrt(pca.explained_variance_[i]) * components[:, :, i]
        pc_shape_pos = mean_shape + 2 * variation
        pc_shape_neg = mean_shape - 2 * variation

        # Plot the mean shape with the variations
        ax[i + 1].scatter(
            mean_shape[:, 0], mean_shape[:, 1], color="red", label="Mean Shape"
        )
        ax[i + 1].scatter(
            pc_shape_pos[:, 0],
            pc_shape_pos[:, 1],
            color="blue",
            label="Mean + 2sd",
        )
        ax[i + 1].scatter(
            pc_shape_neg[:, 0],
            pc_shape_neg[:, 1],
            color="green",
            label="Mean - 2sd",
        )

        # Set the title to include the eigenvalue
        eigenvalue = pca.explained_variance_[i]
        ax[i + 1].set_title(
            f"PC {i + 1} Shape Variation\nEigenvalue: {eigenvalue:.3f}",
            fontsize=AXIS_TITLE_SIZE,
        )
        ax[i + 1].legend(fontsize=LEGEND_SIZE)

    # Hide any unused subplots
    for j in range(n_components + 1, len(ax)):
        ax[j].axis("off")

    # Adjust the layout to prevent overlap
    plt.tight_layout()
    plt.show()


# Load and process images
img_org = io.imread(DATA_DIR + IMG_NAME)

# Apply transformations and display results
apply_transformations(img_org)

# Show video
capture_from_camera_and_show_images()

# Perform image registration and display results
perform_registration()

# Generate synthetic triangle shapes with added randomness
synthetic_shapes = generate_synthetic_triangles()

# Align shapes using Procrustes analysis
aligned_shapes = align_shapes(synthetic_shapes)

# Perform PCA on the aligned shapes
pca_results = perform_pca_on_shapes(aligned_shapes)

# Visualize the results with alignment of PC-perturbed shapes
visualize_pca_results(pca_results, aligned_shapes)
