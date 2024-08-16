import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from data.LDA import LDA

# Load data
in_dir = "data/"
in_file = "ex6_ImagData2Load.mat"
data = sio.loadmat(in_dir + in_file)
ImgT1 = data["ImgT1"]
ImgT2 = data["ImgT2"]
GM = data["ROI_GM"].astype(bool)
WM = data["ROI_WM"].astype(bool)

# Analyze the background using the first 25 rows
background_T1 = ImgT1[:25, :].ravel()
background_T2 = ImgT2[:25, :].ravel()
threshold_T1 = np.mean(background_T1) + 4 * np.std(background_T1)
threshold_T2 = np.mean(background_T2) + 4 * np.std(background_T2)

# Apply thresholds to remove background pixels
mask = (ImgT1 > threshold_T1) & (ImgT2 > threshold_T2)  # Combined mask
T1_values_filtered = ImgT1[mask].ravel()
T2_values_filtered = ImgT2[mask].ravel()


def ex1_3():
    # Create subplots
    fig, axs = plt.subplots(3, 3, figsize=(18, 18), squeeze=False)

    # Display T1 image with GM and WM ROIs
    axs[0, 0].imshow(ImgT1, cmap="gray")
    axs[0, 0].contour(GM, colors="red", linewidths=0.5)
    axs[0, 0].contour(WM, colors="blue", linewidths=0.5)
    axs[0, 0].set_title("T1 Image with ROIs")

    # Display T2 image with GM and WM ROIs
    axs[0, 1].imshow(ImgT2, cmap="gray")
    axs[0, 1].contour(GM, colors="red", linewidths=0.5)
    axs[0, 1].contour(WM, colors="blue", linewidths=0.5)
    axs[0, 1].set_title("T2 Image with ROIs")

    # Plot 1D histogram of T1 values
    axs[1, 0].hist(T1_values_filtered, bins=60, color="blue", alpha=0.7)
    axs[1, 0].set_title("T1 Histogram")
    axs[1, 0].set_xlabel("T1 Intensity")
    axs[1, 0].set_ylabel("Frequency")

    # Plot 1D histogram of T2 values
    axs[1, 1].hist(T2_values_filtered, bins=60, color="green", alpha=0.7)
    axs[1, 1].set_title("T2 Histogram")
    axs[1, 1].set_xlabel("T2 Intensity")
    axs[1, 1].set_ylabel("Frequency")

    # Plot 2D histogram (heatmap) of T1 and T2 values
    hist2d = axs[0, 2].hist2d(
        T1_values_filtered, T2_values_filtered, bins=60, density=True, cmap="plasma"
    )
    axs[0, 2].set_title("2D Histogram of T1 vs T2")
    axs[0, 2].set_xlabel("T1 Intensity")
    axs[0, 2].set_ylabel("T2 Intensity")
    fig.colorbar(hist2d[3], ax=axs[0, 2])

    # Scatter plot of T1 vs T2 values with ROIs highlighted
    axs[1, 2].scatter(
        T1_values_filtered, T2_values_filtered, color="gray", s=1, alpha=0.5
    )
    axs[1, 2].scatter(ImgT1[GM], ImgT2[GM], color="red", s=1, alpha=0.5, label="GM ROI")
    axs[1, 2].scatter(
        ImgT1[WM], ImgT2[WM], color="blue", s=1, alpha=0.5, label="WM ROI"
    )
    axs[1, 2].set_title("Scatter Plot of T1 vs T2 with ROIs")
    axs[1, 2].set_xlabel("T1 Intensity")
    axs[1, 2].set_ylabel("T2 Intensity")
    axs[1, 2].legend()

    # Plot histograms of GM and WM ROIs in T1
    axs[2, 0].hist(ImgT1[GM].ravel(), bins=60, color="red", alpha=0.7, label="GM ROI")
    axs[2, 0].hist(ImgT1[WM].ravel(), bins=60, color="blue", alpha=0.7, label="WM ROI")
    axs[2, 0].set_title("GM and WM ROI T1 Histograms")
    axs[2, 0].set_xlabel("T1 Intensity")
    axs[2, 0].set_ylabel("Frequency")
    axs[2, 0].legend()

    # Plot histograms of GM and WM ROIs in T2
    axs[2, 1].hist(ImgT2[GM].ravel(), bins=60, color="red", alpha=0.7, label="GM ROI")
    axs[2, 1].hist(ImgT2[WM].ravel(), bins=60, color="blue", alpha=0.7, label="WM ROI")
    axs[2, 1].set_title("GM and WM ROI T2 Histograms")
    axs[2, 1].set_xlabel("T2 Intensity")
    axs[2, 1].set_ylabel("Frequency")
    axs[2, 1].legend()

    axs[2, 2].axis("off")  # Keep the grid consistent

    plt.tight_layout()
    plt.show()


# Run the visualization
ex1_3()

# Flatten the ROI masks and extract corresponding T1 and T2 values
GM_T1 = ImgT1[GM].ravel()
GM_T2 = ImgT2[GM].ravel()
WM_T1 = ImgT1[WM].ravel()
WM_T2 = ImgT2[WM].ravel()

# Combine T1 and T2 values for each class
X_GM = np.column_stack((GM_T1, GM_T2))  # Class 1 (GM)
X_WM = np.column_stack((WM_T1, WM_T2))  # Class 2 (WM)

# Concatenate the data points from both classes
X = np.vstack((X_GM, X_WM))

# Create the target vector
T = np.hstack((np.zeros(len(X_GM)), np.ones(len(X_WM))))

# Train the LDA classifier
W = LDA(X, T)

# Apply the LDA classifier to the entire image data
Xall = np.c_[ImgT1.ravel(), ImgT2.ravel()]
Y = np.c_[np.ones((len(Xall), 1)), Xall] @ W.T

# Calculate the posterior probabilities for each class
PosteriorProb = np.clip(np.exp(Y) / np.sum(np.exp(Y), axis=1)[:, np.newaxis], 0, 1)

# Reshape the posterior probability for Class 1 and Class 2
P_C1_img = PosteriorProb[:, 0].reshape(ImgT1.shape)
P_C2_img = PosteriorProb[:, 1].reshape(ImgT1.shape)

# Apply segmentation based on the posterior probabilities
Class1_segmentation = np.where(P_C1_img > 0.5, 1, 0)
Class2_segmentation = np.where(P_C2_img > 0.5, 1, 0)

# Exercise 10: Scatter plot of segmentation results
plt.figure(figsize=(10, 6))
plt.scatter(
    ImgT1[Class1_segmentation == 1],
    ImgT2[Class1_segmentation == 1],
    color="green",
    s=1,
    alpha=0.5,
    label="Class 1 (GM)",
)
plt.scatter(
    ImgT1[Class2_segmentation == 1],
    ImgT2[Class2_segmentation == 1],
    color="black",
    s=1,
    alpha=0.5,
    label="Class 2 (WM)",
)
plt.title("Scatter Plot of Segmentation Results")
plt.xlabel("T1 Intensity")
plt.ylabel("T2 Intensity")
plt.legend()
plt.show()

# Exercise 11: Comparison with the original image
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Plot the original T1 image
axs[0, 0].imshow(ImgT1, cmap="gray")
axs[0, 0].set_title("Original T1 Image")
axs[0, 0].axis("off")

# Plot the original T2 image
axs[0, 1].imshow(ImgT2, cmap="gray")
axs[0, 1].set_title("Original T2 Image")
axs[0, 1].axis("off")

# Plot the segmentation result for Class 1
axs[1, 0].imshow(Class1_segmentation, cmap="gray")
axs[1, 0].set_title("Segmentation Result for Class 1 (GM)")
axs[1, 0].axis("off")

# Plot the segmentation result for Class 2
axs[1, 1].imshow(Class2_segmentation, cmap="gray")
axs[1, 1].set_title("Segmentation Result for Class 2 (WM)")
axs[1, 1].axis("off")

plt.tight_layout()
plt.show()
