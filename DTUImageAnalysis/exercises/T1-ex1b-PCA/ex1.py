import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data
in_dir = "data/"  # Update with the correct path
txt_name = "irisdata.txt"  # Update with the correct filename
iris_data = np.loadtxt(in_dir + txt_name, comments="%")
x = iris_data[0:50, 0:4]  # We use only the first 50 samples

# Check data dimensions
n_feat = x.shape[1]
n_obs = x.shape[0]
print(f"Number of features: {n_feat}, Number of observations: {n_obs}")

# Visualize data structure using seaborn pairplot
# plt.figure()
feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
d = pd.DataFrame(x, columns=feature_names)
sns.pairplot(d)
plt.show()

# Perform PCA using scikit-learn
pca = PCA()
pca.fit(x)

# Extract eigenvalues, explained variance ratios, and principal components
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_ * 100
vectors_pca = pca.components_

print("\nPrincipal Components Analysis (scikit-learn):")
print("Eigenvalues:\n", values_pca)
print("Explained Variance Ratio (%):\n", exp_var_ratio)
print("Principal Components (Eigenvectors):\n", vectors_pca)

# Plot explained variance ratio
plt.plot(exp_var_ratio, marker="o", label="scikit-learn PCA")
plt.xlabel("Principal Component")
plt.ylabel("Percent Explained Variance")
plt.ylim([0, 100])
plt.legend()
plt.show()

# Project data into PCA space
data_transform = pca.transform(x)

# Visualize projected data structure using seaborn pairplot
d_proj = pd.DataFrame(
    data_transform, columns=[f"PC {i+1}" for i in range(data_transform.shape[1])]
)
sns.pairplot(d_proj)
plt.show()
