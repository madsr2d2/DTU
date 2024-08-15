import matplotlib.pyplot as plt
import math
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage import io, img_as_float
import numpy as np


def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(transformed)
    ax2.set_title(transformed_name)
    ax2.axis("off")
    plt.show()


dataDir = "data/"
imgName = "NusaPenida.png"
imgOrg = io.imread(dataDir + imgName)


# angle in degrees - counter clockwise
rotation_angle = 10
centerOfRotation = [0, 0]
rotated_img = rotate(
    imgOrg,
    rotation_angle,
    center=centerOfRotation,
    mode="constant",
    cval=0,
    resize=True,
)
# show_comparison(imgOrg, rotated_img, "Rotated image")


# angle in radians - counter clockwise
rotation_angle = 10.0 * math.pi / 180.0
trans = [10, 20]
trans = [0, 0]
tform = EuclideanTransform(rotation=rotation_angle, translation=trans)

transformed_img = warp(imgOrg, tform.inverse)
show_comparison(imgOrg, transformed_img, "Transformed image")

rotAngle = 10.0 * math.pi / 180.0
trans = [40, 30]
scale = 0.6

tform = SimilarityTransform(rotation=rotation_angle, translation=trans, scale=scale)
transformed_img = warp(imgOrg, tform.inverse)
show_comparison(imgOrg, transformed_img, "Transformed image")


str = 10
rad = 300
swirl_img = swirl(imgOrg, strength=str, radius=rad)
show_comparison(imgOrg, swirl_img, "Swirl image")

src_img = io.imread(dataDir + "Hand1.jpg")
dst_img = io.imread(dataDir + "Hand2.jpg")

# blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
# print(type(blend))
# io.imshow(blend)
# io.show()

src = np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]])
dst = np.array([[623, 295], [382, 163], [198, 271], [283, 446], [594, 448]])
fig, ax = plt.subplots(1, 2, squeeze=False)

ax[0, 0].imshow(src_img)
ax[0, 1].imshow(dst_img)
ax[0, 0].plot(src[:, 0], src[:, 1], ".r", markersize=12)
ax[0, 1].plot(dst[:, 0], dst[:, 1], ".b", markersize=12)
ax[0, 0].set_title("src image")
ax[0, 1].set_title("dst image")
plt.tight_layout()
plt.show()


e_x = src[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f = error_x + error_y
print(f"Landmark alignment error F: {f}")
