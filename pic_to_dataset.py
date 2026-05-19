import cv2
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
from datasaurus_dozen_implem import get_values

parser = argparse.ArgumentParser()
parser.add_argument("-img", "--image", help="Target image (black and white image) to convert to point cloud dataset (e.g. bike.png)", type=str)
parser.add_argument("-n", "--npoints", help="Number of points to try and extract from the image", type=int, const=142, nargs='?')

args = parser.parse_args()


spec_dir = 'target_datasets/input_images/'

# define output directories for the PNG images and coordinates
output_dir = 'target_datasets/'

############################################################################################################
# very important variable, sets the number of points we want to use for each frame of the rickroll/datadance
N = args.npoints
############################################################################################################

# load the black-and-white image in grayscale
image = cv2.imread(spec_dir + args.image, cv2.IMREAD_GRAYSCALE)

# define the range for light gray and black color (we want to keep these regions)
black_threshold = 50  # black pixel intensity threshold (0-255)
light_gray_min = 100  # minimum intensity for light gray
light_gray_max = 220  # maximum intensity for light gray

# create a mask identifying the black and light gray areas
mask = np.zeros_like(image, dtype=np.uint8)
mask[(image <= black_threshold) | ((image >= light_gray_min) & (image <= light_gray_max))] = 255

# get the coordinates of the black and light gray areas
coordinates = np.column_stack(np.where(mask == 255))

# evenly sample the coordinates from the black and light gray areas
# we want to distribute N points over the identified area
step_size = max(1, len(coordinates) // N)

# select the coordinates for the dots (evenly spaced)
selected_coords = coordinates[::step_size][:N]

# create a blank white image to draw the dots
output_image = np.ones_like(image, dtype=np.uint8) * 255  # White background

# draw the dots (black color) on the image
for (y, x) in selected_coords:
    cv2.circle(output_image, (x, y), radius=1, color=(0, 0, 0), thickness=-1)  # Black dot

# save the image with dots as a .png file
# cv2.imwrite(output_dir + args.image.replace('.png', 'test_{}.png'.format(str(N))), output_image)

# compute centroid
cx, cy = selected_coords.mean(axis = 0)
# translate to origin
X0 = selected_coords - np.array([cx, cy])
# apply 90° CW rotation: (x',y') = (y, -x)
R = np.array([[0, 1],
              [-1, 0]])
selected_coords = X0 @ R.T

# normalize per axis (preserve aspect ratio)
mins = selected_coords.min(axis=0)  # [min_x, min_y]
maxs = selected_coords.max(axis=0)  # [max_x, max_y]

# avoid divide by zero
ranges = maxs - mins
ranges[ranges == 0] = 1

# scale to 0,100
selected_coords = (selected_coords - mins) / ranges
selected_coords = selected_coords * 100

# manually scale to be closer to target x and y mean [54.26, 47.83] (helps for some fooling techniques)
selected_coords *= 0.9
selected_coords += 24

# save as csv
coords_file = os.path.join(output_dir, args.image.replace(args.image[-4:], '_{}.csv'.format(str(N))))
df = pd.DataFrame(selected_coords)
df.to_csv(coords_file, index=False, header=None)

df.rename(columns={0: "x", 1: "y"}, inplace=True)
# save the scatter plot as a png
res = get_values(df)
fs = 30
labels = ["X Mean", "Y Mean", "X SD", "Y SD", "Corr."]
max_label_length = max(len(la) for la in labels)

# create figure with tight layout using GridSpec
fig = plt.figure(figsize=(12, 5), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.8])

# === LEFT AXIS: SCATTERPLOT ===
ax = fig.add_subplot(gs[0, 0])
sns.regplot(
    x="x", y="y", data=df, ci=None, fit_reg=False,
    scatter_kws={"s": 4, "alpha": 0.9, "color": "black"},
    ax=ax
)

# find the mins and maxs plus a margin of 15
min_x, min_y = np.min(df.to_numpy(), axis=0) - 15
max_x, max_y = np.max(df.to_numpy(), axis=0) + 15

default_min = -25
default_max = 125

# increase the margins by 5 if the dataset contains values bigger than the default margins (preserve squareness of the xlim and ylim)
while True:
    if min_x <= default_min or min_y <= default_min or max_x >= default_max or max_y >= default_max:
        default_min -= 5
        default_max += 5
    else:
        break

ax.set_xlim(default_min, default_max)
ax.set_ylim(default_min, default_max)
ax.set_aspect("equal", adjustable="box")  # perfect square axes

# === RIGHT AXIS: TEXT BLOCK ===
ax_text = fig.add_subplot(gs[0, 1])
ax_text.axis("off")

y_positions = [0.9, 0.75, 0.60, 0.45, 0.30]

# shadow / lighter text
for i, (label, value) in enumerate(zip(labels, res)):
    ax_text.text(
        0.0, y_positions[i],
        label.ljust(max_label_length) + ": " + format(value, "0.9f")[:-2],
        fontsize=fs, alpha=0.3, transform=ax_text.transAxes
    )

# main bold text
for i, (label, value) in enumerate(zip(labels, res)):
    ax_text.text(
        0.0, y_positions[i],
        label.ljust(max_label_length) + ": " + format(value, "0.9f")[:-7],
        fontsize=fs, alpha=1, transform=ax_text.transAxes
    )

plt.savefig(os.path.join(output_dir, args.image.replace(args.image[-4:], '_{}.png'.format(str(N)))), dpi=72)
plt.clf()
plt.cla()
plt.close()

print('Saved the new dataset as a .csv and .png')
