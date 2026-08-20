import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
from pyscript import document, when
from js import FileReader, URL
from PIL import Image
import base64


# ============================================================
# GLOBAL DATASET
# ============================================================

# Dataset produced after an image has been successfully processed.
# None until an image is successfully uploaded and processed.
uploaded_dataset = None


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def get_values(df):
    xm = df.x.mean()
    ym = df.y.mean()
    xsd = df.x.std()
    ysd = df.y.std()
    pc = df.corr().x.y

    return [xm, ym, xsd, ysd, pc]


def pil_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# SCRIPT 1: IMAGE -> COORDINATE DATASET
# ============================================================

def process_image(image, N):

    cv_image = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    img_gray = cv2.cvtColor(
        cv_image,
        cv2.COLOR_BGR2GRAY
    )

    # Apply binary thresholding
    ret, thresh = cv2.threshold(
        img_gray,
        150,
        255,
        cv2.THRESH_BINARY
    )

    # Get coordinates of black/light-gray areas
    coordinates = np.column_stack(
        np.where(thresh == 0)
    )

    # Randomly sample N points
    idx = np.random.choice(
        len(coordinates),
        N,
        replace=False
    )

    selected_coords = coordinates[idx]

    # Compute centroid
    cx, cy = selected_coords.mean(axis=0)

    # Translate to origin
    X0 = selected_coords - np.array([cx, cy])

    # Apply 90° clockwise rotation
    R = np.array([
        [0, 1],
        [-1, 0]
    ])

    selected_coords = X0 @ R.T

    # Normalize per axis
    mins = selected_coords.min(axis=0)
    maxs = selected_coords.max(axis=0)

    ranges = maxs - mins
    ranges[ranges == 0] = 1

    # Convert to DataFrame
    df = pd.DataFrame(selected_coords)

    df.rename(
        columns={
            0: "x",
            1: "y"
        },
        inplace=True
    )

    # ========================================================
    # CREATE ORIGINAL SCATTER + STATISTICS FIGURE
    # ========================================================

    res = get_values(df)

    fs = 15

    labels = [
        "X Mean",
        "Y Mean",
        "X SD",
        "Y SD",
        "Corr."
    ]

    max_label_length = max(
        len(la)
        for la in labels
    )

    # Find plot limits
    min_x, min_y = (
        np.min(df.to_numpy(), axis=0) - 15
    )

    max_x, max_y = (
        np.max(df.to_numpy(), axis=0) + 15
    )

    default_min = -25
    default_max = 125

    # Increase margins while preserving square axes
    while True:

        if (
            min_x <= default_min
            or min_y <= default_min
            or max_x >= default_max
            or max_y >= default_max
        ):
            default_min -= 5
            default_max += 5
        else:
            break

    # Create figure
    fig = plt.figure(
        figsize=(7, 3),
        constrained_layout=True
    )

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1, 0.8]
    )

    # ========================================================
    # LEFT AXIS: SCATTERPLOT
    # ========================================================

    ax = fig.add_subplot(gs[0, 0])

    # sns.regplot(
        # x="x",
        # y="y",
        # data=df,
        # ci=None,
        # fit_reg=False,
        # scatter_kws={
            # "s": 4,
            # "alpha": 0.9,
            # "color": "black"
        # },
        # ax=ax
    # )
    
    ax.scatter(df["x"],
    df["y"],
    s=4,
    alpha=0.9,
    color="black"
    )

    ax.set_xlim(
        default_min,
        default_max
    )

    ax.set_ylim(
        default_min,
        default_max
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    # ========================================================
    # RIGHT AXIS: STATISTICS
    # ========================================================

    ax_text = fig.add_subplot(
        gs[0, 1]
    )

    ax_text.axis("off")

    y_positions = [
        0.9,
        0.75,
        0.60,
        0.45,
        0.30
    ]

    # Shadow text
    for i, (label, value) in enumerate(
        zip(labels, res)
    ):

        ax_text.text(
            0.0,
            y_positions[i],
            label.ljust(max_label_length)
            + ": "
            + format(value, "0.9f")[:-2],
            fontsize=fs,
            alpha=0.3,
            transform=ax_text.transAxes
        )

    # Main text
    for i, (label, value) in enumerate(
        zip(labels, res)
    ):

        ax_text.text(
            0.0,
            y_positions[i],
            label.ljust(max_label_length)
            + ": "
            + format(value, "0.9f")[:-7],
            fontsize=fs,
            alpha=1,
            transform=ax_text.transAxes
        )

    # Convert figure to PNG/base64
    fig.canvas.draw()

    buf = np.asarray(
        fig.canvas.buffer_rgba()
    )

    plt.close(fig)

    figure = Image.fromarray(buf)

    return {
        "dataset": df,
        "figure": pil_to_base64(figure)
    }


# ============================================================
# IMAGE UPLOAD
# ============================================================

def upload_image(event):

    global uploaded_dataset

    file = event.target.files.item(0)

    if file is None:
        return

    reader = FileReader.new()

    n_points = int(
        document.getElementById(
            "parameter"
        ).value
    )

    def onload(evt):

        global uploaded_dataset

        data = bytes(
            reader.result.to_py()
        )

        image = Image.open(
            io.BytesIO(data)
        )

        # Process image
        result = process_image(
            image,
            n_points
        )

        # Store dataset globally
        uploaded_dataset = result["dataset"]

        # Enable transform button
        transform_button = document.getElementById(
            "transform-button"
        )

        if transform_button is not None:
            transform_button.disabled = False

        # ====================================================
        # DISPLAY ORIGINAL IMAGE
        # ====================================================

        img1 = document.getElementById(
            "processed-image"
        )

        img1.src = URL.createObjectURL(
            file
        )

        img1.style.display = "block"

        # ====================================================
        # DISPLAY SCATTER + STATISTICS
        # ====================================================

        img2 = document.getElementById(
            "scatter-image"
        )

        img2.src = (
            "data:image/png;base64,"
            + result["figure"]
        )

        img2.style.display = "block"

        print(
            "Image successfully processed."
        )

        print(
            "Dataset contains",
            len(uploaded_dataset),
            "points."
        )

    reader.onload = onload

    reader.readAsArrayBuffer(file)


# ============================================================
# SCRIPT 2: LINEAR STATISTICAL TRANSFORMATION
# ============================================================

def project_statistics(
    coords,
    t_xm=54.265,
    t_ym=47.835,
    t_xsd=16.765,
    t_ysd=26.935,
    t_pc=-0.065,
    eps=1e-8
):

    # Make sure coords is a NumPy float array
    coords = np.asarray(
        coords,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Center current scatterplot
    # --------------------------------------------------------

    mean = coords.mean(
        axis=0,
        keepdims=True
    )

    X = coords - mean

    # --------------------------------------------------------
    # Current covariance matrix
    # --------------------------------------------------------

    cov = np.cov(
        X,
        rowvar=False
    )

    # --------------------------------------------------------
    # Target covariance matrix
    # --------------------------------------------------------

    target_cov = np.array([
        [
            t_xsd ** 2,
            t_pc * t_xsd * t_ysd
        ],
        [
            t_pc * t_xsd * t_ysd,
            t_ysd ** 2
        ]
    ], dtype=np.float64)

    # --------------------------------------------------------
    # Whitening transformation
    # --------------------------------------------------------

    eigvals, eigvecs = np.linalg.eigh(
        cov + eps * np.eye(2)
    )

    W = (
        eigvecs
        @ np.diag(
            1.0 / np.sqrt(eigvals)
        )
        @ eigvecs.T
    )

    # --------------------------------------------------------
    # Coloring transformation
    # --------------------------------------------------------

    teigvals, teigvecs = np.linalg.eigh(
        target_cov
    )

    C = (
        teigvecs
        @ np.diag(
            np.sqrt(teigvals)
        )
        @ teigvecs.T
    )

    # --------------------------------------------------------
    # Apply whitening + coloring
    # --------------------------------------------------------

    X_proj = (
        (X @ W.T)
        @ C.T
    )

    # --------------------------------------------------------
    # Apply target mean
    # --------------------------------------------------------

    X_proj += np.array([
        t_xm,
        t_ym
    ])

    return X_proj


# ============================================================
# READ TARGET STATISTICS FROM HTML
# ============================================================

def get_target_statistics():

    t_xm = float(
        document.getElementById(
            "target-x-mean"
        ).value
    )

    t_ym = float(
        document.getElementById(
            "target-y-mean"
        ).value
    )

    t_xsd = float(
        document.getElementById(
            "target-x-sd"
        ).value
    )

    t_ysd = float(
        document.getElementById(
            "target-y-sd"
        ).value
    )

    t_pc = float(
        document.getElementById(
            "target-correlation"
        ).value
    )

    return (
        t_xm,
        t_ym,
        t_xsd,
        t_ysd,
        t_pc
    )

def create_transformed_figure(df):

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    res = get_values(df)

    labels = [
        "X Mean",
        "Y Mean",
        "X SD",
        "Y SD",
        "Corr."
    ]

    # --------------------------------------------------------
    # Determine plot limits
    # --------------------------------------------------------

    min_x, min_y = (
        np.min(df.to_numpy(), axis=0) - 15
    )

    max_x, max_y = (
        np.max(df.to_numpy(), axis=0) + 15
    )

    default_min = -25
    default_max = 125

    # Increase margins while preserving square axes
    while True:

        if (
            min_x <= default_min
            or min_y <= default_min
            or max_x >= default_max
            or max_y >= default_max
        ):
            default_min -= 5
            default_max += 5
        else:
            break
    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(7, 3),
        constrained_layout=True
    )

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1, 0.8]
    )

    # --------------------------------------------------------
    # LEFT: SCATTERPLOT
    # --------------------------------------------------------

    ax = fig.add_subplot(gs[0, 0])

    sns.regplot(
        x="x",
        y="y",
        data=df,
        ci=None,
        fit_reg=False,
        scatter_kws={
            "s": 4,
            "alpha": 0.9,
            "color": "black"
        },
        ax=ax
    )

    ax.set_xlim(
        default_min,
        default_max
    )

    ax.set_ylim(
        default_min,
        default_max
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    # --------------------------------------------------------
    # RIGHT: STATISTICS
    # --------------------------------------------------------

    ax_text = fig.add_subplot(
        gs[0, 1]
    )

    ax_text.axis("off")

    fs = 15

    y_positions = [
        0.9,
        0.75,
        0.60,
        0.45,
        0.30
    ]

    max_label_length = max(
        len(label)
        for label in labels
    )

    # Shadow
    for i, (label, value) in enumerate(
        zip(labels, res)
    ):

        ax_text.text(
            0.0,
            y_positions[i],
            label.ljust(max_label_length)
            + ": "
            + format(value, "0.9f")[:-2],
            fontsize=fs,
            alpha=0.3,
            transform=ax_text.transAxes
        )

    # Main text
    for i, (label, value) in enumerate(
        zip(labels, res)
    ):

        ax_text.text(
            0.0,
            y_positions[i],
            label.ljust(max_label_length)
            + ": "
            + format(value, "0.9f")[:-7],
            fontsize=fs,
            alpha=1,
            transform=ax_text.transAxes
        )

    # --------------------------------------------------------
    # Convert figure to base64
    # --------------------------------------------------------

    fig.canvas.draw()

    buf = np.asarray(
        fig.canvas.buffer_rgba()
    )

    plt.close(fig)

    figure = Image.fromarray(buf)

    return pil_to_base64(figure)


# ============================================================
# TRANSFORM DATASET
# ============================================================

@when("click", "#transform-button")
def transform_dataset():

    global uploaded_dataset

    print("=== TRANSFORM DATASET ===")

    # --------------------------------------------------------
    # Make sure an image has been processed
    # --------------------------------------------------------

    if uploaded_dataset is None:

        print(
            "No dataset available."
        )

        return

    print(
        "Dataset found!"
    )

    print(
        "Number of points:",
        len(uploaded_dataset)
    )

    # --------------------------------------------------------
    # Get target statistics
    # --------------------------------------------------------

    (
        t_xm,
        t_ym,
        t_xsd,
        t_ysd,
        t_pc
    ) = get_target_statistics()

    print(
        "Target statistics:"
    )

    print(
        "X Mean:",
        t_xm
    )

    print(
        "Y Mean:",
        t_ym
    )

    print(
        "X SD:",
        t_xsd
    )

    print(
        "Y SD:",
        t_ysd
    )

    print(
        "Correlation:",
        t_pc
    )

    # --------------------------------------------------------
    # Extract coordinate matrix
    # --------------------------------------------------------

    coords = uploaded_dataset[
        ["x", "y"]
    ].to_numpy()

    # --------------------------------------------------------
    # Apply transformation
    # --------------------------------------------------------

    transformed_coords = project_statistics(
        coords,
        t_xm=t_xm,
        t_ym=t_ym,
        t_xsd=t_xsd,
        t_ysd=t_ysd,
        t_pc=t_pc
    )

    # --------------------------------------------------------
    # Convert back to DataFrame
    # --------------------------------------------------------

    transformed_df = pd.DataFrame(
        transformed_coords,
        columns=[
            "x",
            "y"
        ]
    )

    # --------------------------------------------------------
    # Calculate resulting statistics
    # --------------------------------------------------------

    result = get_values(
        transformed_df
    )

    print(
        "=== TRANSFORMED STATISTICS ==="
    )

    print(
        "X Mean:",
        result[0]
    )

    print(
        "Y Mean:",
        result[1]
    )

    print(
        "X SD:",
        result[2]
    )

    print(
        "Y SD:",
        result[3]
    )

    print(
        "Correlation:",
        result[4]
    )

    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    figure_base64 = create_transformed_figure(
        transformed_df
    )


    # ========================================================
    # DISPLAY ON RIGHT SIDE
    # ========================================================

    transformed_image = document.getElementById(
        "transformed-image"
    )

    transformed_image.src = (
        "data:image/png;base64,"
        + figure_base64
    )

    transformed_image.style.display = "block"


    # Hide placeholder
    placeholder = document.getElementById(
        "image-placeholder"
    )

    if placeholder is not None:
        placeholder.style.display = "none"


    return transformed_df


# ============================================================
# REGISTER HTML-CALLABLE FUNCTIONS
# ============================================================

globals()["upload_image"] = upload_image
# globals()["transform_dataset"] = transform_dataset