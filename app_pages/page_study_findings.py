import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random


def page_study_findings_body():
    """
    Showing the average and variability images of both healthy and non
    healthy leaves as well the difference between the two averages images
    and an image montage
    """
    st.write("### Cherry Leaf Visualizer")
    st.info(
        "The client is interested in conducting a study to visually "
        "differentiate a healthy cherry leaf from one with powdery mildew.\n"
    )

    version = "v1"

    # Difference between average and variability image
    if st.checkbox("Difference between average and variability image"):
        try:
            avg_healthy = plt.imread(f"outputs/{version}/avg_var_Healthy.png")
            avg_powdery_mildew = plt.imread(
                    f"outputs/{version}/avg_var_Powdery_Mildew.png")

            st.warning(
                "We notice the average and variability images did not show "
                "clear patterns where we could intuitively differentiate one "
                "from another. However, there is a small difference where the "
                "infected leaves have small white marks."
            )

            st.image(
                avg_healthy,
                caption="Healthy cherry leaf - Average and Variability"
            )

            st.image(
                avg_powdery_mildew,
                caption=(
                    "Cherry leaf with powdery mildew - "
                    "Average and Variability"
                )

            )

        except FileNotFoundError as e:
            st.error(f"Error loading images: {e}")

        st.write("---")

    if st.checkbox(
            "Differences between average healthy and average infected leaf"):
        try:
            diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

            st.warning(
                "We notice this study didn't show patterns where "
                "we could intuitively differentiate one from another.\n"
            )
            st.image(
                diff_between_avgs, caption="Difference between average images")

        except FileNotFoundError:
            st.error("Error: 'avg_diff.png' not found.")

    if st.checkbox("Image Montage"):
        st.write(
            f"* To refresh the montage, click on the 'Create Montage' button")

        my_data_dir = "inputs/datasets/raw/cherry-leaves"
        labels = os.listdir(os.path.join(my_data_dir, "validation"))
        label_to_display = st.selectbox(
            label="Select label", options=labels, index=0)

        if st.button("Create Montage"):
            image_montage(
                dir_path=os.path.join(my_data_dir, "validation"),
                label_to_display=label_to_display,
                nrows=8,
                ncols=3,
                figsize=(10, 25),
            )

        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    """
    Creating the image montage
    """
    sns.set_style("white")
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(os.path.join(dir_path, label_to_display))

        if nrows * ncols <= len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            print(
                f"Decrease nrows or ncols to create your montage. \n"
                f"There are {len(images_list)} images in your subset. "
                f"You requested a montage with {nrows * ncols} spaces."
            )
            return

        plot_idx = list(itertools.product(range(nrows), range(ncols)))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for x in range(nrows * ncols):
            img = imread(os.path.join(dir_path, label_to_display, img_idx[x]))
            img_shape = img.shape
            row, col = plot_idx[x]

            axes[row, col].imshow(img)
            axes[row, col].set_title(
                f"Width {img_shape[1]}px x Height {img_shape[0]}px")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        plt.tight_layout()
        st.pyplot(fig=fig)

    else:
        print(f"The label '{label_to_display}' does not exist.")
        print(f"Available labels: {labels}")
