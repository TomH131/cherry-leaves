import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        "**General Information**\n\n"
        "This machine learning project involves creating a model for Farmy & "
        "Foods to accurately predict whether a cherry leaf from their crops "
        "is healthy. Currently, some cherry leaves show signs of white "
        "powdery mildew, indicating a fungal infection. However, verifying "
        "the health of each tree takes an employee, on average, 30 minutes to "
        "examine and sample.\n\n"
        "The goal of this project is to significantly reduce the time "
        "required for health assessments, as the company manages thousands of "
        "cherry leaves across multiple farms nationwide.\n\n"

        "**Project Dataset**\n\n"
        "The dataset provided by the client contains over 4,000 images taken "
        "from their crop fields, evenly split between healthy leaves and "
        "those affected by disease. The images depict healthy cherry leaves "
        "as well as leaves showing signs of powdery mildew, a fungal disease "
        "that impacts many plant species.")

    st.success(
        "The project has 2 business requirements:\n\n"
        "1 - The client is interested in conducting a study to visually "
        "differentiate a healthy cherry leaf from one with powdery mildew.\n\n"
        "2 - The client is interested in predicting if a cherry leaf is "
        "healthy or contains powdery mildew.\n\n"
        )
