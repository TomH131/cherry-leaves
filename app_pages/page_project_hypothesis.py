import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    """
    Explaining the hypotheses and how they were validated
    """
    st.write('### Project Hypothesis and Validation')

    st.success(
        "1. We hypothesise that the presence of visible white powdery marks "
        "on fungal-infected cherry leaves will enable a machine learning "
        "model to reliably distinguish between healthy and infected leaves.\n"
        "2. We hypothesise that we can reach the minimum accuracy of 97% set "
        "by the client. Although this is a high threshold, we believe that "
        "the model can meet or exceed this target using the dataset of four "
        "thousand images provided by the client.\n"
        "3. Currently, farm workers spend approximately 30 minutes per tree "
        "collecting samples to check for infection. We believe that a machine "
        "learning model can drastically reduce this time by instantly "
        "identifying whether a leaf is healthy or infected. Additionally, the "
        "option to upload and analyse multiple images simultaneously will "
        "further streamline the process.\n"
    )
    st.write('---')
    st.warning(
        "Upon validating our machine learning model, we have confirmed its "
        "success.\n"
        "1. Due to the clear visual differences between healthy and infected "
        "leaves, the model was able to reliably distinguish between the two "
        "categories.\n"
        "2. The model achieved an accuracy of 99%, surpassing the predefined "
        "client target of 97%. This is particularly impressive given the "
        "subtle similarities between some images of healthy and infected "
        "leaves and minor differences we could see in the average images we "
        "produced.\n"
        "3. The model's ability to process multiple images simultaneously, "
        "combined with its exceptional accuracy, significantly reduces the "
        "time required to inspect each tree. Moreover, the functionality to "
        "download results enhances the practicality of the solution, enabling "
        "faster implementation of interventions for fungal infections.\n\n"
        "In conclusion, the machine learning model successfully achieved 100% "
        "accuracy on the validation dataset, surpassing the client's target "
        "of 97%. This demonstrates the model's ability to reliably "
        "distinguish between healthy and infected cherry leaves. The solution "
        "significantly reduces inspection time, offering the potential to "
        "save labour hours across the client's farms. While further testing "
        "on diverse datasets is recommended to ensure generalisability, the "
        "results strongly support deploying the model at scale.\n\n"
        "As a result, we can see that the business requirements set by the "
        "client at the outset have been met. We conducted a study to visually "
        "distinguish healthy cherry leaves from those affected by powdery "
        "mildew, enabling the client to accurately predict the condition of "
        "cherry leaves from images.\n\n"
    )
