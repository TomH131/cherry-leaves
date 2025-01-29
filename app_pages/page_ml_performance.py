import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    st.info(
        "Here, you can see that we split our image dataset into training, "
        "validation, and test sets to train and evaluate our machine learning "
        "model."
    )

    labels_distribution = plt.imread(
        f"outputs/{version}/labels_distribution.png")
    st.image(
        labels_distribution, caption='Labels Distribution on Train, '
        'Validation and Test Sets')
    st.write("---")

    st.write("### Model History")

    st.info(
        "Here, you can observe how the model's accuracy on the training and "
        "validation sets improved over time, as shown by the accuracy and "
        "val_accuracy graphs. Similarly, the model's error rate decreased "
        "over time, as depicted by the loss and val_loss curves. This shows "
        "our hypothesis of reaching the minimum target set by the client of "
        "97% accuracy has been met and exceeded."
    )

    col1, col2 = st.columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(
        load_test_evaluation(version), index=['Loss', 'Accuracy']))
