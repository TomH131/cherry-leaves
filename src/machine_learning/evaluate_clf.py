import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation(version):
    """
    load the model evaluation file
    """
    return load_pkl_file(f'outputs/{version}/evaluation.pkl')
