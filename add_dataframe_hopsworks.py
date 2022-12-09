from transformers import pipeline
from pytube import YouTube

import gradio as gr
import librosa

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

uploaded_file_path = dataset_api.upload(
    local_path = "./model_scores.png",
    upload_path = "Resources/best_model", overwrite=True)