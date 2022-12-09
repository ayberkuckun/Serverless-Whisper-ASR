from transformers import pipeline
from pytube import YouTube

import gradio as gr
import librosa

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/best_model/model_scores.png", overwrite=True)  # change link

pipe = pipeline(model="ayberkuckun/whisper-small-se-hyperparameter-searched", task="automatic-speech-recognition",
                chunk_length_s=30)


def transcribe(mic=None, url=None):
    if (mic is None) and (len(url) == 0):
        return None, "ERROR: You have to either use the microphone or paste a Youtube URL", None

    elif (mic is not None) and (len(url) > 0):
        return mic, pipe(mic)["text"], "model_scores.png"

    elif (len(url) > 0):
        selected_video = YouTube(url)

        try:
            audio = selected_video.streams.filter(only_audio=True)[0]
        except:
            raise Exception("Can't find an mp4 audio.")

        audio.download(filename="audio.mp3")

        speech_array, _ = librosa.load("audio.mp3", sr=16000)

        output = pipe(speech_array)

        return "audio.mp3", output["text"], "model_scores.png"
    else:
        return mic, pipe(mic)["text"], "model_scores.png"


iface = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(source="microphone", type="filepath", optional=True),
        gr.Textbox("https://www.youtube.com/shorts/1j2lRgtsDZk", label="Paste a YouTube video URL", optional=True),
    ],
    outputs=[gr.Audio(label="Transcripted Audio"),
             gr.Textbox(label="Transcription"),
             gr.Image(label="Model Scores")
             ],
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
    allow_flagging="never"
)

iface.launch()