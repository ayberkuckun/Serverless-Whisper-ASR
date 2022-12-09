# Serverless-Whisper-ASR

**Feature Engineering Pipeline:**
- ``feature_engineering.ipynb``

We separate the feature engineering from the training since it doesn't require the precious GPU resource.
We process the ``mozilla-foundation/common_voice_11_0`` Swedish data in order to fine-tune ``openai/whisper-small``
We change the sampling rate of the audio first. Then, process the audio and the transcription labels 
using feature extractor and tokenizer. Lastly we save the processed data to our drive.

PS: These notebooks have been prepared to be run on Colab.

**Training Pipeline:**
- ``training.ipynb``

We download the processed data form our drive and the raw model from HuggingFace.
The final model is trained for 2000 steps with the best parameters obtained from the hyperparameter search 
for 300 steps. Hyperparameter search has been done using Wandb and the results can be found under 
``parameter_search.pdf``. Lastly the final model has been pushed to the HuggingFace

**Inference Program (Hugging Face Space):**
- https://huggingface.co/spaces/ayberkuckun/whisper-small-sv-SE

Users can give a YouTube link to a Swedish video in order to obtain the audio and its trascribtion from our model.
Also, the WER scores of ``base``, ``2K`` and ``2K-fine-tuned`` model can be seen in the page.