# Serverless-Whisper-ASR

**Feature Engineering Pipeline:**
- ``feature_engineering.ipynb``

We separate the feature engineering from the training since it doesn't require the precious GPU resource.
We process the ``mozilla-foundation/common_voice_11_0`` Swedish data in order to fine-tune ``openai/whisper-small``
We change the sampling rate of the audio first. Then, process the audio and the transcription labels 
using feature extractor and tokenizer. Lastly we save the processed data to our drive.

PS: These notebooks have been prepared to be run on Colab.

**Training Pipeline:**
- Hyperparameter tuning for best model
- Final training on best parameters found

We download the processed data form our drive and the raw model from HuggingFace.
The final model is trained for 2000 steps with the best parameters obtained from the hyperparameter search 
for 300 steps. Hyperparameter search has been done using Wandb and the results can be found under 
``parameter_search.pdf``. Lastly the final model has been pushed to the HuggingFace

**Inference Program (Hugging Face Space):**
- https://huggingface.co/spaces/reyrobs/whisper-small-sv-SE

The user can use a Youtube video which is in Swedish, or use the microphone in order to do a transcription. The default link present is a short 30 Youtube "shorts" video. In the event that both a Youtube link is entered as well as a microphone input, the space defaults to translating the microphone input. As output, we get an audio representing the input entry, a transciption and each of the model's WER score. 
