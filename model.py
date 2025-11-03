import cv2
from transformers import ViTImageProcessor as ImgProcessor, ViTForImageClassification as ImgClassifier
from PIL import Image
import numpy as np
import torch
import streamlit as st

@st.cache_resource
def init_model(path_to_model):
    preproc = ImgProcessor.from_pretrained(path_to_model)
    net = ImgClassifier.from_pretrained(path_to_model)
    return preproc, net


def detect_mood(image_input, net, preproc):
    if isinstance(image_input, np.ndarray):
        pic = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    elif isinstance(image_input, str):
        pic = Image.open(image_input)
    else:
        pic = Image.open(image_input)

    data = preproc(pic, return_tensors="pt")

    with torch.no_grad():
        result = net(**data)
        probs = torch.nn.functional.softmax(result.logits, dim=-1)
        label_idx = torch.argmax(probs, dim=-1).item()

    mood_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    mood = mood_labels[label_idx]
    certainty = probs[0][label_idx].item()

    return mood, certainty
