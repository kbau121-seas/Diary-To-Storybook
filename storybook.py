import streamlit as st
import tkinter as tk
from tkinter import filedialog

import os
import tempfile

import requests
import google.auth
from google.cloud import vision
from google.cloud.vision_v1 import types
import vertexai
from vertexai.language_models import TextGenerationModel

google_api_credentials = None

# Initializes the operating system's Google API Credentials
def set_google_api_credentials():
	uploaded_file = st.file_uploader("Google API Credentials", ["json"])
	if (uploaded_file is None):
		return False, None

	with tempfile.NamedTemporaryFile(delete=False) as fp:
		data = uploaded_file.read()
		fp.write(data)

	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fp.name
	client = vision.ImageAnnotatorClient()

	global google_api_credentials
	google_api_credentials = fp

	st.success("Google API Credentials Set...  \nFilename: " + uploaded_file.name)

	return True, client

# Initializes a Vertex API Model
def init_vertex_api_model(
	project_id="cis-581-project-testing",
	model_id="text-bison@002",
	location="us-central1"
	):
	vertexai.init(project=project_id, location=location)

	model = TextGenerationModel.from_pretrained(model_id)

	st.success("Vertex API Model Set...  \nProject: " + project_id + "  \nModel: " + model_id + "  \nServer Location: " + location)

	return True, model

# Reads text from an image
def read_image_text(client, content):
	image = vision.Image(content=content)

	response = client.text_detection(image=image)
	annotations = response.text_annotations

	if not annotations: return False

	st.success("Successfully read the input image")
	st.markdown(annotations[0].description.replace("\n", "  \n"))

	return True, annotations[0].description

def predict_vertex_api_text(
	model,
	text,
	parameters={
		"temperature": 0,
		"max_output_tokens": 256,
		"top_p": 0.95,
		"top_k": 40,
		}
	):
	response = model.predict(text, **parameters)

	st.success("Successfully summarized the input text")
	st.markdown(response.text.replace("\n", "  \n"))

	return True, response.text

# Requests an image from the user
def input_image(types=['png', 'jpg']):
	uploaded_file = st.file_uploader("Input Image", types)
	if (uploaded_file is None):
		return False, uploaded_file

	file_data = uploaded_file.read()
	file_size_MB = round(len(file_data) / (1024 ** 2), 1)

	st.success("Upload Successful...  \nFilename: " + uploaded_file.name + "  \nFile Size: " + str(file_size_MB) + "MB")

	st.image(file_data)
	return True, file_data

# Runs the diary to storybook pipeline
def run_pipeline():
	success, client = set_google_api_credentials()
	if not success: return

	success, vertex_api_model = init_vertex_api_model()
	if not success: return

	success, image = input_image()
	if not success: return

	success, text = read_image_text(client, image)
	if not success: return

	success = predict_vertex_api_text(vertex_api_model, text)
	if not success: return

# --- Start --- #

st.title("Diary to Storybook")

run_pipeline()

# Cleanup

if google_api_credentials is not None:
	os.unlink(google_api_credentials.name)