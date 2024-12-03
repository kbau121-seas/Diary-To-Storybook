import storybook_backend as backend

import streamlit as st
import tkinter as tk
from tkinter import filedialog

import os
import tempfile
import math

import io
from PIL import Image

import base64

# Requests an image from the user
def input_image(title="Input Image", types=['png', 'jpg']):
	uploaded_file = st.file_uploader(title, types)
	if (uploaded_file is None):
		return False, None

	file_data = uploaded_file.read()
	file_size_MB = round(len(file_data) / (1024 ** 2), 1)

	st.success("Upload Successful...  \nFilename: " + uploaded_file.name + "  \nFile Size: " + str(file_size_MB) + "MB")

	st.image(file_data)
	return True, file_data

def input_text(label="Input Text"):
	text_box = st.text_input(label)

	return text_box != None and text_box.strip() != "", text_box

# Displays an array of Vertex AI Images in order
def display_vertex_images(images):
	columns = 3
	l, m, r = st.columns(columns)
	cols = [l, m, r]

	rows = math.ceil(len(images) / columns)

	for row in range(rows):
		for col in range(columns):
			idx = col + row * columns
			if idx >= len(images): break

			b_img = io.BytesIO()
			images[idx][0]._pil_image.save(b_img, format='PNG')
			image_bytes = b_img.getvalue()

			cols[col].image(image_bytes)

	return True

# Runs the diary to storybook pipeline
def run_pipeline():
	success, client = backend.set_google_api_credentials()
	if not success: return

	success = backend.init_vertex_ai()
	if not success: return

	success, text_model = backend.init_vertex_ai_text_model()
	if not success: return

	success, image_model = backend.init_vertex_ai_image_model()
	if not success: return

	success, journal_image = input_image("Journal Image")
	if not success: return

	success, author_image = input_image("Author Image")
	if not success: return

	success, author_name = input_text("Author Name")
	if not success: return

	success, text = backend.read_image_text(client, journal_image)
	if not success: return

	success, entries = backend.parse_entries(text_model, text)
	if not success: return

	success, author_description = backend.predict_author_description(image_model, author_image, author_name)
	if not success: return

	success, entry_prompts = backend.predict_all_image_prompts(text_model, entries, author_description, name="Author", max_scenes=2)
	if not success: return

	success, images = backend.predict_scene_images(entry_prompts, author_description, theme="poly")
	if not success: return

	success = display_vertex_images(images)
	if not success: return

# --- Start --- #

st.title("Diary to Storybook")

run_pipeline()

# Cleanup

if backend.google_api_credentials is not None:
	os.unlink(backend.google_api_credentials.name)