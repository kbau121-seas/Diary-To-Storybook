import storybook_backend as backend

import streamlit as st
from st_clickable_images import clickable_images

import tkinter as tk
from tkinter import filedialog

import io
import os
import tempfile
import math

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import base64

DEBUG=False
backend.DEBUG=DEBUG

# Requests an image from the user
def input_image(title="Input Image", types=['png', 'jpg']):
	uploaded_file = st.file_uploader(title, types)
	if (uploaded_file is None):
		return False, None

	file_data = uploaded_file.read()
	file_size_MB = round(len(file_data) / (1024 ** 2), 1)

	if DEBUG:
		st.success("Upload Successful...  \nFilename: " + uploaded_file.name + "  \nFile Size: " + str(file_size_MB) + "MB")

	st.image(file_data)
	return True, file_data

def input_text(label="Input Text"):
	text_box = st.text_input(label)

	return text_box != None and text_box.strip() != "", text_box

def wait_for_button(label="Button"):
	return st.button(label)

def select_artstyle(default=None, key=None):
	index = None
	if default in list(backend.artstyles):
		index = list(backend.artstyles).index(default)

	artstyle = st.selectbox(
		"Art Style",
		[style.capitalize() for style in list(backend.artstyles)],
		index=index,
		placeholder="Select Art Style...",
		key=key
	)
	if artstyle == None: return False, None

	artstyle = artstyle.lower()
	return True, artstyle

def edit_image_form(images):
	clicked = clickable_images(
		images,
		titles=["" for i in range(len(images))],
		div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
		img_style={"cursor":"pointer", "margin": "5px", "height": "200px"},
	)

	if 'prompts' not in st.session_state: return False
	if 'author_descriptions' not in st.session_state: return False
	if 'artstyles' not in st.session_state: return False

	prompts = st.session_state.prompts
	author_descriptions = st.session_state.author_descriptions
	artstyles = st.session_state.artstyles

	if (clicked < 0): return
	editted_prompt = st.text_area("Prompt", prompts[clicked])
	editted_author_description = st.text_area("Author Description", author_descriptions[clicked])
	_, editted_artstyles = select_artstyle(default=artstyles[clicked], key="edit_artstyle")

	do_regenerate = wait_for_button("Regenerate")
	if do_regenerate:
		st.session_state.prompts[clicked] = editted_prompt
		st.session_state.author_descriptions[clicked] = editted_author_description
		st.session_state.artstyles[clicked] = editted_artstyles

		success, new_image = backend.predict_prompt_image(editted_prompt, editted_author_description, editted_artstyles)
		if not success: return

		save_vertex_image(new_image, clicked)

		new_image = base64_to_pil(vertex_image_to_base64(new_image))
		draw_number(new_image, clicked + 1)

		st.session_state.images[clicked] = f"data:image/png;base64,{pil_to_base64(new_image)}"

	return True

# Image manipulation
def pil_to_base64(image, format="png"):
	buffered = io.BytesIO()
	image.save(buffered, format=format)
	return base64.b64encode(buffered.getvalue()).decode()

def base64_to_pil(encoded):
	data = base64.b64decode(encoded)
	return Image.open(io.BytesIO(data))

def vertex_image_to_base64(image, format="png"):
	buffered = io.BytesIO()
	image._pil_image.save(buffered, format=format)
	return base64.b64encode(buffered.getvalue()).decode()

def clear(image_dir):
	image_paths = get_image_paths(image_dir)

	for file in image_paths:
		os.remove(file)
	return []

def save_vertex_images(images, image_dir="images", format='png'):
	image_dir = os.path.join(os.getcwd(), image_dir)
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)
	else:
		clear(image_dir)

	for idx, image in enumerate(images):
		file = os.path.join(image_dir, f"Panel_{idx}.{format}")
		image._pil_image.save(file, format=format)

	return True

def save_vertex_image(image, idx, image_dir="images", format="png"):
	image_dir = os.path.join(os.getcwd(), image_dir)
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)

	file = os.path.join(image_dir, f"Panel_{idx}.{format}")
	image._pil_image.save(file, format=format)

	return True

def get_image_paths(image_dir="images"):
	image_dir_root = image_dir
	image_dir = os.path.join(os.getcwd(), image_dir)
	if not os.path.exists(image_dir):
		return []

	image_paths = os.listdir(image_dir)
	return [os.path.join(image_dir_root, local_file) for local_file in image_paths]

def draw_number(image, number, font_type="./arial.ttf", font_size=90):
	font = ImageFont.truetype(font_type, font_size)
	draw = ImageDraw.Draw(image)
	draw.text((5, 0), str(number), (255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))

def init_style():
	st.markdown("""
	<style>
	div.stButton {text-align:center}
	</style>
	""", unsafe_allow_html=True)

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

	success, artstyle = select_artstyle()
	if not success: return

	do_generate = wait_for_button("Generate")
	if do_generate:
		# Generate new images

		success, text = backend.read_image_text(client, journal_image)
		if not success: return

		success, entries = backend.parse_entries(text_model, text)
		if not success: return

		success, author_description = backend.predict_author_description(image_model, author_image, author_name)
		if not success: return

		success, entry_prompts = backend.predict_all_image_prompts(text_model, entries, author_description, name="Author", max_scenes=2)
		if not success: return

		success, images = backend.predict_scene_images(entry_prompts, author_description, theme=artstyle)
		if not success: return

		success = save_vertex_images(images)
		if not success: return

		images = [base64_to_pil(vertex_image_to_base64(image)) for image in images]

		for i, image in enumerate(images):
			draw_number(image, i + 1)

		images = [f"data:image/png;base64,{pil_to_base64(image)}" for image in images]
		st.session_state.images = images
		st.session_state.prompts = [prompt.strip() for prompts in entry_prompts for prompt in prompts]
		st.session_state.author_descriptions = [author_description for image in images]
		st.session_state.artstyles = [artstyle for image in images]

	elif 'images' not in st.session_state:
		# Load the stored images

		image_paths = get_image_paths()
		images = []

		font = ImageFont.truetype("./arial.ttf", 90)
		for i, file in enumerate(image_paths):
			img = Image.open(file)
			draw_number(img, i + 1)

			encoded = pil_to_base64(img)
			images.append(f"data:image/png;base64,{encoded}")

		st.session_state.images = images

	else:
		# Access the state variable images
		images = st.session_state.images

	success = edit_image_form(images)
	if not success: return

# --- Start --- #

init_style()

st.title("Diary to Storybook")

run_pipeline()

# Cleanup
backend.cleanup()