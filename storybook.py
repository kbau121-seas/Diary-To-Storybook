import streamlit as st
import tkinter as tk
from tkinter import filedialog

import os
import tempfile

import requests
import google.auth
from google.cloud import vision, aiplatform
from google.cloud.vision_v1 import types
import vertexai
from vertexai.preview.language_models import ChatModel
from vertexai.preview.generative_models import GenerativeModel, Image
from vertexai.preview.vision_models import ImageGenerationModel
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline
import io
from PIL import Image

import base64

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

# Initializes a Generative Vertex AI Model
def init_vertex_ai(
	project_id="cis-581-project-testing",
	location="us-central1"
	):
	vertexai.init(project=project_id, location=location)

	st.success(f"Vertex AI Initialized...  \nProject: {project_id}  \nServerLocation: {location}")

	return True

def init_vertex_ai_generative_model(model_id, model_title=""):
	model = GenerativeModel(model_id)

	if (len(model_title) > 0): model_title = model_title.strip() + " "

	if (model != None):
		st.success(f"Generative Vertex AI {model_title} Model Set...  \nModel: {model_id}")

	return (model != None), model


def init_vertex_ai_text_model(model_id="gemini-1.5-pro-002"):
	return init_vertex_ai_generative_model(model_id, "Text")

def init_vertex_ai_image_model(model_id="gemini-1.0-pro-vision"):
	return init_vertex_ai_generative_model(model_id, "Image")

# Reads text from an image
def read_image_text(client, content):
	image = vision.Image(content=content)

	response = client.text_detection(image=image)
	annotations = response.text_annotations

	if not annotations: return False, None

	st.success("Successfully read the input image")
	st.markdown(annotations[0].description.replace("\n", "  \n"))

	return True, annotations[0].description

def parse_entries(model, text):
	prompt = f"The following text is a collection of 1 or more journal entries.\
		Keep the journal entries as they are, but in between each days' entry insert\
		' <ENTRY> ' so that I can split each entry for easy text manipulation.\
		\n\n{text}"

	response = model.generate_content(prompt)

	if not response.candidates: return False, None
	if not response.candidates[0].content.parts: return False, None

	delimited_text = response.candidates[0].content.parts[0].text

	entries = [token.strip() for token in delimited_text.split("<ENTRY>") if len(token.strip()) > 0]
	entries_count = len(entries)
	if entries_count <= 0: return False, None

	st.success(f"Successfully split {entries_count} entries")
	for entry_idx in range(entries_count):
		st.markdown(f"Entry #{entry_idx + 1}:")
		st.markdown(entries[entry_idx].replace("\n", "  \n"))

	return True, entries

def predict_author_description(model, image, name="Author"):
	prompt = "I am blind. describe what the subject looks like visually.\
		I only want a pure description of the SUBJECT.\
		If its a person, be highly descriptive of facial features.\
		Information on the composition such as 'headshot' or 'panoramic' is not necessary."

	prompt += f" The subject's name is {name}"

	vertex_image = vertexai.preview.generative_models.Image.from_bytes(image)
	response = model.generate_content([prompt, vertex_image])

	if not response.candidates: return False, None
	if not response.candidates[0].content.parts: return False, None

	st.success("Successfully described the author image...")
	st.markdown(response.candidates[0].content.parts[0].text.replace("\n", "  \n"))

	return True, response.candidates[0].content.parts[0].text

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

def predict_image_prompt(model, author_description, context, name="Author"):
	prompt = "\n".join([
		f"The author's name is: {name}",
		"",
		f"A visual description of the author is: {author_description}",
		"",
		f"You are generating a prompt for an image generation model.\
		Using the description of the author create a prompt that would generate an image\
		of the author in the following context: {context}"
	])

	response = model.generate_content(prompt)

	if not response.candidates: return False, None
	if not response.candidates[0].content.parts: return False, None

	return True, response.candidates[0].content.parts[0].text

def predict_image_prompts(model, journal_entry, author_description, min_scenes=1, max_scenes=3, name="Main Character"):
	if min_scenes > max_scenes:
		raise ValueError("min_scenes cannot be greater than max_scenes")
	if min_scenes < 1 or max_scenes < 1:
		raise ValueError("min_scenes and max_scenes must be greater than 0")
	if max_scenes > 4:
		raise ValueError("max_scenes cannot be greater than 4")

	model = GenerativeModel("gemini-1.5-pro-002")
	chat_session = model.start_chat()

	prompt = [
	    "Forget any previous information or context.",
        f"The author's name is: {name}",
        "",
        f"A visual description of the author is: {author_description}",
        "",
        f"The following/given journal entry text will be used to create a story.\
        Ignore prior or future context of other scenes. \
        ONLY DESCRIBE WHAT IS HAPPENING IN THE CURRENT PROVIDED TEXT:\
          Summarize and split said text into minimum {min_scenes} scenes to {max_scenes} max scenes.\
          It is VERY important to stay within these number of scenes.",
        f"Each scene \
          should be split with '<SCENE> ' in your response. Each scene response will be used as the prompt\
          input for an image generation model.",
        f"The author's name is {name} and a visual description of them is as follows: {author_description}.",
        f"Avoid using words that will trigger content moderation filters in image generation models. (avoid explicit and sensitive terms, use neutral language, etc)",
        f"Create the prompts for each scene in the following text. Do not include any other heading text (such as scene count, etc):\
          \n\n{journal_entry}"
    ]

	st.markdown(prompt)

	prompt = "\n".join(prompt)

	response = chat_session.send_message(prompt)

	if not response.candidates: return False, None
	if not response.candidates[0].content.parts: return False, None

	return True, response.candidates[0].content.parts[0].text

def imagen_generate_images(prompt, number_of_images=1, aspect_ratio="1:1", fast=True):
	if fast:
		fast_ = "imagen-3.0-fast-generate-001"
		fast_generation_model = ImageGenerationModel.from_pretrained(fast_)
		generation_model = fast_generation_model
	else:
		standard_ = "imagen-3.0-generate-001"
		standard_generation_model = ImageGenerationModel.from_pretrained(standard_)
		generation_model = standard_generation_model

	response = generation_model.generate_images(
		prompt=prompt,
		number_of_images=number_of_images,
		aspect_ratio=aspect_ratio
		)

	return response.images

def predict_scene_images(scenes):
	images = []
	fail_count = 0

	for entry in scenes:
		for scene in entry:
			prompt = scene.strip()
			if len(prompt) > 0:
				response = imagen_generate_images(prompt, 1)
				if response != []:
					images.append(response)
				else:
					fail_count += 1

	st.success(f"Generated {len(images)} images... {fail_count} Fails")

	return True, images

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

def display_images(images):
	for image in images:
		b_img = io.BytesIO()
		image[0]._pil_image.save(b_img, format='PNG')
		image_bytes = b_img.getvalue()
		st.image(image_bytes)

	return True

# Runs the diary to storybook pipeline
def run_pipeline():
	success, client = set_google_api_credentials()
	if not success: return

	success = init_vertex_ai()
	if not success: return

	success, text_model = init_vertex_ai_text_model()
	if not success: return

	success, image_model = init_vertex_ai_image_model()
	if not success: return

	success, journal_image = input_image("Journal Image")
	if not success: return

	success, author_image = input_image("Author Image")
	if not success: return

	success, text = read_image_text(client, journal_image)
	if not success: return

	success, entries = parse_entries(text_model, text)
	if not success: return

	# TODO author name
	success, author_description = predict_author_description(image_model, author_image, "Author")
	if not success: return

	st.success(f"Generating Prompts...")
	prompts = []
	for entry in entries:
		success, entry_prompts = predict_image_prompts(text_model, author_description, entry, name="Author", max_scenes=2)
		prompts += [entry_prompts]

		if not success: return

	split_scenes = [scene_prompts.split("<SCENE>") for scene_prompts in prompts]
	st.success(f"Generated {len(split_scenes)} Scenes...")

	for entry in split_scenes:
		for scene in entry:
			st.markdown(scene)

	success, images = predict_scene_images(split_scenes)
	if not success: return

	success = display_images(images)
	if not success: return

# --- Start --- #

st.title("Diary to Storybook")

run_pipeline()

# Cleanup

if google_api_credentials is not None:
	os.unlink(google_api_credentials.name)