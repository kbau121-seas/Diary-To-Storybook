import streamlit as st
import tkinter as tk
from tkinter import filedialog

import os
import tempfile
import math

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

DEBUG=False
google_api_credentials = None

# tested and valid themes
artstyles = {
<<<<<<< Updated upstream
	"studio-ghibli" : "Studio Ghibli Anime",
    "medieval-fantasy" : "Photo-realistic Medieval Fantasy",
    "light-comic" : "A dreamy scene with pastel colors and flowing lines in an impressionist style. Water color paints.",
    "poly": "A fragmented and geometric depiction of the subject, with overlapping planes and a multi-perspective approach, reminiscent of early 20th-century cubist art.",
    "cyberpunk" : "Photo-realistic, futuristic and neon-lit scene with dark tones, glowing accents, and a blend of gritty urban decay and advanced technology. Sometimes there are robots and other futuristic devices and buildings.",
=======
    "studio-ghibli": "Studio Ghibli Anime",
    "light-comic": "A dreamy scene with pastel colors and flowing lines in an impressionist style. Water color paints.",
    "poly": "A fragmented and geometric depiction of the subject, with overlapping planes and a multi-perspective approach, reminiscent of early 20th-century cubist art.",
    "watercolor": "A soft and flowing watercolor painting with a dreamy and ethereal quality. Whimsical and light colors.",
    "oil-painting": "A rich and textured oil painting with deep colors and a realistic style. Thick brush strokes and a sense of depth.",
    "sketch": "A quick and loose pencil sketch with minimal detail and shading. A simple and expressive style.",
    
    # "medieval-fantasy": "Photo-realistic Medieval Fantasy",
    # "cyberpunk": "Photo-realistic, futuristic and neon-lit scene with dark tones, glowing accents, and a blend of gritty urban decay and advanced technology. Sometimes there are robots and other futuristic devices and buildings.",
    # "none": "Photo-realistic",
>>>>>>> Stashed changes
}

# tested and valid effects
effects = {
    "test" : "Enlarge the subjects head 2 or 3 times for comedic effect"
}

def cleanup():
	if google_api_credentials is not None:
		os.unlink(google_api_credentials.name)

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

	if DEBUG:
		st.success("Google API Credentials Set...  \nFilename: " + uploaded_file.name)

	return True, client

# Initializes the Generative Vertex AI Connection
def init_vertex_ai(
	project_id="cis-581-project-testing",
	location="us-central1"
	):
	vertexai.init(project=project_id, location=location)

	if DEBUG:
		st.success(f"Vertex AI Initialized...  \nProject: {project_id}  \nServerLocation: {location}")

	return True

# Initializes a Generative Vertex AI Model
def init_vertex_ai_generative_model(model_id, model_title=""):
	model = GenerativeModel(model_id)

	if (len(model_title) > 0): model_title = model_title.strip() + " "

	if (DEBUG and model != None):
		st.success(f"Generative Vertex AI {model_title} Model Set...  \nModel: {model_id}")

	return (model != None), model

# Initializes a Generative Vertex AI Text Model
def init_vertex_ai_text_model(model_id="gemini-1.5-pro-002"):
	return init_vertex_ai_generative_model(model_id, "Text")

# Initializes a Generative Vertex AI Image Model
def init_vertex_ai_image_model(model_id="gemini-1.0-pro-vision"):
	return init_vertex_ai_generative_model(model_id, "Image")

# Reads text from an image
def read_image_text(client, content):
	image = vision.Image(content=content)

	response = client.text_detection(image=image)
	annotations = response.text_annotations

	if not annotations: return False, None

	if DEBUG:
		st.success("Successfully read the input image")
		st.markdown(annotations[0].description.replace("\n", "  \n"))

	return True, annotations[0].description

# Splits multiple journal entries apart
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

	if DEBUG:
		st.success(f"Successfully split {entries_count} entries")
		for entry_idx in range(entries_count):
			st.markdown(f"Entry #{entry_idx + 1}:")
			st.markdown(entries[entry_idx].replace("\n", "  \n"))

	return True, entries

# Describes an author's image
def predict_author_description(model, image, name="Author", prompt = None):
	if not prompt:
		prompt = "I am blind. describe what the subject looks like visually.\
			I only want a pure description of the SUBJECT.\
			If its a person, be highly descriptive of facial features.\
			Information on the composition such as 'headshot' or 'panoramic' is not necessary."

	prompt += f" The subject's name is {name}"

	vertex_image = vertexai.preview.generative_models.Image.from_bytes(image)
	response = model.generate_content([prompt, vertex_image])

	if not response.candidates: return False, None
	if not response.candidates[0].content.parts: return False, None

	if DEBUG:
		st.success("Successfully described the author image...")
		st.markdown(response.candidates[0].content.parts[0].text.replace("\n", "  \n"))

	return True, response.candidates[0].content.parts[0].text

# Generates an image prompt based on a journal entry
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

# Generates a set of image prompts based on a journal entry
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

	prompt = "\n".join(prompt)

	response = chat_session.send_message(prompt)

	if not response.candidates: return False, None
	if not response.candidates[0].content.parts: return False, None

	return True, response.candidates[0].content.parts[0].text

def predict_all_image_prompts(model, entries, author_description, min_scenes=1, max_scenes=3, name="Main Character"):
	prompts = []
	for entry in entries:
		success, entry_prompts = predict_image_prompts(model, entry, author_description, name="Author", max_scenes=2)

		if not success: continue

		prompts += [entry_prompts]

	if len(prompts) <= 0: return False, None

	split_scenes = [scene_prompts.split("<SCENE>") for scene_prompts in prompts]

	scene_count = 0
	for entry in split_scenes:
		for scene in entry:
			scene_count += 1

	if DEBUG:
		st.success(f"Generated {scene_count} Scenes...")

		for entry in split_scenes:
			for scene in entry:
				st.markdown(scene)

	return True, split_scenes


# Generates an image from a prompt
def imagen_generate_images(prompt, author_description, theme=None, effect=None, number_of_images=1, aspect_ratio="1:1", fast=True):
	if fast:
		fast_ = "imagen-3.0-fast-generate-001"
		fast_generation_model = ImageGenerationModel.from_pretrained(fast_)
		generation_model = fast_generation_model
	else:
		standard_ = "imagen-3.0-generate-001"
		standard_generation_model = ImageGenerationModel.from_pretrained(standard_)
		generation_model = standard_generation_model

	gen_prompt = [
		f"Description of subject: {author_description}",
        prompt
	]

	if theme and theme in artstyles:
		gen_prompt.append(f"Draw the image in the artstyle of {artstyles[theme]}. This is nonegotiable")
	else:
		gen_prompt.append(f"Create this image in photo-realism. This is nonegotiable")
	
	if effect and effect in effects:
		gen_prompt.append(effects[effect])

	gen_prompt = "\n".join(gen_prompt)

	response = generation_model.generate_images(
		prompt=gen_prompt,
		number_of_images=number_of_images,
		aspect_ratio=aspect_ratio
		)

	return response.images

# Generates an image for each scene
def predict_scene_images(scenes, author_description, theme=None, effect=None):
	images = []
	fail_idx = []
	fail_count = 0

	idx = 0

	for entry in scenes:
		for scene in entry:
			prompt = scene.strip()
			if len(prompt) > 0:
				response = imagen_generate_images(prompt, author_description, theme, effect)
				if response != []:
					images.append(response[0])
				else:
					fail_count += 1
					fail_idx.append(idx)
				idx += 1

	if DEBUG:
		st.success(f"Generated {len(images)} images... {fail_count} Fails")

	return True, images, fail_idx

def predict_prompt_image(prompt, author_description, theme=None, effect=None):
	response = imagen_generate_images(prompt, author_description, theme, effect)
	if response != []:
		return True, response[0]
	else:
		return False, None
