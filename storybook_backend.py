import streamlit as st
import tkinter as tk
from tkinter import filedialog

import os
from pathlib import Path
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

from backend.managers import (
    GeminiManager,
    VisionManager,
    ImagenManager,
    DiaryToStorybookManager,
)

from backend.utils import (
    generate_filename,
    upload_and_make_public,
    generate_qr_code,
    standard_image,
)

google_api_credentials = None

# tested and valid themes
artstyles = {
    "studio-ghibli": "Studio Ghibli Anime",
    "medieval-fantasy": "Photo-realistic Medieval Fantasy",
    "light-comic": "A dreamy scene with pastel colors and flowing lines in an impressionist style. Water color paints.",
    "poly": "A fragmented and geometric depiction of the subject, with overlapping planes and a multi-perspective approach, reminiscent of early 20th-century cubist art.",
    "cyberpunk": "Photo-realistic, futuristic and neon-lit scene with dark tones, glowing accents, and a blend of gritty urban decay and advanced technology. Sometimes there are robots and other futuristic devices and buildings.",
}

# tested and valid effects
effects = {"test": "Enlarge the subjects head 2 or 3 times for comedic effect"}


# Initializes the operating system's Google API Credentials
def set_google_api_credentials():
    # Automatically get credentials from first top level *.json file

    # Search the current directory for .json files
    current_directory = os.getcwd()
    json_files = [
        file for file in os.listdir(current_directory) if file.endswith(".json")
    ]

    if not json_files:
        print("No .json files found in the current directory.")
        return False, None

    # Select the first .json file found & set ENV variable
    credentials_file = os.path.join(current_directory, json_files[0])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

    global google_api_credentials
    google_api_credentials = credentials_file

    st.success("Google API Credentials Set...  \nFilename: " + credentials_file)
    return True, None


# Initializes the Generative Vertex AI Connection
def init_vertex_ai(project_id="cis-581-diary-to-storybook", location="us-central1"):
    vertexai.init(project=project_id, location=location)

    st.success(
        f"Vertex AI Initialized...  \nProject: {project_id}  \nServerLocation: {location}"
    )

    return True


# Initializes Gemini Manager & Respective Model
def init_model_managers():
    gemini_manager = GeminiManager()
    vision_manager = VisionManager()
    image_manager = ImagenManager()

    all_ = all([gemini_manager, vision_manager, image_manager])
    if all_:
        st.success(f"Intialized all Google Cloud models")

    return all_, (gemini_manager, vision_manager, image_manager)


def init_context_manager(managers, author_fn, journal_fn, name, theme=None, effect=None):
    try:
        ctx_manager = DiaryToStorybookManager(name, author_fn, journal_fn, theme=theme, effect=effect)
        ctx_manager.set_managers(*managers)
    except Exception as err:
        err = str(err)
        st.error("Error intializing DiaryToStorybook context manager\n" + err)
        return False, None

    st.success(f"Intialized DiaryToStorybook context manager")

    return True, ctx_manager


def generate_contexts(ctx, delay=0.1):
    # try:
    #     ctx.generate_context(delay=delay)
    # except Exception as err:
    #     err = str(err)
    #     st.error("Error generating context\n"+err)
    #     return False, None

    ctx.generate_context(delay=delay)

    st.success("Parsed inputs and generated necessary contexts")
    return True, None


def generate_images(ctx, delay=0.1):
    # try:
    #     ctx.generate_images(delay=delay)
    # except Exception as err:
    #     err = str(err)
    #     st.error("Error generating images\n"+err)
    #     return False, None

    ctx.generate_images(delay=delay)

    st.success(f"Successfully generated {len(ctx.images)} images")

    if len(ctx.image_errors) > 0:
        st.error(
            f"Error generating images:\n"
            + "\n".join([f"{idx} | {prompt}" for idx, prompt in ctx.image_errors])
        )

    return True, None


def save_and_upload(ctx):
    fn = generate_filename()

    output_dir = Path("output_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / fn

    try:
        standard_image(ctx.images, save_path=str(save_path))
        public_url = upload_and_make_public(
            "diary-to-storybook-images", str(save_path), fn
        )

        st.success(f"Successfully generated album image and uploaded to cloud")
        return True, (public_url, fn)
    except Exception as e:
        st.error(f"Error saving image and uploading to cloud: {str(e)}")
        return False, None


def display_qr_code(url, fn):
    # Use pathlib to construct the QR code path
    output_dir = Path("output_qr_codes")
    output_dir.mkdir(parents=True, exist_ok=True)
    qr_path = output_dir / f"qr_{fn}"

    try:
        # Pass the generated QR code path to the function
        generate_qr_code(url, str(qr_path))

        # Open the image file using PIL
        qr_image = Image.open(str(qr_path))
        st.image(qr_image, caption="QR Code", use_column_width=True)
        return True
    except Exception as e:
        st.error(f"Error generating QR code: {str(e)}")
        return False
