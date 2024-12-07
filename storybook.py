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
# def input_image(title="Input Image", types=["png", "jpg"]):
#     uploaded_file = st.file_uploader(title, types)
#     if uploaded_file is None:
#         return False, None

#     file_data = uploaded_file.read()
#     file_size_MB = round(len(file_data) / (1024**2), 1)

#     st.success(
#         "Upload Successful...  \nFilename: "
#         + uploaded_file.name
#         + "  \nFile Size: "
#         + str(file_size_MB)
#         + "MB"
#     )

#     st.image(file_data)
#     return True, file_data


def input_image(title="Input Image", types=["png", "jpg"]):
    uploaded_file = st.file_uploader(title, types)
    if uploaded_file is None:
        return False, None

    # Save the file to a temporary directory
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
    ) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    file_size_MB = round(os.path.getsize(file_path) / (1024**2), 1)

    st.success(
        f"Upload Successful...  \nFilename: {uploaded_file.name}  \nFile Size: {file_size_MB}MB"
    )

    st.image(file_path)
    return True, file_path


def input_text(label="Input Text"):
    text_box = st.text_input(label)

    return text_box != None and text_box.strip() != "", text_box


# Displays an array of Vertex AI Images in order
def display_vertex_images(images):
    images = [img for img in images if img != None]
    columns = 3
    l, m, r = st.columns(columns)
    cols = [l, m, r]

    rows = math.ceil(len(images) / columns)

    for row in range(rows):
        for col in range(columns):
            idx = col + row * columns
            if idx >= len(images):
                break

            b_img = io.BytesIO()
            images[idx][0]._pil_image.save(b_img, format="PNG")
            image_bytes = b_img.getvalue()

            cols[col].image(image_bytes)

    return True


# Runs the diary to storybook pipeline
def run_pipeline():
    # Automatic
    success, client = backend.set_google_api_credentials()
    if not success:
        return

    success = backend.init_vertex_ai()
    if not success:
        return

    success, (gemini_manager, vision_manager, imagen_manager) = (
        backend.init_model_managers()
    )
    if not success:
        return

    success, journal_image = input_image("Journal Image")
    if not success:
        return

    success, author_image = input_image("Author Image")
    if not success:
        return

    success, author_name = input_text("Author Name")
    if not success:
        return

    success, ctx = backend.init_context_manager(
        (gemini_manager, vision_manager, imagen_manager),
        author_image,
        journal_image,
        author_name,
    )
    if not success:
        return

    # TODO : add inputs for set theme

    # TODO : add inputs for set effects

    # TODO : add slider (between 1 and 3-4?) to allow for # of images
    # generated per daily entry

    success, _ = backend.generate_contexts(ctx, delay=0.1)
    if not success:
        return

    success, _ = backend.generate_images(ctx, delay=0.25)
    if not success:
        return

    success = display_vertex_images(ctx.images)
    if not success:
        return

    # TODO : Modify indidivual images as needed

    success, (url, fn) = backend.save_and_upload(ctx)
    if not success:
        return

    success = backend.display_qr_code(url, fn)
    if not success:
        return


# --- Start --- #

st.title("Diary to Storybook")

run_pipeline()

# Cleanup

# if backend.google_api_credentials is not None:
# 	os.unlink(backend.google_api_credentials)
