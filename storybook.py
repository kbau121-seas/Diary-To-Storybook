import storybook_backend as backend

import streamlit as st
from st_clickable_images import clickable_images

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

def wait_for_button(label="Button"):
  return st.button(label)

def edit_image_form(ctx):
  images = [f"data:image/png;base64,{image}" for image in ctx.images]
  clicked = clickable_images(
      images,
      div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
      img_style={"cursor":"pointer", "margin": "5px", "height": "200px"},
    )

  if clicked < 0: return False

  st.image(images[clicked])
  editted_prompt = st.text_area("Prompt", ctx.editable_prompts[clicked])

  do_regenerate = wait_for_button("Regenerate")
  if do_regenerate:
    ctx.regenerate_single_image(clicked, editted_prompt)
    push_state(ctx)
    st.rerun()

def check_state():
  return {
    'images',
    'image_prompts',
    'editable_prompts',
    'author_description',
    'theme',
    'effect'
  }.issubset(st.session_state)

def push_state(ctx):
  st.session_state.images = ctx.images
  st.session_state.image_prompts = ctx.image_prompts
  st.session_state.editable_prompts = ctx.editable_prompts
  st.session_state.author_description = ctx.author_description
  st.session_state.theme = ctx.theme
  st.session_state.effect = ctx.effect

def pull_state(ctx):
  ctx.images = st.session_state.images
  ctx.image_prompts = st.session_state.image_prompts
  ctx.editable_prompts = st.session_state.editable_prompts
  ctx.author_description = st.session_state.author_description
  ctx.theme = st.session_state.theme
  ctx.effect = st.session_state.effect

def init_style():
  st.markdown("""
  <style>
  div.stButton {text-align:center}
  </style>
  """, unsafe_allow_html=True)

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

    # TODO : add inputs for set theme
    success, artstyle = select_artstyle()
    if not success:
        return

    success, ctx = backend.init_context_manager(
        (gemini_manager, vision_manager, imagen_manager),
        author_image,
        journal_image,
        author_name,
        theme=artstyle
    )
    if not success:
        return

    # TODO : add inputs for set effects

    # TODO : add slider (between 1 and 3-4?) to allow for # of images
    # generated per daily entry

    do_generate = wait_for_button("Generate")
    if do_generate:
        # Generate new images

        success, _ = backend.generate_contexts(ctx, delay=0.1)
        if not success:
            return

        success, _ = backend.generate_images(ctx, delay=0.25)
        if not success:
            return

        push_state(ctx)

    if not check_state(): return

    if len(st.session_state.images) <= 0: return
    pull_state(ctx)

    # TODO : Modify indidivual images as needed
    edit_image_form(ctx)

    return

    success, (url, fn) = backend.save_and_upload(ctx)
    if not success:
        return

    success = backend.display_qr_code(url, fn)
    if not success:
        return


# --- Start --- #

init_style()

st.title("Diary to Storybook")

run_pipeline()

# Cleanup

# if backend.google_api_credentials is not None:
#   os.unlink(backend.google_api_credentials)
