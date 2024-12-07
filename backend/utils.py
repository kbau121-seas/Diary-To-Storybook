import random
import os
import string

from google.cloud import storage

import qrcode

from matplotlib import pyplot as plt

import io
import base64

from PIL import (
    Image,
    ImageFont,
    ImageDraw
)


def generate_filename(n: int = 6) -> str:
    """Generate a random string"""
    return (
        f"gen_image_{''.join([random.choice(string.hexdigits) for _ in range(n)])}.png"
    )


# Save and view album
def standard_image(images, cols=3, save_path=None):

    if not save_path:
        raise Exception("No save path provided")

    images = [img for img in images if img != None]

    # Add numberings
    images = [b64_to_pil(image) for image in images]
    for idx, image in enumerate(images): draw_number(image, idx + 1)
    images = [pil_to_b64(image) for image in images]

    # Number of columns in the grid
    num_cols = cols  # Adjust based on your preference
    num_images = len(images)
    num_rows = (
        num_images + num_cols - 1
    ) // num_cols  # Calculate number of rows needed

    # Set up the figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    # Display each image in the grid
    for i, img_wrapper in enumerate(images):
        if img_wrapper is None:
            axes[i].axis("off")
            continue

        # Get PIL image from base64 image
        img = b64_to_pil(img_wrapper)

        # Display the image
        axes[i].imshow(img)
        axes[i].axis("off")

    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(
            save_path, format="png", dpi=300
        )  # Save as PNG with high resolution
        print(f"Plot saved to {save_path}")


def upload_and_make_public(bucket_name, source_file_name, destination_blob_name) -> str:
    """Uploads a file to the G-Cloud bucket and makes it publicly accessible."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(source_file_name)

    # Make the object public
    blob.make_public()

    print(
        f"File uploaded to {destination_blob_name} and is publicly accessible at {blob.public_url}"
    )
    return blob.public_url


def generate_qr_code(data, output_path):
    """
    Generate a QR code from the given data and save it to a file.

    Args:
        data (str): The data to encode in the QR code (e.g., a public URL).
        output_path (str): The path to save the QR code image.
    """

    qr = qrcode.QRCode(
        version=1,  # Controls the size of the QR Code
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
        box_size=10,  # Size of each box in the QR code
        border=4,  # Border size (minimum is 4)
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Create an image from the QR Code instance
    img = qr.make_image(fill_color="black", back_color="white")

    # Save the image
    img.save(output_path)
    print(f"QR code saved to {output_path}")

def generated_to_b64(image, format="png"):
    buffered = io.BytesIO()
    image._pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

def b64_to_pil(encoded):
    data = base64.b64decode(encoded)
    return Image.open(io.BytesIO(data))

def pil_to_b64(image, format="png"):
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

def draw_number(image, number, font_type="./arial.ttf", font_size=90):
    font = ImageFont.truetype(font_type, font_size)
    draw = ImageDraw.Draw(image)
    draw.text((5, 0), str(number), (255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
