import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import vision

import numpy as np
import cv2

import io
import time

from backend.utils import (
    generated_to_b64
)

class GeminiManager:
    """
    # Initialize vertex AI
      self.vertexai.init(project=project_id, location=location)

      MUST BE CALLED BEFORE ANY VERTEXAI MODELS ARE CREATED
    """

    def __init__(self):
        # Define and initialize a generative text model
        self.gemini_text_model = GenerativeModel("gemini-1.5-pro-002")
        self.gemini_vision_model = GenerativeModel(model_name="gemini-1.0-pro-vision")

    def parse_entries(self, text: str) -> str:
        prompt = f"""
    The following text is a collection of 1 or more journal entries.
    Keep the journal entries as they are, but in between each days' entry insert ' <ENTRY> ' so that I can split each entry for easy text manipulation.
    Text:
    {text}
    """.strip()

        response = self.gemini_text_model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text

    def split_entries_text(self, text: str):
        return [
            token.strip() for token in text.split("<ENTRY>") if len(token.strip()) > 0
        ]

    def summarize(self, text: str, name: str = "'Main Character'") -> str:
        """
        Splits & summarizes a single journal entry into 2 scenes.
        It utilizes the author description to directly generate image prompts.
        It also uses context of the character name.
        """

        prompt = f"""
    Forget any previous information or context.

    The author's name is: {name}

    The following/given journal entry text will be used to create a story. Ignore prior or future context of other scenes. ONLY DESCRIBE WHAT IS HAPPENING IN THE CURRENT PROVIDED TEXT.

    Summarize and split said text into 2 scenes. Each scene should be split with ' <SCENE> ' in your response. Each scene response will be used as the prompt input for an image generation model.

    The author's name is {name}.
    Specify the author/subject's expressions to match the context of the scene. Always describe and emphasize the subject doing the action described in the scene.

    Avoid using words that will trigger content moderation filters in image generation models. (avoid explicit and sensitive terms, use neutral language, etc)

    Create the prompts for each scene in the following text. Do not include any other heading text (such as scene count, etc).
    Text:

    {text}
    """.strip()

        response = self.gemini_text_model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text

    def author_description(self, file, name=None):
        prompt = f"""
    I am blind. Describe what the subject looks like visually.
    I only want a pure description of the SUBJECT. If its a person be highly descriptive of facial features.
    Information on the composition such as 'headshot' or 'panoramic' is not necessary.
    """.strip()

        if not name:
            prompt += " The subject will be referred to as the Main Character"
        else:
            prompt += f" The subject's name is {name}"

        image = self.load_image(file)

        response = self.gemini_vision_model.generate_content([prompt, image])
        return response.candidates[0].content.parts[0].text

    def load_image(self, file):
        return vertexai.preview.generative_models.Image.load_from_file(file)


class VisionManager:
    def __init__(self):
        # Initialize the Vision API client
        self.client = vision.ImageAnnotatorClient()

    def text_detection(self, image_path: str, imcv=False):
        """
        Detects text in the file.
        """

        # Load the image as content
        with io.open(image_path, "rb") as image_file:
            content = image_file.read()

        # Create an image object from the uploaded content
        image = vision.Image(content=content)

        if imcv:
            # If using file upload, load it into OpenCV format
            nparr = np.frombuffer(content, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # OpenCV image
        else:
            img_cv = None

        # Perform text detection
        response = self.client.text_detection(image=image)
        texts = response.text_annotations

        # # Print detected text
        # if texts:
        #   print(f"Detected text: {texts[0].description}")
        # else:
        #   print("No text detected")

        # Error handling
        if response.error.message:
            raise Exception(f"{response.error.message}")

        if texts:
            return (True, texts[0].description, img_cv)
        else:
            return (False, "No text detected", img_cv)

    def multi_image_text_detection(self, image_paths: list, im_cv=False):
        return [self.text_detection(img_path, im_cv) for img_path in image_paths]

    class ImagenManager:
        def __init__(self):
            self.model_fast = ImageGenerationModel.from_pretrained(
                "imagen-3.0-fast-generate-001"
            )
            self.model_standard = ImageGenerationModel.from_pretrained(
                "imagen-3.0-generate-001"
            )

            self.artstyles = {
                "studio-ghibli": "Studio Ghibli Anime",
                "medieval-fantasy": "Photo-realistic Medieval Fantasy",
                "light-comic": "A dreamy scene with pastel colors and flowing lines in an impressionist style. Water color paints.",
                "poly": "A fragmented and geometric depiction of the subject, with overlapping planes and a multi-perspective approach, reminiscent of early 20th-century cubist art.",
                "cyberpunk": "Photo-realistic, futuristic and neon-lit scene with dark tones, glowing accents, and a blend of gritty urban decay and advanced technology. Sometimes there are robots and other futuristic devices and buildings.",
            }

            self.effects = {
                "test": "Enlarge the subjects head 2 or 3 times for comedic effect"
            }

        def generate_image(
            self, prompt: str, author_desc: str, theme=None, effect=None, fast=True
        ):
            gen_prompt = [f"Description of subject: {author_desc}", prompt]

            if theme and theme in self.artstyles:
                gen_prompt.append(
                    f"Draw the image in the artstyle of {self.artstyles[theme]}. This is nonegotiable.\n"
                )
            else:
                gen_prompt.append(
                    f"Create this image in photo-realism. This is nonegotiable.\n"
                )

            if effect and effect in self.effects:
                gen_prompt.append(self.effects[effect])

            gen_prompt = "\n".join(gen_prompt)

            if fast:
                img_model = self.model_fast
            else:
                img_model = self.model_standard

            response = img_model.generate_images(
                prompt=gen_prompt,
                number_of_images=1,
                aspect_ratio="1:1",
                # safety_filter_level="block_some",
                # person_generation="allow_all",
            )
            return response.images, gen_prompt.strip()

        def load_image(self, file):
            return vertexai.preview.generative_models.Image.load_from_file(file)


class ImagenManager:
    def __init__(self):
        self.model_fast = ImageGenerationModel.from_pretrained(
            "imagen-3.0-fast-generate-001"
        )
        self.model_standard = ImageGenerationModel.from_pretrained(
            "imagen-3.0-generate-001"
        )

        self.artstyles = {
            "studio-ghibli": "Studio Ghibli Anime",
            "medieval-fantasy": "Photo-realistic Medieval Fantasy",
            "light-comic": "A dreamy scene with pastel colors and flowing lines in an impressionist style. Water color paints.",
            "poly": "A fragmented and geometric depiction of the subject, with overlapping planes and a multi-perspective approach, reminiscent of early 20th-century cubist art.",
            "cyberpunk": "Photo-realistic, futuristic and neon-lit scene with dark tones, glowing accents, and a blend of gritty urban decay and advanced technology. Sometimes there are robots and other futuristic devices and buildings.",
        }

        self.effects = {
            "test": "Enlarge the subjects head 2 or 3 times for comedic effect"
        }

    def generate_image(
        self, prompt: str, author_desc: str, theme=None, effect=None, fast=True
    ):
        gen_prompt = [f"Description of subject: {author_desc}", prompt]

        if theme and theme in self.artstyles:
            gen_prompt.append(
                f"Draw the image in the artstyle of {self.artstyles[theme]}. This is nonegotiable.\n"
            )
        else:
            gen_prompt.append(
                f"Create this image in photo-realism. This is nonegotiable.\n"
            )

        if effect and effect in self.effects:
            gen_prompt.append(self.effects[effect])

        gen_prompt = "\n".join(gen_prompt)

        if fast:
            img_model = self.model_fast
        else:
            img_model = self.model_standard

        response = img_model.generate_images(
            prompt=gen_prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            # safety_filter_level="block_some",
            # person_generation="allow_all",
        )
        return response.images, gen_prompt.strip()

    def load_image(self, file):
        return vertexai.preview.generative_models.Image.load_from_file(file)


class DiaryToStorybookManager:
    def __init__(
        self,
        name: str,
        author_path: str,
        journal_path: str,
        theme=None,
        effect=None,
    ):

        self.name = name
        self.theme = theme
        self.effect = None

        self.author_path = author_path
        self.journal_path = journal_path

        self.author_description = None

        self.split_scenes = None
        self.image_prompts = []
        self.editable_prompts = []
        self.images = []
        self.image_errors = []

        self.public_url = None
        self.qr_code_path = None

        self.errors = []

        self.gemini_manager = None
        self.vision_manager = None
        self.image_manager = None

    def set_managers(self, gemini_manager, vision_manager, image_manager):
        self.gemini_manager = gemini_manager
        self.vision_manager = vision_manager
        self.image_manager = image_manager

    def require_managers(method):
        def wrapper(self, *args, **kwargs):
            # Check if all managers are defined
            if not all([self.gemini_manager, self.vision_manager, self.image_manager]):
                raise ValueError(
                    "Required managers are not set. Ensure gemini_manager, "
                    "vision_manager, and image_manager are initialized before calling this method."
                )
            return method(self, *args, **kwargs)

        return wrapper

    @require_managers
    def generate_context(self, delay=0, debug=False):
        # Get text from journal
        text_from_journal_image = self.vision_manager.text_detection(self.journal_path)[
            1
        ]
        time.sleep(delay)

        if debug:
            print("Text from Journal Image:")
            print(text_from_journal_image)
            print("---------------")

        # Split entries
        split_entries_str = self.gemini_manager.parse_entries(text_from_journal_image)

        entries = self.gemini_manager.split_entries_text(split_entries_str)
        time.sleep(delay)

        if debug:
            print("Entries from Journal Text:")
            for en in entries:
                print(en.strip())
            print("-----------------")

        # Get description of author
        self.author_description = self.gemini_manager.author_description(
            self.author_path, name=self.name
        )
        time.sleep(delay)

        entry_scene_prompts = [
            self.gemini_manager.summarize(entry, name=self.name) for entry in entries
        ]

        self.split_scenes = [
            scene_prompts.split("<SCENE>") for scene_prompts in entry_scene_prompts
        ]
        # cleaning
        self.split_scenes = [
            [scene.strip() for scene in entry] for entry in self.split_scenes
        ]

    @require_managers
    def generate_images(self, delay=0, fast=True, limit=12):
        counter = 0
        image_errors = []

        for entry in self.split_scenes:
            for scene in entry:

                if counter >= limit:
                    continue

                prompt = scene.strip()
                if len(prompt) > 0:
                    res, img_prompt = self.image_manager.generate_image(
                        prompt,
                        self.author_description,
                        theme=self.theme,
                        effect=self.effect,
                        fast=fast,
                    )

                if res != []:
                    self.images.append(generated_to_b64(res[0]))
                    self.editable_prompts.append(prompt)
                else:
                    self.images.append(None)
                    image_errors.append((counter, prompt))

                self.image_prompts.append(img_prompt)

                time.sleep(delay)

            counter += 1

        self.image_errors = image_errors

    @require_managers
    def regenerate_single_image(self, index, prompt, fast=True):
        if index < 0:
            index = 0

        if index >= len(self.images):
            index = len(self.images) - 1

        res, img_prompt = self.image_manager.generate_image(
            prompt,
            self.author_description,
            theme=self.theme,
            effect=self.effect,
            fast=fast,
        )

        if res != []:
            self.images[index] = generated_to_b64(res[0])
            self.editable_prompts[index] = prompt
            self.image_errors = [
                (idx, prompt) for idx, prompt in self.image_errors if idx != index
            ]

        else:

            for idx, prompt in self.image_errors:
                if idx == index:
                    self.image_errors[index] = (index, img_prompt)

        self.image_prompts[index] = [img_prompt]
