import folder_paths
import os
import json
import re
from PIL import Image, ImageOps
import numpy as np
import torch
import hashlib


class LoadImageExtended:
    def __init__(self):

        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {"vae": ("VAE",)},
        }

    CATEGORY = "Chibi-Nodes/Image"

    # changes here
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "LATENT",
        "STRING",
        "STRING",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "IMAGE",
        "MASK",
        "LATENT",
        "filename",
        "image Info",
        "width",
        "height",
    )
    #

    FUNCTION = "load_image"

    def load_image(self, image, vae=None):

        image_path = folder_paths.get_annotated_filepath(image)
        filename = image_path.rsplit("/", 1)[-1]

        im = Image.open(image_path)

        # Start ai-info.py section, with no exif
        def type_changer(value):
            if value.isnumeric():
                return int(value)
            else:
                return value

        im.load()

        prompt = {}
        if "prompt" in im.info.keys():
            # comfyui, workflow is also available but we aren't getting that today
            # prompt = {}
            prompt.update({"prompt": json.loads(im.info["prompt"])})
        else:
            # automatic111, gosh this is a mess.
            if "parameters" in im.info.keys():
                parameters = im.info["parameters"]
                prompt = {"parameters": {}}
                parameters = re.split(
                    "(Negative prompt): |(Negative Template): |(Template): |(ControlNet): |\n",
                    parameters,
                )

                # removes None and new lines
                parameters_clean_none = []
                for i in range(0, len(parameters)):
                    if parameters[i] is None:
                        pass
                    elif parameters[i] == "":
                        pass
                    else:
                        parameters_clean_none.append(parameters[i])
                parameters = parameters_clean_none

                # settings field
                parameters_settings = {}
                for i in range(0, len(parameters)):
                    if parameters[i].split(":", 1)[0] == "Steps":
                        parameters[i] = re.split(", ", parameters[i])
                        for k in parameters[i]:
                            k = k.split(": ", 1)
                            if len(k) == 2:
                                k[1] = type_changer(k[1])

                                # makes "Size" : "(widthxheight)" into two keys
                                if k[0] == "Size":
                                    k[1] = k[1].split("x")
                                    for s in range(0, len(k[1])):
                                        k[1][s] = type_changer(k[1][s])
                                    parameters_settings.update(
                                        {"width": k[1][0]})
                                    parameters_settings.update(
                                        {"height": k[1][1]})

                                else:
                                    parameters_settings.update({k[0]: k[1]})

                        parameters[i] = parameters_settings

                # builder
                parameters_built = {}
                for i in range(0, len(parameters)):
                    match parameters[i]:
                        case "Negative prompt":
                            parameters_built.update(
                                {parameters[i]: parameters[i + 1]})
                        case "Negative Template":
                            parameters_built.update(
                                {parameters[i]: parameters[i + 1]})
                        case "Template":
                            parameters_built.update(
                                {parameters[i]: parameters[i + 1]})
                        case "ControlNet":
                            parameters_built.update(
                                {parameters[i]: parameters[i + 1]})
                        case dict():
                            parameters_built.update(parameters[i])
                        case _:
                            if i == 0:
                                parameters_built.update(
                                    {"Positive prompt": parameters[i]}
                                )
                            pass

                prompt["parameters"] = parameters_built
        if type(prompt) is dict:
            prompt = json.dumps(prompt, indent=2)

        elif type(prompt) is str:
            prompt = json.dumps(json.loads(prompt), indent=2)

        # end section

        im = ImageOps.exif_transpose(im)
        image = im.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        shape = image.shape
        width = shape[2]
        height = shape[1]
        if "A" in im.getbands():
            mask = np.array(im.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        if vae is not None:

            latent = image
            x = (latent.shape[1] // 8) * 8
            y = (latent.shape[2] // 8) * 8
            if latent.shape[1] is not x or latent.shape[2] is not y:
                x_offset = (latent.shape[1] % 8) // 2
                y_offset = (latent.shape[2] % 8) // 2
                latent = latent[:, x_offset: x +
                                x_offset, y_offset: y + y_offset, :]
            latent = vae.encode(latent[:, :, :, :3])

            return (
                image,
                mask.unsqueeze(0),
                {"samples": latent},
                filename,
                str(prompt),
                width,
                height,
            )
        else:
            return (
                image,
                mask.unsqueeze(0),
                None,
                filename,
                str(prompt),
                width,
                height,
            )

    @classmethod
    def IS_CHANGED(s, image, vae=None):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, vae=None):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
