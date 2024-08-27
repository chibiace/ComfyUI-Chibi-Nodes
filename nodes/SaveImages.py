import random
import numpy as np
import json
import os
from PIL import Image
import folder_paths
from comfy.cli_args import args
from PIL.PngImagePlugin import PngInfo
import time

# TODO:
# WARNING: SaveImages.IS_CHANGED() got an unexpected keyword argument 'filename_type'


class SaveImages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename_type": (["Timestamp", "Fixed", "Fixed Single"],),
                "fixed_filename": (
                    "STRING",
                    {
                        "default": "output",
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
                "vae": ("VAE",),
                "fixed_filename_override": (
                    "STRING",
                    {"forceInput": True},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "images",
        "filename_list",
    )
    FUNCTION = "saveimage"
    OUTPUT_NODE = True
    CATEGORY = "Chibi-Nodes"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        random.seed()
        return float("NaN")

    def saveimage(
        self,
        filename_type,
        fixed_filename,
        fixed_filename_override=None,
        vae=None,
        latents=None,
        images=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        if fixed_filename_override is not None:
            fixed_filename_override = fixed_filename_override.rsplit(".", 1)[0]
            fixed_filename = fixed_filename_override

        now = str(round(time.time()))

        results = list()
        filename_list = []
        counter = 0
        if images is not None:
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    now,
                    folder_paths.get_output_directory(),
                    images[0].shape[1],
                    images[0].shape[0],
                )
            )
            for image in images:
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                if filename_type == "Timestamp":
                    file = f"{now}_{counter:03}.png"
                if filename_type == "Fixed":
                    file = f"{fixed_filename}_{counter:03}.png"
                if filename_type == "Fixed Single":
                    file = f"{fixed_filename}.png"

                filename_list.append(file)
                img.save(
                    os.path.join(full_output_folder, file),
                    pnginfo=metadata,
                    compress_level=4,
                )
                results.append(
                    {"filename": file, "subfolder": subfolder, "type": "output"}
                )
                counter += 1
                return_results = images
        if vae is not None:
            if latents is not None:

                decoded_latents = vae.decode(latents["samples"])
                full_output_folder, filename, counter, subfolder, filename_prefix = (
                    folder_paths.get_save_image_path(
                        now,
                        folder_paths.get_output_directory(),
                        decoded_latents[0].shape[1],
                        decoded_latents[0].shape[0],
                    )
                )
                for latent in decoded_latents:
                    i = 255.0 * latent.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(
                                    x, json.dumps(extra_pnginfo[x]))

                    if filename_type == "Timestamp":
                        file = f"{now}_{counter:03}.png"
                    if filename_type == "Fixed":
                        file = f"{fixed_filename}_{counter:03}.png"
                    if filename_type == "Fixed Single":
                        file = f"{fixed_filename}.png"

                    filename_list.append(file)
                    img.save(
                        os.path.join(full_output_folder, file),
                        pnginfo=metadata,
                        compress_level=4,
                    )
                    results.append(
                        {"filename": file, "subfolder": subfolder, "type": "output"}
                    )
                    counter += 1
                return_results = decoded_latents

        # return { "ui":  { "images": results }}
        return {
            "ui": {"images": results},
            "result": (
                return_results,
                str(filename_list),
            ),
        }
