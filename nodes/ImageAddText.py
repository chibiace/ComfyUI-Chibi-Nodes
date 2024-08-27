from PIL import Image, ImageOps, ImageFont, ImageDraw
import folder_paths
import numpy as np
import torch
import os

MAX_RESOLUTION = 32768

node_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.dirname(node_path)
extras_dir = os.path.join(base_path, "extras")
folder_paths.folder_names_and_paths["chibi-fonts"] = (
    [os.path.join(extras_dir, "fonts")],
    {".ttf"},
)


class ImageAddText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "Chibi-Nodes", "multiline": True},
                ),
                "font": [sorted(folder_paths.get_filename_list("chibi-fonts"))],
                "font_size": ("INT", {"default": 24, "min": 0, "max": 200, "step": 1}),
                "font_colour": (["black", "white", "red", "green", "blue"],),
                "invert_mask": ([False, True],),
                "position_x": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1},
                ),
                "position_y": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "STRING",
    )
    RETURN_NAMES = (
        "IMAGE",
        "MASK",
        "text",
    )
    FUNCTION = "addtext"
    CATEGORY = "Chibi-Nodes/Image"

    def addtext(
        self,
        text,
        width,
        height,
        font,
        font_size,
        position_x,
        position_y,
        font_colour,
        invert_mask,
        image=None,
    ):
        if image is not None:
            width = image.shape[2]
            height = image.shape[1]
            image = Image.fromarray(
                np.clip(255.0 * image[0].cpu().numpy(),
                        0, 255).astype(np.uint8)
            )
            image = image.convert("RGBA")
        else:
            image = Image.new("RGBA", (width, height), (255, 255, 255, 0))

        text_image = Image.new("RGBA", (width, height), (0, 255, 255, 0))
        imaget = ImageDraw.Draw(
            text_image,
        )

        msg = text
        imaget.fontmode = "L"
        fnt = ImageFont.truetype(
            folder_paths.get_full_path("chibi-fonts", font), font_size
        )

        imaget.text((position_x, position_y), msg, font=fnt, fill=font_colour)

        if "A" in text_image.getbands():
            mask = np.array(text_image.getchannel(
                "A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        image.paste(text_image, (0, 0), text_image)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        if invert_mask:
            mask = 1.0 - mask

        return (
            image,
            mask.unsqueeze(0),
            text,
        )
