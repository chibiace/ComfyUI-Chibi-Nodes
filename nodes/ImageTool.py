from PIL import Image, ImageOps
import torch
import numpy as np
MAX_RESOLUTION = 32768


class ImageTool:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": 1920, "min": 16, "max": MAX_RESOLUTION, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 1080, "min": 16, "max": MAX_RESOLUTION, "step": 1},
                ),
                "crop": ([False, True],),
                "rotate": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
                "mirror": ([False, True],),
                "flip": ([False, True],),
                "bgcolor": (["black", "white"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "imagetools"
    CATEGORY = "Chibi-Nodes/Image"

    def imagetools(self, image, height, width, crop, rotate, mirror, flip, bgcolor):
        image = Image.fromarray(
            np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
        )
        image = image.rotate(rotate, fillcolor=bgcolor)

        # black and white
        # corrections?
        # generate mask from background color (crop, rotate)

        if mirror:
            image = ImageOps.mirror(image)

        if flip:
            image = ImageOps.flip(image)

        if crop:
            im_width, im_height = image.size
            left = (im_width - width) / 2
            top = (im_height - height) / 2
            right = (im_width + width) / 2
            bottom = (im_height + height) / 2
            image = image.crop((left, top, right, bottom))

        else:
            image = image.resize((width, height), Image.LANCZOS)

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)
