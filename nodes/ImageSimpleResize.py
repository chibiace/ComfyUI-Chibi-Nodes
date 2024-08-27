from PIL import Image, ImageOps
import numpy as np
import torch
MAX_RESOLUTION = 32768


class ImageSimpleResize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 1},
                ),
                "edge": (["largest", "smallest", "all", "width", "height"],),
            },
            "optional": {
                "size_override": ("INT", {"forceInput": True}),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    OUTPUT_NODE = False
    FUNCTION = "imagesimpleresize"
    CATEGORY = "Chibi-Nodes/Image"

    def imagesimpleresize(self, image, size, edge, size_override=None, vae=None):
        if size_override:
            size = size_override

        width = image.shape[2]
        height = image.shape[1]
        ratio = height / width
        image = Image.fromarray(
            np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
        )

        if edge == "largest":
            if width > height:
                if size < width:
                    image = ImageOps.contain(
                        image, (size, MAX_RESOLUTION), Image.LANCZOS
                    )
                else:
                    image = image.resize(
                        (round(size), round(size * ratio)), Image.LANCZOS
                    )
            if width < height:
                if size < height:
                    image = ImageOps.contain(
                        image, (MAX_RESOLUTION, size), Image.LANCZOS
                    )
                else:
                    image = image.resize(
                        (round(size / ratio), round(size)), Image.LANCZOS
                    )
            if width == height:
                if size < width:
                    image = ImageOps.contain(
                        image, (size, size), Image.LANCZOS)
                else:
                    image = image.resize(
                        (round(size), round(size)), Image.LANCZOS)

        if edge == "smallest":
            if width > height:
                if size < height:
                    image = ImageOps.contain(
                        image, (MAX_RESOLUTION, size), Image.LANCZOS
                    )
                else:
                    image = image.resize(
                        (round(size / ratio), round(size)), Image.LANCZOS
                    )
            if width < height:
                if size < width:
                    image = ImageOps.contain(
                        image, (size, MAX_RESOLUTION), Image.LANCZOS
                    )
                else:
                    image = image.resize(
                        (round(size), round(size * ratio)), Image.LANCZOS
                    )
            if width == height:
                if size < width:
                    image = ImageOps.contain(
                        image, (size, size), Image.LANCZOS)
                else:
                    image = image.resize(
                        (round(size), round(size)), Image.LANCZOS)

        if edge == "all":
            image = image.resize((round(size), round(size)), Image.LANCZOS)

        if edge == "width":
            image = image.resize((round(size), round(height)), Image.LANCZOS)

        if edge == "height":
            image = image.resize((round(width), round(size)), Image.LANCZOS)

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

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

            return (image, {"samples": latent})
        else:
            return (
                image,
                None,
            )
