class ImageSizeInfo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",)},
            "hidden": {
                "width": ("INT",),
                "height": ("INT",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "IMAGE",
        "width",
        "height",
    )
    OUTPUT_NODE = True
    FUNCTION = "imagesizeinfo"
    CATEGORY = "Chibi-Nodes/Image"

    def imagesizeinfo(self, image, width=0, height=0):
        shape = image.shape
        width = shape[2]
        height = shape[1]
        return {
            "ui": {"width": [width], "height": [height]},
            "result": (
                image,
                width,
                height,
            ),
        }
