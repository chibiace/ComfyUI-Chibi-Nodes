class TextSplit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "", "forceInput": True},
                ),
                "separator": (
                    "STRING",
                    {"default": "."},
                ),
                "reverse": ([False, True],),
                "return_half": (["First Half", "Second Half"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "do_split"
    CATEGORY = "Chibi-Nodes/Text"

    def do_split(self, text, separator, reverse, return_half):
        if reverse:
            text = text.rsplit(separator, 1)
        else:
            text = text.split(separator, 1)

        if len(text) == 2:
            if return_half == "First Half":
                text = text[0]
            if return_half == "Second Half":
                text = text[1]

        return (text,)
