class Textbox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": False,
                        "print_to_screen": True,
                    },
                ),
            },
            "optional": {
                "passthrough": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                )
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "textbox"
    CATEGORY = "Chibi-Nodes/Text"

    def textbox(self, text="", passthrough=""):
        if passthrough != "":
            text = passthrough
            return {"ui": {"text": text}, "result": (text,)}
        else:
            return (text,)
