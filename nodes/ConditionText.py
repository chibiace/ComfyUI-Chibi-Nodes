class ConditionText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": (
                    "STRING",
                    {"forceInput": True},
                ),
            }
        }

    RETURN_TYPES = (
        "CLIP",
        "CONDITIONING",
    )
    FUNCTION = "conditiontext"
    CATEGORY = "Chibi-Nodes/Text"

    def conditiontext(self, clip, text=None):

        if text is not None:
            tokens = clip.tokenize(text)
        else:
            tokens = clip.tokenize("")
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        return (
            clip,
            [[cond, {"pooled_output": pooled}]],
        )
