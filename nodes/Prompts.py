class Prompts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Positive": (
                    "STRING",
                    {"default": "Positive Prompt", "multiline": True},
                ),
                "Negative": (
                    "STRING",
                    {"default": "Negative Prompt", "multiline": True},
                ),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CLIP", "STRING", "STRING")
    RETURN_NAMES = (
        "Positive CONDITIONING",
        "Negative CONDITIONING",
        "CLIP",
        "Positive text",
        "Negative text",
    )
    FUNCTION = "prompts"
    CATEGORY = "Chibi-Nodes"

    def prompts(
        self,
        Positive,
        Negative,
        clip=None,
    ):
        if clip:
            pos_cond_raw = clip.tokenize(Positive)
            neg_cond_raw = clip.tokenize(Negative)
            pos_cond, pos_pooled = clip.encode_from_tokens(pos_cond_raw, return_pooled=True)
            neg_cond, neg_pooled = clip.encode_from_tokens(neg_cond_raw, return_pooled=True)

            return (
                [[pos_cond, {"pooled_output": pos_pooled}]],
                [[neg_cond, {"pooled_output": neg_pooled}]],
                clip,
                Positive,
                Negative,
            )
        else:
            return (None, None, None, Positive, Negative)
