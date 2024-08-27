class ConditionTextPrompts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive": (
                    "STRING",
                    {"default": "", "multiline": False, "forceInput": True},
                ),
                "negative": (
                    "STRING",
                    {"default": "", "multiline": False, "forceInput": True},
                ),
            },
        }

    RETURN_TYPES = (
        "CLIP",
        "CONDITIONING",
        "CONDITIONING",
    )
    RETURN_NAMES = (
        "CLIP",
        "positive",
        "negative",
    )
    FUNCTION = "conditiontext"
    CATEGORY = "Chibi-Nodes/Text"

    def conditiontext(self, clip, positive, negative,):
        returnedcond = []

        positive_raw = clip.tokenize(positive)
        positive_cond, positive_pooled = clip.encode_from_tokens(
            positive_raw, return_pooled=True
        )
        returnedcond.append(
            [[positive_cond, {"pooled_output": positive_pooled}]])

        negative_raw = clip.tokenize(negative)
        negative_cond, negative_pooled = clip.encode_from_tokens(
            negative_raw, return_pooled=True
        )
        returnedcond.append(
            [[negative_cond, {"pooled_output": negative_pooled}]])

        return (
            clip,
            returnedcond[0],
            returnedcond[1],
        )
