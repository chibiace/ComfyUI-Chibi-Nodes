class ConditionTextMulti:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
            },
            "optional": {
                "first": (
                    "STRING",
                    {"default": "", "multiline": False, "forceInput": True},
                ),
                "second": (
                    "STRING",
                    {"default": "", "multiline": False, "forceInput": True},
                ),
                "third": (
                    "STRING",
                    {"default": "", "multiline": False, "forceInput": True},
                ),
                "fourth": (
                    "STRING",
                    {"default": "", "multiline": False, "forceInput": True},
                ),
            },
        }

    RETURN_TYPES = (
        "CLIP",
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
    )
    RETURN_NAMES = (
        "CLIP",
        "first",
        "second",
        "third",
        "fourth",
    )
    FUNCTION = "conditiontext"
    CATEGORY = "Chibi-Nodes/Text"

    def conditiontext(
        self,
        clip,
        first="",
        second="",
        third="",
        fourth="",
    ):
        emptystring = ""
        returnedcond = []

        # TODO: I probably want to fix this mess at some point.
        if first != "":
            firstraw = clip.tokenize(first)
            first_cond, first_pooled = clip.encode_from_tokens(
                firstraw, return_pooled=True
            )
            returnedcond.append(
                [[first_cond, {"pooled_output": first_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(
                emptyraw, return_pooled=True
            )
            returnedcond.append(
                [[empty_cond, {"pooled_output": empty_pooled}]])

        if second != "":
            secondraw = clip.tokenize(second)
            second_cond, second_pooled = clip.encode_from_tokens(
                secondraw, return_pooled=True
            )
            returnedcond.append(
                [[second_cond, {"pooled_output": second_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(
                emptyraw, return_pooled=True
            )
            returnedcond.append(
                [[empty_cond, {"pooled_output": empty_pooled}]])

        if third != "":
            thirdraw = clip.tokenize(third)
            third_cond, third_pooled = clip.encode_from_tokens(
                thirdraw, return_pooled=True
            )
            returnedcond.append(
                [[third_cond, {"pooled_output": third_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(
                emptyraw, return_pooled=True
            )
            returnedcond.append(
                [[empty_cond, {"pooled_output": empty_pooled}]])

        if fourth != "":
            fourthraw = clip.tokenize(fourth)
            fourth_cond, fourth_pooled = clip.encode_from_tokens(
                fourthraw, return_pooled=True
            )
            returnedcond.append(
                [[fourth_cond, {"pooled_output": fourth_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(
                emptyraw, return_pooled=True
            )
            returnedcond.append(
                [[empty_cond, {"pooled_output": empty_pooled}]])

        return (
            clip,
            returnedcond[0],
            returnedcond[1],
            returnedcond[2],
            returnedcond[3],
        )
