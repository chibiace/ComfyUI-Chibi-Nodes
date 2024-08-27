import folder_paths
import random
import os

node_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.dirname(node_path)
extras_dir = os.path.join(base_path, "extras")
folder_paths.folder_names_and_paths["chibi-wildcards"] = (
    [os.path.join(extras_dir, "chibi-wildcards")],
    {".txt"},
)


class Wildcards:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "textfile": [sorted(folder_paths.get_filename_list("chibi-wildcards"))],
                "keyword": ("STRING", {"default": "__wildcard__", "multiline": False}),
                "entries_returned": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
            },
            "optional": {
                "clip": ("CLIP",),
                "seed": ("INT", {"forceInput": True}),
                "text": (
                    "STRING",
                    {"default": "", "multiline": False, "forceInput": True},
                ),

            },
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    RETURN_NAMES = (
        "CONDITIONING",
        "text",
    )
    FUNCTION = "wildcards"
    CATEGORY = "Chibi-Nodes"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        random.seed()
        return float("NaN")

    def wildcards(
        self,
        textfile,
        keyword,
        entries_returned,
        clip=None,
        seed=None,
        text="",
    ):
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        entries = ""
        with open(folder_paths.get_full_path("chibi-wildcards", textfile)) as f:
            lines = f.readlines()
            for i in range(0, entries_returned):
                aline = random.choice(lines).rstrip()
                if entries == "":
                    entries = aline
                else:
                    entries = entries + " " + aline

        aline = entries
        if text != "":
            raw = text.replace(keyword, aline)

            if clip:
                cond_raw = clip.tokenize(raw)
                cond, pooled = clip.encode_from_tokens(
                    cond_raw, return_pooled=True)
                return (
                    [[cond, {"pooled_output": pooled}]],
                    raw,
                )
            else:
                return (
                    None,
                    raw,
                )

        else:
            if clip:
                cond_raw = clip.tokenize(aline)
                cond, pooled = clip.encode_from_tokens(
                    cond_raw, return_pooled=True)
                return (
                    [[cond, {"pooled_output": pooled}]],
                    aline,
                )
            else:
                return (
                    None,
                    aline,
                )
