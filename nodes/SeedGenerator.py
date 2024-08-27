import random
import math


class SeedGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["Random", "Fixed"],),
                "fixed_seed": (
                    "INT",
                    {
                        "default": 8008135,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "INT",
        "STRING",
    )
    RETURN_NAMES = ("seed", "text")
    OUTPUT_NODE = False
    FUNCTION = "generator"
    CATEGORY = "Chibi-Nodes/Numbers"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        random.seed()
        return float("NaN")

    def generator(self, mode, fixed_seed):
        if mode == "Random":
            random_seed = math.floor(random.random() * 10000000000000000)
            return (
                random_seed,
                str(random_seed),
            )
        if mode == "Fixed":
            return (
                fixed_seed,
                str(fixed_seed),
            )
