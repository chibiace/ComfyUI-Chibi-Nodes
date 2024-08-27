import random
import torch


class RandomResolutionLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
        }

    RETURN_TYPES = (
        "LATENT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "LATENT",
        "width",
        "height",
    )
    OUTPUT_NODE = True
    FUNCTION = "random_resolution"
    CATEGORY = "Chibi-Nodes/Numbers"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        random.seed()
        return float("NaN")

    def random_resolution(self, batch_size):
        resolutions = [512, 768, 1024]

        res_list = []

        for x in resolutions:
            for y in resolutions:
                a = (x, y)
                b = (y, x)
                if a not in res_list:
                    res_list.append(a)
                if b not in res_list:
                    res_list.append(b)

        rand_res = random.choice(res_list)
        latent = torch.zeros(
            [batch_size, 4, rand_res[0] // 8, rand_res[1] // 8])

        return (
            {"samples": latent},
            rand_res[0],
            rand_res[1],
        )
