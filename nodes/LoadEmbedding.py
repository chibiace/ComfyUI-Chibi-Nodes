import folder_paths


class LoadEmbedding:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "", "multiline": False, "forceInput": True},
                ),
                "embedding": [
                    sorted(
                        folder_paths.get_filename_list("embeddings"),
                    )
                ],
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -2, "max": 2,
                        "step": 0.1, "round": 0.01},
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = ("text",)
    FUNCTION = "load_embedding"
    CATEGORY = "Chibi-Nodes"

    def load_embedding(self, text, embedding, weight):
        output = text + ", (embedding:" + embedding + ":" + str(weight) + ")"
        return (output,)
