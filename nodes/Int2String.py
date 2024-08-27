class Int2String:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"Int": ("INT", {"forceInput": True})},
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = False
    FUNCTION = "int2string"
    CATEGORY = "Chibi-Nodes/Text"

    def int2string(self, Int):
        return (str(Int),)
