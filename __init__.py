from .nodes.Loader import Loader
from .nodes.SimpleSampler import SimpleSampler
from .nodes.Prompts import Prompts
from .nodes.ImageTool import ImageTool
from .nodes.Wildcards import Wildcards
from .nodes.LoadEmbedding import LoadEmbedding
from .nodes.ConditionText import ConditionText
from .nodes.ConditionTextMulti import ConditionTextMulti
from .nodes.ConditionTextPrompts import ConditionTextPrompts
from .nodes.Textbox import Textbox
from .nodes.ImageSizeInfo import ImageSizeInfo
from .nodes.ImageSimpleResize import ImageSimpleResize
from .nodes.ImageAddText import ImageAddText
from .nodes.Int2String import Int2String
from .nodes.LoadImageExtended import LoadImageExtended
from .nodes.SeedGenerator import SeedGenerator
from .nodes.SaveImages import SaveImages
from .nodes.TextSplit import TextSplit
from .nodes.RandomResolutionLatent import RandomResolutionLatent


WEB_DIRECTORY = "js"
__all__ = ["NODE_CLASS_MAPPINGS"]


NODE_CLASS_MAPPINGS = {
    "Loader": Loader,
    "SimpleSampler": SimpleSampler,
    "Prompts": Prompts,
    "ImageTool": ImageTool,
    "Wildcards": Wildcards,
    "LoadEmbedding": LoadEmbedding,
    "ConditionText": ConditionText,
    "ConditionTextPrompts": ConditionTextPrompts,
    "ConditionTextMulti": ConditionTextMulti,
    "Textbox": Textbox,
    "ImageSizeInfo": ImageSizeInfo,
    "ImageSimpleResize": ImageSimpleResize,
    "ImageAddText": ImageAddText,
    "Int2String": Int2String,
    "LoadImageExtended": LoadImageExtended,
    "SeedGenerator": SeedGenerator,
    "SaveImages": SaveImages,
    "TextSplit": TextSplit,
    "RandomResolutionLatent": RandomResolutionLatent,
}
