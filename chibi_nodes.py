import torch
import folder_paths
import comfy.sd
import comfy.utils
from PIL import Image, ImageOps
import numpy as np

### GLOBALS ###
MAX_RESOLUTION=32768


class Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "Checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
            "Vae": (["Included"] + folder_paths.get_filename_list("vae"), ),
            "stop_at_clip_layer": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        }}

    RETURN_TYPES = ("MODEL","VAE","CLIP","LATENT",)
    FUNCTION = "loader"
    CATEGORY = "Chibi-Nodes"

    def loader(self, Checkpoint,Vae,stop_at_clip_layer,width,height,batch_size):
        ckpt_path = folder_paths.get_full_path("checkpoints", Checkpoint)
        output_vae = False
        if Vae == "Included":
            output_vae = True
        ckpt = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=output_vae, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        if Vae == "Included":
            vae = ckpt[:3][2]
        else:
            vae_path = folder_paths.get_full_path("vae", Vae)
            vae = comfy.sd.VAE(ckpt_path=vae_path)

        clip = ckpt[:3][1].clone()
        clip.clip_layer(stop_at_clip_layer)

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])

        return(ckpt[:3][0],vae,clip,{"samples":latent})



class Prompts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "Positive": ("STRING", {"default": "Positive Prompt","multiline": True}),
                "Negative": ("STRING", {"default": "Negative Prompt","multiline": True}),

               },
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("Positive Conditioning", "Negative Conditioning")
    FUNCTION = "prompts"
    CATEGORY = "Chibi-Nodes"

    def prompts(self, clip, Positive, Negative):
        pos_cond_raw = clip.tokenize(Positive)
        neg_cond_raw = clip.tokenize(Negative)
        pos_cond, pos_pooled = clip.encode_from_tokens(pos_cond_raw, return_pooled=True)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_cond_raw, return_pooled=True)
        
        return ([[pos_cond, {"pooled_output": pos_pooled}]],[[neg_cond, {"pooled_output": neg_pooled}]],)

class ImageTool:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 1920, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
            "height": ("INT", {"default": 1080, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
            "crop": ([False,True],),
            "rotate":  ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
            "mirror": ([False,True],),
            "flip":([False,True],),
            "bgcolor": (["black","white"],),
               },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "imagetools"
    CATEGORY = "Chibi-Nodes"

    def imagetools(self, image, height, width, crop, rotate, mirror, flip, bgcolor):
        image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(),0,255).astype(np.uint8))
        image = image.rotate(rotate,fillcolor=bgcolor)
        if mirror:
            image = ImageOps.mirror(image)
        
        if flip:
            image = ImageOps.flip(image)

        if crop:
            im_width, im_height = image.size   
            left = (im_width - width)/2
            top = (im_height - height)/2
            right = (im_width + width)/2
            bottom = (im_height + height)/2
            image = image.crop((left, top, right, bottom))            
            
        else:
            image = image.resize((width,height), Image.Resampling.LANCZOS)

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        return(image,)

NODE_CLASS_MAPPINGS = {
    "Loader":Loader,
    "Prompts": Prompts,
    "ImageTool": ImageTool,

    
}
