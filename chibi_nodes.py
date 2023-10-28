import torch
import folder_paths
import comfy.sd
import comfy.utils
from comfy.cli_args import args
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import random
import os
import time
import json 

### GLOBALS ###
MAX_RESOLUTION=32768
base_path = os.path.dirname(os.path.realpath(__file__))

models_dir = os.path.join(base_path, "extras")

folder_paths.folder_names_and_paths["chibi-wildcards"] = ([os.path.join(models_dir, "chibi-wildcards")], {".txt"})



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
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

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

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CLIP","TEXT","TEXT")
    RETURN_NAMES = ("Positive Conditioning", "Negative Conditioning", "CLIP", "Positive Text", "Negative Text")
    FUNCTION = "prompts"
    CATEGORY = "Chibi-Nodes"

    def prompts(self, clip, Positive, Negative):
        pos_cond_raw = clip.tokenize(Positive)
        neg_cond_raw = clip.tokenize(Negative)
        pos_cond, pos_pooled = clip.encode_from_tokens(pos_cond_raw, return_pooled=True)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_cond_raw, return_pooled=True)
        
        return ([[pos_cond, {"pooled_output": pos_pooled}]],[[neg_cond, {"pooled_output": neg_pooled}]],clip,Positive,Negative)

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
    
class Wildcards:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "clip": ("CLIP",),
                "textfile" : [sorted(folder_paths.get_filename_list("chibi-wildcards"))],
                "keyword":("STRING", {"default": "__wildcard__","multiline": False}),
            },
            "optional":{
                "text" : ("TEXT",),
            },
        }
    RETURN_TYPES = ("CONDITIONING","TEXT",)
    FUNCTION = "wildcards"
    CATEGORY = "Chibi-Nodes"
    seed = random.seed()
    def IS_CHANGED(s,seed):
        seed = random.seed()

    def wildcards(self, textfile,keyword,clip,text=None,):
        with open(folder_paths.get_full_path("chibi-wildcards", textfile)) as f:
            lines = f.readlines()
            aline = random.choice(lines)

        if text != None:
            raw = text.replace(keyword,aline)
            cond_raw = clip.tokenize(raw)
            cond, pooled = clip.encode_from_tokens(cond_raw, return_pooled=True)
            return([[cond, {"pooled_output": pooled}]],raw,)
        
        else:
            cond_raw = clip.tokenize(aline)
            cond, pooled = clip.encode_from_tokens(cond_raw, return_pooled=True)
            return([[cond, {"pooled_output": pooled}]],aline,)
        

       
class LoadEmbedding:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return{"required":{
            "text" : ("TEXT",),
            "embedding":[sorted(folder_paths.get_filename_list("embeddings"))],
                        "weight": ("FLOAT", {"default": 1.0, "min": -2, "max": 2, "step": 0.1, "round": 0.01}),
                         }}
    
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "loadembedding"
    CATEGORY = "Chibi-Nodes"

    def loadembedding(self, text, embedding,weight):

        output = text + ", (embedding:" + embedding + ":" + str(weight) + ")"
        # embedding.rsplit('.', maxsplit=1)[0]
        return(output,)
    
class ConditionText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
               },
               "optional":{
                "first" : ("TEXT",),
                "second" : ("TEXT",),
                "third" : ("TEXT",),
                "fourth" : ("TEXT",),
               }
        }

    RETURN_TYPES = ("CLIP","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("CLIP","first","second","third","fourth",)
    FUNCTION = "conditiontext"
    CATEGORY = "Chibi-Nodes"

    def conditiontext(self, clip, first=None, second=None, third=None, fourth=None, ):
        emptystring = ""
        returnedcond = []
        if first != None:
            firstraw = clip.tokenize(first)
            first_cond, first_pooled = clip.encode_from_tokens(firstraw, return_pooled=True)
            returnedcond.append([[first_cond, {"pooled_output": first_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(emptyraw, return_pooled=True)
            returnedcond.append([[empty_cond, {"pooled_output": empty_pooled}]])

        if second != None:
            secondraw = clip.tokenize(second)
            second_cond, second_pooled = clip.encode_from_tokens(secondraw, return_pooled=True) 
            returnedcond.append([[second_cond, {"pooled_output": second_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(emptyraw, return_pooled=True)
            returnedcond.append([[empty_cond, {"pooled_output": empty_pooled}]])

        if third != None:
            thirdraw = clip.tokenize(third)
            third_cond, third_pooled = clip.encode_from_tokens(thirdraw, return_pooled=True)
            returnedcond.append([[third_cond, {"pooled_output": third_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(emptyraw, return_pooled=True)
            returnedcond.append([[empty_cond, {"pooled_output": empty_pooled}]])

        if fourth != None:
            fourthraw = clip.tokenize(fourth)
            fourth_cond, fourth_pooled = clip.encode_from_tokens(fourthraw, return_pooled=True)
            returnedcond.append([[fourth_cond, {"pooled_output": fourth_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(emptyraw, return_pooled=True)
            returnedcond.append([[empty_cond, {"pooled_output": empty_pooled}]])

        
        return (clip,returnedcond[0],returnedcond[1],returnedcond[2],returnedcond[3],)
    
class SaveImages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               },
               "optional":{
                "images" : ("IMAGE",),
                "latents" : ("LATENT",),
                "vae" : ("VAE",),
               },
               "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ()
    FUNCTION = "saveimage"
    OUTPUT_NODE = True
    CATEGORY = "Chibi-Nodes"
    

    def saveimage(self,vae=None, latents=None, images=None, prompt=None, extra_pnginfo=None):

        now = str(round(time.time()))
        


        results = list()
        counter = 0
        if images != None:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(now, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0])
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                file = f"{now}_{counter:03}_.png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": "output"
                })
                counter += 1
        if vae != None:
            if latents != None:
                
                decoded_latents = vae.decode(latents["samples"])
                full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(now, folder_paths.get_output_directory(), decoded_latents[0].shape[1], decoded_latents[0].shape[0])
                for latent in decoded_latents:
                    i = 255. * latent.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    file = f"{now}_{counter:03}_.png"
                    img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
                    results.append({
                        "filename": file,
                        "subfolder": subfolder,
                        "type": "output"
                    })
                    counter += 1

        return { "ui": { "images": results } }


NODE_CLASS_MAPPINGS = {
    "Loader":Loader,
    "Prompts": Prompts,
    "ImageTool": ImageTool,
    "Wildcards": Wildcards,
    "LoadEmbedding": LoadEmbedding,
    "ConditionText": ConditionText, 
    "SaveImages":SaveImages,

}
