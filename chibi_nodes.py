import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.samplers
from comfy.cli_args import args
from PIL import Image, ImageOps, ImageFont, ImageDraw, ExifTags
from PIL.PngImagePlugin import PngInfo
import numpy as np
import re
import random
import os
import time
import json 
import math
import hashlib
import latent_preview
### GLOBALS ###
MAX_RESOLUTION=32768
base_path = os.path.dirname(os.path.realpath(__file__))

extras_dir = os.path.join(base_path, "extras")

folder_paths.folder_names_and_paths["chibi-wildcards"] = ([os.path.join(extras_dir, "chibi-wildcards")], {".txt"})
folder_paths.folder_names_and_paths["fonts"] = ([os.path.join(extras_dir, "fonts")], {".ttf"})



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
                
                "Positive": ("STRING", {"default": "Positive Prompt","multiline": True}),
                "Negative": ("STRING", {"default": "Negative Prompt","multiline": True}),

               },
               "optional":{
                   "clip": ("CLIP",),
               },

        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CLIP","STRING","STRING")
    RETURN_NAMES = ("Positive CONDITIONING", "Negative CONDITIONING", "CLIP", "Positive text", "Negative text")
    FUNCTION = "prompts"
    CATEGORY = "Chibi-Nodes"

    def prompts(self,  Positive, Negative, clip=None,):
        if clip:
            pos_cond_raw = clip.tokenize(Positive)
            neg_cond_raw = clip.tokenize(Negative)
            pos_cond, pos_pooled = clip.encode_from_tokens(pos_cond_raw, return_pooled=True)
            neg_cond, neg_pooled = clip.encode_from_tokens(neg_cond_raw, return_pooled=True)
            
            return ([[pos_cond, {"pooled_output": pos_pooled}]],[[neg_cond, {"pooled_output": neg_pooled}]],clip,Positive,Negative)
        else:
            return (None, None, None, Positive,Negative)

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
    CATEGORY = "Chibi-Nodes/Image"

    def imagetools(self, image, height, width, crop, rotate, mirror, flip, bgcolor):
        image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(),0,255).astype(np.uint8))
        image = image.rotate(rotate,fillcolor=bgcolor)

        #black and white
        #corrections?
        #generate mask from background color (crop, rotate)

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
            image = image.resize((width,height), Image.LANCZOS)

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
                "textfile" : [sorted(folder_paths.get_filename_list("chibi-wildcards"))],
                "keyword":("STRING", {"default": "__wildcard__","multiline": False}),
                "entries_returned":  ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional":{
                "clip": ("CLIP",),
                "text" : ("STRING",{"default": '', "multiline": False, "forceInput": True}),
            },
        }
    RETURN_TYPES = ("CONDITIONING","STRING",)
    RETURN_NAMES = ("CONDITIONING","text",)
    FUNCTION = "wildcards"
    CATEGORY = "Chibi-Nodes"

    seed = random.seed()
    

    def IS_CHANGED(s,seed):
        seed = random.seed()

    def wildcards(self, textfile,keyword,entries_returned,clip=None,text='',):
        entries = ""
        with open(folder_paths.get_full_path("chibi-wildcards", textfile)) as f:
            lines = f.readlines()
            for i in range(0,entries_returned):
                aline = random.choice(lines).rstrip()
                if entries == "":
                    entries = aline
                else:
                    entries = entries + " " + aline 
            
        aline = entries
        if text != '':
            raw = text.replace(keyword,aline)

            if clip:
                cond_raw = clip.tokenize(raw)
                cond, pooled = clip.encode_from_tokens(cond_raw, return_pooled=True)
                return([[cond, {"pooled_output": pooled}]],raw,)
            else:
                return(None,raw,)
        
        else:
            if clip:
                cond_raw = clip.tokenize(aline)
                cond, pooled = clip.encode_from_tokens(cond_raw, return_pooled=True)
                return([[cond, {"pooled_output": pooled}]],aline,)
            else:
                return(None,aline,)

       
class LoadEmbedding:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return{"required":{
            "text" : ("STRING",{"default": '', "multiline": False, "forceInput": True}),
            "embedding":[sorted(folder_paths.get_filename_list("embeddings"),)],
                        "weight": ("FLOAT", {"default": 1.0, "min": -2, "max": 2, "step": 0.1, "round": 0.01}),
                        
                         },
                         "hidden":{
                             "preview_image": ("IMAGE",)
                         }}
    
    RETURN_TYPES = ("STRING","IMAGE",)
    RETURN_NAMES = ("text","Preview Image")
    FUNCTION = "loadembedding"
    CATEGORY = "Chibi-Nodes"


    def loadembedding(self, text, embedding,weight, preview_image=None):

        output = text + ", (embedding:" + embedding + ":" + str(weight) + ")"

        if os.path.exists(folder_paths.get_full_path("embeddings", embedding).replace(".pt",".preview.png")):
            img_path = folder_paths.get_full_path("embeddings", embedding).replace(".pt",".preview.png") 
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            preview_image = torch.from_numpy(image)[None,]
        
            return (output,preview_image)
        
        else:
            W, H = (256,256)
            image = Image.new('RGB', (W, H), (255, 255, 255))
            imaget = ImageDraw.Draw(image)
            msg = "No Preview"
            imaget.text(((W-60)/2,H/2),msg,(0,0,0))
        

            image = np.array(image).astype(np.float32) / 255.0
            preview_image = torch.from_numpy(image)[None,]
        
            return (output,preview_image)
        


class ConditionTextMulti:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
               },
               "optional":{
                "first" : ("STRING", {"default": '', "multiline": False, "forceInput": True}),
                "second" : ("STRING", {"default": '', "multiline": False, "forceInput": True}),
                "third" : ("STRING", {"default": '', "multiline": False, "forceInput": True}),
                "fourth" : ("STRING", {"default": '', "multiline": False, "forceInput": True}),
               }
        }

    RETURN_TYPES = ("CLIP","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("CLIP","first","second","third","fourth",)
    FUNCTION = "conditiontext"
    CATEGORY = "Chibi-Nodes/Text"

    def conditiontext(self, clip, first='', second='', third='', fourth='', ):
        emptystring = ""
        returnedcond = []

        #!I probably want to fix this mess at some point.
        if first != '':
            firstraw = clip.tokenize(first)
            first_cond, first_pooled = clip.encode_from_tokens(firstraw, return_pooled=True)
            returnedcond.append([[first_cond, {"pooled_output": first_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(emptyraw, return_pooled=True)
            returnedcond.append([[empty_cond, {"pooled_output": empty_pooled}]])

        if second != '':
            secondraw = clip.tokenize(second)
            second_cond, second_pooled = clip.encode_from_tokens(secondraw, return_pooled=True) 
            returnedcond.append([[second_cond, {"pooled_output": second_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(emptyraw, return_pooled=True)
            returnedcond.append([[empty_cond, {"pooled_output": empty_pooled}]])

        if third != '':
            thirdraw = clip.tokenize(third)
            third_cond, third_pooled = clip.encode_from_tokens(thirdraw, return_pooled=True)
            returnedcond.append([[third_cond, {"pooled_output": third_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(emptyraw, return_pooled=True)
            returnedcond.append([[empty_cond, {"pooled_output": empty_pooled}]])

        if fourth != '':
            fourthraw = clip.tokenize(fourth)
            fourth_cond, fourth_pooled = clip.encode_from_tokens(fourthraw, return_pooled=True)
            returnedcond.append([[fourth_cond, {"pooled_output": fourth_pooled}]])
        else:
            emptyraw = clip.tokenize(emptystring)
            empty_cond, empty_pooled = clip.encode_from_tokens(emptyraw, return_pooled=True)
            returnedcond.append([[empty_cond, {"pooled_output": empty_pooled}]])

        
        return (clip,returnedcond[0],returnedcond[1],returnedcond[2],returnedcond[3],)
    
class ConditionText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text" : ("STRING", {"forceInput": True},),
                }}

    RETURN_TYPES = ("CLIP","CONDITIONING",)
    FUNCTION = "conditiontext"
    CATEGORY = "Chibi-Nodes/Text"

    def conditiontext(self, clip, text=None ):

        if text != None:
            tokens = clip.tokenize(text)
        else:
            tokens = clip.tokenize("")
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)


        return (clip,[[cond, {"pooled_output": pooled}]],)

class SaveImages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename_type":(["Timestamp","Fixed","Fixed Single"],),
                "fixed_filename":("STRING",{"default":"output",})
            
               },
               "optional":{
                "images" : ("IMAGE",),
                "latents" : ("LATENT",),
                "vae" : ("VAE",),
                "fixed_filename_override":("STRING",{"forceInput": True},)
               },
               "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("images","filename_list",)
    FUNCTION = "saveimage"
    OUTPUT_NODE = True
    CATEGORY = "Chibi-Nodes"
    
    seed = random.seed()
    

    def IS_CHANGED(s,seed):
        seed = random.seed()

    def saveimage(self,filename_type,fixed_filename,fixed_filename_override=None,vae=None, latents=None, images=None, prompt=None, extra_pnginfo=None):
        if fixed_filename_override != None:
            fixed_filename_override = fixed_filename_override.rsplit(".",1)[0]
            fixed_filename = fixed_filename_override

        now = str(round(time.time()))
        


        results = list()
        filename_list = []
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
                
                if filename_type == "Timestamp":
                    file = f"{now}_{counter:03}.png"
                if filename_type == "Fixed":
                    file = f"{fixed_filename}_{counter:03}.png"
                if filename_type == "Fixed Single":
                    file = f"{fixed_filename}.png"

                filename_list.append(file)
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": "output"
                })
                counter += 1
                return_results = images
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

                    if filename_type == "Timestamp":
                        file = f"{now}_{counter:03}.png"
                    if filename_type == "Fixed":
                        file = f"{fixed_filename}_{counter:03}.png"
                    if filename_type == "Fixed Single":
                        file = f"{fixed_filename}.png"

                    filename_list.append(file)
                    img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
                    results.append({
                        "filename": file,
                        "subfolder": subfolder,
                        "type": "output"
                    })
                    counter += 1
                return_results = decoded_latents

        # return { "ui":  { "images": results }}
        return {"ui": { "images": results },"result": (return_results,str(filename_list),)}
    


class Textbox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "text":("STRING", {"default": '',"multiline": True,"forceInput": False,"print_to_screen": True}),

            },
               "optional": {
                   "passthrough":("STRING", {"default": "","multiline": True,"forceInput": True})
               },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "textbox"
    CATEGORY = "Chibi-Nodes/Text"

    def textbox(self,text="",passthrough=""):
        if passthrough != "":
            text = passthrough
            return {"ui": {"text": text},"result": (text,)}
        else:
            return (text,)

class ImageSizeInfo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image": ("IMAGE",)
            },
            "hidden":{
                "width": ("INT",),
                "height": ("INT",),
            }}
    RETURN_TYPES = ("IMAGE","INT","INT",)
    RETURN_NAMES = ("IMAGE","width","height",)
    OUTPUT_NODE = True
    FUNCTION = "imagesizeinfo"
    CATEGORY = "Chibi-Nodes/Image"

    def imagesizeinfo(self, image, width=0, height=0):
        shape = image.shape
        width = shape[2]
        height = shape[1]
        return {"ui": {"width": [width], "height": [height]},"result": (image,width,height,)}


class ImageSimpleResize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image": ("IMAGE",),
                "size": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
                "edge":(["largest","smallest","all","width","height"],),
            },
            "optional":{
                "size_override":("INT",{"forceInput": True}),
                "vae":("VAE",)
            }}
    RETURN_TYPES = ("IMAGE","LATENT")
    OUTPUT_NODE = False
    FUNCTION = "imagesimpleresize"
    CATEGORY = "Chibi-Nodes/Image"

    def imagesimpleresize(self, image, size, edge, size_override=None, vae=None):
        if size_override:
            size = size_override

        width = image.shape[2]
        height = image.shape[1]
        ratio = height / width
        image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(),0,255).astype(np.uint8))

        if edge == "largest":
            if width > height:
                if size < width:
                    image = ImageOps.contain(image, (size,MAX_RESOLUTION), Image.LANCZOS)
                else:
                    image = image.resize((round(size),round(size*ratio)), Image.LANCZOS)
            if width < height:
                if size < height:
                    image = ImageOps.contain(image, (MAX_RESOLUTION,size), Image.LANCZOS)
                else:
                    image = image.resize((round(size/ratio),round(size)), Image.LANCZOS)
            if width == height:
                if size < width:
                    image = ImageOps.contain(image, (size,size), Image.LANCZOS)
                else:
                    image = image.resize((round(size),round(size)), Image.LANCZOS)


        if edge == "smallest":
            if width > height:
                if size < height:
                    image = ImageOps.contain(image, (MAX_RESOLUTION,size), Image.LANCZOS)
                else:
                    image = image.resize((round(size/ratio),round(size)), Image.LANCZOS)
            if width < height:
                if size < width:
                    image = ImageOps.contain(image, (size,MAX_RESOLUTION), Image.LANCZOS)
                else:
                    image = image.resize((round(size),round(size*ratio)), Image.LANCZOS)
            if width == height:
                if size < width:
                    image = ImageOps.contain(image, (size,size), Image.LANCZOS)
                else:
                    image = image.resize((round(size),round(size)), Image.LANCZOS)

        if edge == "all":
            image = image.resize((round(size),round(size)), Image.LANCZOS)

        if edge == "width":
            image = image.resize((round(size),round(height)), Image.LANCZOS)

        if edge == "height":
            image = image.resize((round(width),round(size)), Image.LANCZOS)


        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        if vae != None:

            latent = image
            x = (latent.shape[1] // 8) * 8
            y = (latent.shape[2] // 8) * 8
            if latent.shape[1] != x or latent.shape[2] != y:
                x_offset = (latent.shape[1] % 8) // 2
                y_offset = (latent.shape[2] % 8) // 2
                latent = latent[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            latent = vae.encode(latent[:,:,:,:3])

            return (image,{"samples":latent})
        else:
            return(image,None,)
    
class Int2String:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "Int":("INT",{"forceInput": True})
            },}
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = False
    FUNCTION = "int2string"
    CATEGORY = "Chibi-Nodes/Text"

    def int2string(self, Int):
        return(str(Int),)


class LoadImageExtended:
    def __init__(self):

        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                     },
                    "optional":{"vae": ("VAE", )}
                }

    CATEGORY = "Chibi-Nodes/Image"

    #changes here
    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "STRING", "STRING","INT","INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "LATENT","filename","image Info","width","height",)
    #

    FUNCTION = "load_image"

    def load_image(self, image, vae=None):


        image_path = folder_paths.get_annotated_filepath(image)
        filename = image_path.rsplit('/',1)[-1]

        im = Image.open(image_path)

        ### Start ai-info.py section, with no exif
        def type_changer(value):
            if value.isnumeric():
                return int(value)
            else:
                return value

        

        im.load()

        prompt = {}
        if "prompt" in im.info.keys():
            #comfyui, workflow is also available but we aren't getting that today
            # prompt = {}
            prompt.update({"prompt":json.loads(im.info["prompt"])})
        else:
            # automatic111, gosh this is a mess.
            if "parameters" in im.info.keys():
                parameters = im.info["parameters"]
                prompt = {"parameters": {}}
                parameters = re.split('(Negative prompt): |(Negative Template): |(Template): |(ControlNet): |\n',parameters)


                #removes None and new lines
                parameters_clean_none = []
                for i in range(0,len(parameters)):
                    if parameters[i] == None:
                        pass
                    elif parameters[i] == "":
                        pass
                    else:
                        parameters_clean_none.append(parameters[i])
                parameters = parameters_clean_none


                #settings field
                parameters_settings = {}
                for i in range(0,len(parameters)):
                    if parameters[i].split(":",1)[0] == "Steps":
                        parameters[i] = re.split(", ",parameters[i])
                        for k in parameters[i]:
                            k = k.split(": ",1)
                            if len(k) == 2:
                                k[1] = type_changer(k[1])

                                #makes "Size" : "(widthxheight)" into two keys
                                if k[0] == "Size":
                                    k[1] = k[1].split("x")
                                    for s in range(0,len(k[1])):
                                        k[1][s] = type_changer(k[1][s])
                                    parameters_settings.update({"width" : k[1][0]})
                                    parameters_settings.update({"height" : k[1][1]})
                                
                                else:
                                    parameters_settings.update({k[0]:k[1]})


                        parameters[i] = parameters_settings
                    
                #builder
                parameters_built = {}
                for i in range(0,len(parameters)):
                    match parameters[i]:
                        case "Negative prompt":
                            parameters_built.update({parameters[i]:parameters[i+1]})
                        case "Negative Template":
                            parameters_built.update({parameters[i]:parameters[i+1]})
                        case "Template":
                            parameters_built.update({parameters[i]:parameters[i+1]})
                        case "ControlNet":
                            parameters_built.update({parameters[i]:parameters[i+1]})
                        case dict():
                            parameters_built.update(parameters[i])
                        case _:
                            if i == 0:
                                parameters_built.update({"Positive prompt": parameters[i]})
                            pass


                
                prompt["parameters"] = parameters_built
        if type(prompt) == dict:
            prompt = json.dumps(prompt,indent=2)

        elif type(prompt) == str:
            prompt = json.dumps(json.loads(prompt),indent=2)


        ###end section


        im = ImageOps.exif_transpose(im)
        image = im.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        shape = image.shape
        width = shape[2]
        height = shape[1]
        if 'A' in im.getbands():
            mask = np.array(im.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        if vae != None:

            latent = image
            x = (latent.shape[1] // 8) * 8
            y = (latent.shape[2] // 8) * 8
            if latent.shape[1] != x or latent.shape[2] != y:
                x_offset = (latent.shape[1] % 8) // 2
                y_offset = (latent.shape[2] % 8) // 2
                latent = latent[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            latent = vae.encode(latent[:,:,:,:3])

            return (image, mask.unsqueeze(0),{"samples":latent},filename,str(prompt),width,height,)
        else:
            return (image, mask.unsqueeze(0),None,filename,str(prompt),width,height,)

    @classmethod
    def IS_CHANGED(s, image, vae=None):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, vae=None):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
    



class SimpleSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "model": ("MODEL",),
                    "sampler":(["Normal - euler","Normal - uni_pc","LCM Lora - lcm","SDXL Turbo - dpmpp_sde karras"],),
                    
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latents": ("LATENT", ),
                    "mode": (["txt2img","img2img"],)
                    
                     },
                     "optional": {
                         "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,"forceInput": True}),
                         }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Chibi-Nodes"


    def IS_CHANGED(s,seed):
        seed = random.seed()

    def sample(self, model,sampler, positive, negative, latents, mode,seed=None, scheduler="normal",sampler_name="euler"):

        # ['euler', 'euler_ancestral', 'heun', 'heunpp2', 'dpm_2', 'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive','dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'ddim', 'uni_pc', 'uni_pc_bh2']
        # ['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform']

        match sampler:
            case "Normal - euler":
                sampler_name = "uni_pc"
                steps = 20
                cfg = 7
            case "Normal - uni_pc":
                sampler_name = "uni_pc"
                steps = 20
                cfg = 7
            case "LCM Lora - lcm":
                sampler_name = "lcm"
                steps = 8
                cfg = 1.8
            case "SDXL Turbo - dpmpp_sde karras":
                sampler_name = "ddmpp_sde"
                steps = 8
                cfg = 1.8
                scheduler = "karras"
            case _:
                steps = 20
                cfg = 7

        match mode:
            case "txt2img":
                denoise = 1.0
            case "img2img":
                denoise = 0.6
            case _:
                denoise = 1.0

        

        if seed == None:
            seed = random.seed()
            seed = math.floor(random.random() * 10000000000000000)

        latent_image = latents["samples"]

        batch_inds = latents["batch_index"] if "batch_index" in latents else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latents:
            noise_mask = latents["noise_mask"]


        callback = latent_preview.prepare_callback(model, steps)

        samples = comfy.sample.sample(model=model, noise=noise, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, latent_image=latent_image,
                                    denoise=denoise, disable_noise=False, start_step=0, last_step=steps,
                                    force_full_denoise=True, noise_mask=noise_mask, callback=callback, disable_pbar=False, seed=seed)
        out = latents.copy()
        out["samples"] = samples
        return (out,)

class SeedGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "mode":(["Random","Fixed"],),
                "fixed_seed":("INT",{"default": 8008135, "min": 0, "max": 0xffffffffffffffff, "step": 1})

            },}
    RETURN_TYPES = ("INT","STRING",)
    RETURN_NAMES = ("seed","text")
    OUTPUT_NODE = False
    FUNCTION = "generator"
    CATEGORY = "Chibi-Nodes/Numbers"


    def IS_CHANGED(s,fixed_seed):
        seed = random.seed()

    def generator(self, mode,fixed_seed):
        if mode == "Random":
            fixed_seed = math.floor(random.random() * 10000000000000000)
        if mode == "Fixed":
            fixed_seed = fixed_seed
        
        return(fixed_seed,str(fixed_seed),)
    
class ImageAddText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return{
            "required":{

                "text": ("STRING",{"default":"Chibi-Nodes","multiline":True},),
                "font" : [sorted(folder_paths.get_filename_list("fonts"))],
                "font_size":("INT",{"default": 24, "min": 0, "max": 200, "step": 1}),
                "font_colour":(["black","white","red","green","blue"],),
                "invert_mask":([False,True],),
                "position_x":("INT",{"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "position_y":("INT",{"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "width":("INT",{"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height":("INT",{"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1})
            },
            "optional":{
                                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE","MASK","STRING",)
    RETURN_NAMES = ("IMAGE","MASK","text",)
    FUNCTION = "addtext"
    CATEGORY = "Chibi-Nodes/Image"

    def addtext(self, text, width, height,font,font_size,position_x,position_y,font_colour,invert_mask,image=None):
            if image != None:
                width = image.shape[2]
                height = image.shape[1]
                image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(),0,255).astype(np.uint8))
                image = image.convert("RGBA")
            else:
                image = Image.new('RGBA', (width,height), (255, 255, 255, 0))

            text_image = Image.new('RGBA', (width,height), (0, 255, 255, 0))
            imaget = ImageDraw.Draw(text_image,)

            msg = text
            imaget.fontmode = 'L'
            fnt = ImageFont.truetype(folder_paths.get_full_path("fonts", font), font_size)

            imaget.text((position_x,position_y),msg,font=fnt,fill=font_colour)

            if 'A' in text_image.getbands():
                mask = np.array(text_image.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            image.paste(text_image,(0,0),text_image)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if invert_mask:
                mask = 1.0 - mask
            
            
            return (image,mask.unsqueeze(0),text,)


NODE_CLASS_MAPPINGS = {
    "Loader":Loader,
    "SimpleSampler" : SimpleSampler,
    "Prompts": Prompts,
    "ImageTool": ImageTool,
    "Wildcards": Wildcards,
    "LoadEmbedding": LoadEmbedding,
    "ConditionText": ConditionText, 
    "ConditionTextMulti": ConditionTextMulti, 

    "Textbox":Textbox,



    "ImageSizeInfo" : ImageSizeInfo,
    "ImageSimpleResize" : ImageSimpleResize,
    "ImageAddText" : ImageAddText,

    "Int2String": Int2String,
    "LoadImageExtended": LoadImageExtended,
    "SeedGenerator" : SeedGenerator,
    "SaveImages":SaveImages,
}
