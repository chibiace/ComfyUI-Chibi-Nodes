import torch
import folder_paths
import comfy.sd

### GLOBALS ###
MAX_RESOLUTION=8192


class Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "Checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
            "Vae": (folder_paths.get_filename_list("vae"), ),
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
        ckpt = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=False, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

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

    
NODE_CLASS_MAPPINGS = {
    "Loader":Loader,
    "Prompts": Prompts
    
}
