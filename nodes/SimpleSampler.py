import random
import math
import comfy.sample
import latent_preview


class SimpleSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sampler": (
                    [
                        "Normal - euler",
                        "Normal - uni_pc",
                        "LCM Lora - lcm",
                        "SDXL Turbo - dpmpp_sde karras",
                    ],
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latents": ("LATENT",),
                "mode": (["txt2img", "img2img"],),
            },
            "optional": {
                "seed": (
                    "INT",
                    {
                        "forceInput": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Chibi-Nodes"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        random.seed()
        return float("NaN")

    def sample(
        self,
        model,
        sampler,
        positive,
        negative,
        latents,
        mode,
        seed=None,
        scheduler="normal",
        sampler_name="euler",
    ):

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

        if seed is not None:
            random.seed(seed)
        else:
            random.seed()
            seed = math.floor(random.random() * 10000000000000000)

        latent_image = latents["samples"]

        batch_inds = latents["batch_index"] if "batch_index" in latents else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latents:
            noise_mask = latents["noise_mask"]

        callback = latent_preview.prepare_callback(model, steps)

        samples = comfy.sample.sample(
            model=model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            disable_noise=False,
            start_step=0,
            last_step=steps,
            force_full_denoise=True,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=False,
            seed=seed,
        )
        out = latents.copy()
        out["samples"] = samples
        return (out,)
