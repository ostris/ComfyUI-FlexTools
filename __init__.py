from functools import partial
import torch
from comfy.model_base import Flux
import folder_paths
import node_helpers
import comfy.sd
import comfy.utils
import comfy.patcher_extension
import comfy.conds
from comfy.patcher_extension import CallbacksMP, WrappersMP


class FlexGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "bypass_guidance_embedder": (["yes", "no"], {"default": "no"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "do_it"

    CATEGORY = "advanced/conditioning/flux"

    def do_it(self, conditioning, guidance, bypass_guidance_embedder):
        bypass_guidance_embedder = bypass_guidance_embedder == "yes"
        guidance_value = guidance
        if bypass_guidance_embedder:
            guidance_value = None
        cond = node_helpers.conditioning_set_values(
            conditioning, {"guidance": guidance_value}
        )
        return (cond, )


class FlexLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",
                       "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads Loras and automatically converts Flux loras to Flex loras."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            # convert it to Flex LoRA
            # the pruning squashed double idx 5-15 into idx 4
            # making idx 16, 17, 18 become 5, 6, 7
            # we will drop double blocks with idx 5-15
            # and move idx 16, 17, 18 to 5, 6, 7
            # it is best to drop idx 4 as well since it is so divergent due to pruning

            # loras have different naming patterns, the ones I know about are below
            block_test_targets = [
                "double_blocks.{idx}.",
                "transformer.transformer_blocks.{idx}.",
                "lora_unet_double_blocks_{idx}_",
                "lycoris_unet_double_blocks_{idx}_",
                "lycoris_transformer_blocks_{idx}_",
                "lora_transformer_blocks_{idx}_",
            ]

            # we trained the guidance embedder from scratch, the weights will not match at all
            # loras will destroy it, so we will ignore it
            ignore_if_contains = [
                "guidance_in",
                "guidance_embedder"
            ]

            # check if any of the keys start with the block_test_targets with idx 8-18,
            # if they do, then this it is a Flux lora

            is_flux_lora = False
            for idx in range(8, 19):
                for target in block_test_targets:
                    if any(k.startswith(target.format(idx=idx)) for k in lora.keys()):
                        is_flux_lora = True
                        break
                if is_flux_lora:
                    break

            if is_flux_lora:
                flex_lora = {}
                drop_idxs = list(range(4, 16))
                move_idxs = {16: 5, 17: 6, 18: 7}
                for k, v in lora.items():
                    if any(k.startswith(target.format(idx=idx)) for target in block_test_targets for idx in drop_idxs):
                        # drop it
                        continue
                    if any(target in k for target in ignore_if_contains):
                        continue
                    for old_idx, new_idx in move_idxs.items():
                        replaced = False
                        for target in block_test_targets:
                            formatted_target = target.format(idx=old_idx)
                            if k.startswith(formatted_target):
                                k = k.replace(formatted_target,
                                              target.format(idx=new_idx))
                                replaced = True
                                break
                        if replaced:
                            break
                    flex_lora[k] = v
                lora = flex_lora

            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class FlexLoraLoaderModelOnly(FlexLoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "lora_name": (folder_paths.get_filename_list("loras"), ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0)[0],)


def flex2_concat_cond(self: Flux, **kwargs):
    # will break otherwise
    return None


def flex2_extra_conds(self, **kwargs):
    out = self._flex2_orig_extra_conds(**kwargs)
    noise = kwargs.get("noise", None)
    device = kwargs["device"]
    flex2_concat_latent = kwargs.get("flex2_concat_latent", None)
    flex2_concat_latent_no_control = kwargs.get(
        "flex2_concat_latent_no_control", None)
    control_strength = kwargs.get("flex2_control_strength", 1.0)
    control_start_percent = kwargs.get("flex2_control_start_percent", 0.0)
    control_end_percent = kwargs.get("flex2_control_end_percent", 0.1)
    if flex2_concat_latent is not None:
        flex2_concat_latent = comfy.utils.resize_to_batch_size(
            flex2_concat_latent, noise.shape[0])
        flex2_concat_latent = self.process_latent_in(flex2_concat_latent)
        flex2_concat_latent = flex2_concat_latent.to(device)
        out['flex2_concat_latent'] = comfy.conds.CONDNoiseShape(
            flex2_concat_latent)
    if flex2_concat_latent_no_control is not None:
        flex2_concat_latent_no_control = comfy.utils.resize_to_batch_size(
            flex2_concat_latent_no_control, noise.shape[0])
        flex2_concat_latent_no_control = self.process_latent_in(
            flex2_concat_latent_no_control)
        flex2_concat_latent_no_control = flex2_concat_latent_no_control.to(
            device)
        out['flex2_concat_latent_no_control'] = comfy.conds.CONDNoiseShape(
            flex2_concat_latent_no_control)

    out['flex2_control_start_percent'] = comfy.conds.CONDConstant(
        control_start_percent)
    out['flex2_control_end_percent'] = comfy.conds.CONDConstant(
        control_end_percent)

    return out


def flex_apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
    sigma = t
    xc = self.model_sampling.calculate_input(sigma, x)
    if c_concat is not None:
        xc = torch.cat([xc] + [c_concat], dim=1)

    flex2_control_start_sigma = 1.0 - \
        kwargs.get("flex2_control_start_percent", 0.0)
    flex2_control_end_sigma = 1.0 - \
        kwargs.get("flex2_control_end_percent", 1.0)

    flex2_concat_latent_active = kwargs.get("flex2_concat_latent", None)
    flex2_concat_latent_inactive = kwargs.get(
        "flex2_concat_latent_no_control", None)

    sigma_float = sigma.mean().cpu().item()
    sigma_int = int(sigma_float * 1000)

    # simple, but doesnt work right because of shift
    is_being_controlled = sigma_float <= flex2_control_start_sigma and sigma_float >= flex2_control_end_sigma

    sigmas = transformer_options.get("sample_sigmas", None)

    if sigmas is not None:
        # we have all the timesteps here, determine what percent we are through the
        # timesteps we are doing. This way is more intuitive to user.
        all_timesteps = [int(sigma.cpu().item() * 1000) for sigma in sigmas]
        current_idx = all_timesteps.index(sigma_int)
        current_percent = current_idx / len(all_timesteps)
        current_percent_sigma = 1.0 - current_percent
        is_being_controlled = current_percent_sigma <= flex2_control_start_sigma and current_percent_sigma >= flex2_control_end_sigma

    if is_being_controlled:
        # it is active
        xc = torch.cat([xc] + [flex2_concat_latent_active], dim=1)
    else:
        # it is inactive
        xc = torch.cat([xc] + [flex2_concat_latent_inactive], dim=1)

    context = c_crossattn
    dtype = self.get_dtype()

    if self.manual_cast_dtype is not None:
        dtype = self.manual_cast_dtype

    xc = xc.to(dtype)
    t = self.model_sampling.timestep(t).float()
    if context is not None:
        context = context.to(dtype)

    extra_conds = {}
    for o in kwargs:
        extra = kwargs[o]
        if hasattr(extra, "dtype"):
            if extra.dtype != torch.int and extra.dtype != torch.long:
                extra = extra.to(dtype)
        extra_conds[o] = extra

    t = self.process_timestep(t, x=x, **extra_conds)
    model_output = self.diffusion_model(
        xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
    return self.model_sampling.calculate_denoised(sigma, model_output, x)


class Flex2Conditioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "vae": ("VAE", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "bypass_guidance_embedder": (["yes", "no"], {"default": "no"}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "control_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            "optional": {
                "latent": ("LATENT", ),
                "inpaint_image": ("IMAGE", ),
                "inpaint_mask": ("MASK", ),
                "control_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latent")
    FUNCTION = "do_it"

    CATEGORY = "advanced/conditioning/flex"

    def do_it(
        self,
        model,
        vae,
        positive,
        negative,
        guidance,
        bypass_guidance_embedder,
        control_strength,
        control_start_percent,
        control_end_percent,
        latent=None,
        inpaint_image=None,
        inpaint_mask=None,
        control_image=None,
    ):
        # replace concat_cond of the flux model as default one breaks flex2
        # todo, find a better way to do this
        if not hasattr(model.model, "_flex2_orig_concat_cond"):
            model.model._flex2_orig_concat_cond = model.model.concat_cond
            model.model.concat_cond = partial(flex2_concat_cond, model.model)
        # replace extra_conds
        if not hasattr(model.model, "_flex2_orig_extra_conds"):
            model.model._flex2_orig_extra_conds = model.model.extra_conds
            model.model.extra_conds = partial(flex2_extra_conds, model.model)

        if not hasattr(model.model, "_flex2_orig_apply_model"):
            model.model._flex2_orig_apply_model = model.model._apply_model
            model.model._apply_model = partial(flex_apply_model, model.model)

        # masks come in as (bs, h, w)  0 to 1
        # images come in as (bs, h, w, c) 0 to 1
        # latents come in as (bs, ch, h, w) -1 to 1
        batch_size = 1
        latent_height: int = None
        latent_width: int = None

        # DETERIMINE SIZES
        # size order is latent size, inpaint size, control size
        if latent is not None:
            latent_height = latent['samples'].shape[2]
            latent_width = latent['samples'].shape[3]
            if latent['samples'].shape[1] == 4:
                # make it 16 channels
                latent['samples'] = torch.cat(
                    [latent['samples'] for _ in range(4)], dim=1)
            batch_size = latent['samples'].shape[0]
        elif inpaint_image is not None:
            batch_size = inpaint_image.shape[0]
            latent_height = inpaint_image.shape[1] // 8
            latent_width = inpaint_image.shape[2] // 8
        elif control_image is not None:
            batch_size = control_image.shape[0]
            latent_height = control_image.shape[1] // 8
            latent_width = control_image.shape[2] // 8
        else:
            raise ValueError("No latent, inpaint or control image provided")

        img_width = latent_width * 8
        img_height = latent_height * 8

        # apply differential diffusion to model
        model = model.clone()
        # model.set_model_denoise_mask_function(self.denoise_mask_function)

        # guidance embedder
        bypass_guidance_embedder = bypass_guidance_embedder == "yes"

        guidance_value = guidance
        if bypass_guidance_embedder:
            guidance_value = None
        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "guidance": guidance_value
            }
        )

        # out input is our latent(16) + (inpaint_image(16) + mask(1) + control image(16))
        # We just need to build the non latent part
        concat_latent = torch.zeros(
            (batch_size, 33, latent_height, latent_width),
            device='cpu',
            dtype=torch.float32
        )

        # when we are not using inpainting, the mask needs to be all 1s
        concat_latent[:, 16:17, :, :] = torch.ones(
            (batch_size, 1, latent_height, latent_width),
            device='cpu',
            dtype=torch.float32
        )

        out_latent = {
            "samples": torch.zeros(
                (batch_size, 16, latent_height, latent_width),
                device='cpu',
                dtype=torch.float32
            )
        }

        # inpaint
        if inpaint_image is not None:
            if inpaint_image.shape[1] != img_height or inpaint_image.shape[2] != img_width:
                inpaint_image = torch.nn.functional.interpolate(
                    inpaint_image.permute(0, 3, 1, 2),
                    size=(img_height, img_width),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)

            if inpaint_mask is not None:
                inpaint_mask_latent = torch.nn.functional.interpolate(
                    inpaint_mask.reshape(
                        (-1, 1, inpaint_mask.shape[-2], inpaint_mask.shape[-1])),
                    size=(latent_height, latent_width),
                    mode="bilinear"
                )
            else:
                # make it all 1s
                inpaint_mask_latent = torch.ones(
                    (batch_size, 1, latent_height, latent_width),
                    device='cpu',
                    dtype=torch.float32
                )

            inpaint_latent_orig = vae.encode(inpaint_image)

            # set this so we can partially denoise with it if desired
            out_latent["samples"] = inpaint_latent_orig.clone()

            # mask is currently 0 for keep and 1 for inpaint
            inpaint_latent_masked = inpaint_latent_orig * \
                (1 - inpaint_mask_latent)

            # put it on our concat latent
            concat_latent[:, 0:16, :, :] = inpaint_latent_masked
            # put the mask in the last channel, 0 for keep, 1 for inpaint
            concat_latent[:, 16:17, :, :] = inpaint_mask_latent

        concat_latent_no_control = concat_latent.clone()

        # control
        if control_image is not None:
            if control_image.shape[1] != img_height or control_image.shape[2] != img_width:
                control_image = torch.nn.functional.interpolate(
                    control_image.permute(0, 3, 1, 2),
                    size=(img_height, img_width),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)

            control_latent = vae.encode(control_image)

            # put the control image in the last 16 channels
            concat_latent[:, 17:33, :, :] = control_latent * \
                control_strength

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(
                conditioning,
                {
                    "flex2_concat_latent": concat_latent,
                    "flex2_concat_latent_no_control": concat_latent_no_control,
                    "flex2_control_strength": control_strength,
                    "flex2_control_start_percent": control_start_percent,
                    "flex2_control_end_percent": control_end_percent,
                }
            )
            out.append(c)
        positive = out[0]
        negative = out[1]

        return (model, positive, negative, out_latent)


NODE_CLASS_MAPPINGS = {
    "FlexGuidance": FlexGuidance,
    "FlexLoraLoader": FlexLoraLoader,
    "FlexLoraLoaderModelOnly": FlexLoraLoaderModelOnly,
    "Flex2Conditioner": Flex2Conditioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlexGuidance": "Flex Guidance",
    "FlexLoraLoader": "Flex LoRA Loader",
    "FlexLoraLoaderModelOnly": "Flex LoRA Loader (Model Only)",
    "Flex2Conditioner": "Flex 2 Conditioner",
}
