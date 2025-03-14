import folder_paths
import node_helpers
import comfy.sd
import comfy.utils


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


NODE_CLASS_MAPPINGS = {
    "FlexGuidance": FlexGuidance,
    "FlexLoraLoader": FlexLoraLoader,
    "FlexLoraLoaderModelOnly": FlexLoraLoaderModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlexGuidance": "Flex Guidance",
    "FlexLoraLoader": "Flex LoRA Loader",
    "FlexLoraLoaderModelOnly": "Flex LoRA Loader (Model Only)",
}
