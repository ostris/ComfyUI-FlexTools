# Flex.1 tools

Some tools to help with [Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) inference on Comfy UI.

## Installation

Clone this repo into your `custom_nodes` directory.

## Nodes

- **Flex Guidance**: Allows you to set the guidance for the Flex.1 guidance embedder, or bypass it completly to use true CFG.
- **Flex LoRA Loader**: Loads LoRAs and automatically prunes them to Flex.1 layers. It will not be perfect as Flex is heavily diverged from Flux dev and is not a direct ancenstor of it, but it should be good enough for most purposes.
- **Flex LoRA Loader (Model Only)**: Same as Flex LoRA Loader, but only loads the model and not the text encoder. Most Flux LoRAs do not train the text encoder.