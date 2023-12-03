# ComfyUI Chibi Nodes

Enhance your [ComfyUI](https://github.com/comfyanonymous/ComfyUI) experience with this collection of experimental nodes.

## Current Nodes

While the selection is currently limited, expect more additions in the future. Please note that these nodes are in an experimental stage and may contain bugs or issues.

![Screenshot of Current Nodes](https://github.com/chibiace/ComfyUI-Chibi-Nodes/blob/main/screenshot.png)

### Loader

A comprehensive Checkpoint VAE loader with additional features such as a clip skip value selector and initial empty latent image generation.

### Prompts

Combined positive and negative prompts, designed to save space.

### ImageTool (Work in Progress)

Utilizes PIL for simple image manipulation. Currently, batch image processing is not supported, but development is ongoing.

### Wildcards

Replaces a keyword from input text with a random line from a text file. If no input is given, it will output just a random line. It can now return multiple random lines from the same file, but be cautious as duplicates may occur. Wildcards files are stored under the extras/chibi-wildcards directory.

### LoadEmbedding

Appends the embedding text to the end of the input text ", (embedding:filename.pt:weight) ".

### ConditionText

A 4-in-4-out node for text conditioning.

### SaveImages

This feature merges the functionalities of VaeDecode and SaveImage, offering three distinct modes for filenames:

1. **Timestamp:** Saves files to the standard output directory with a Unix timestamp as the filename (e.g., 1698462650001.png).
2. **Fixed:** Utilizes the `fixed_filename` variable for individual images and batches (e.g., output_001.png).
3. **Fixed Single:** Uses the `fixed_filename` variable but saves all images under the same filename, not ideal for batches (e.g., output.png).

### Textbox

A simple text box. If pass-through text is supplied, it updates the textbox contents and sends it forward.

### ImageSizeInfo

Displays the resolution of the input image and sends it along with two int values for the dimensions.

### ImageSimpleResize

A quick way to resize a large image to a size your GPU can handle:

- "largest" sets the biggest dimension to the specified size while maintaining aspect ratio.
- "smallest" sets the lowest dimension to the specified size while maintaining aspect ratio.
- "all" sets all dimensions to the specified size.
- "height" and "width" only set the chosen dimension.

### Int2String

Converts a number into a text string, primarily for debugging purposes.

## Installation

To install, download the repository to the custom_nodes directory:

```bash
cd custom_nodes
git clone https://github.com/chibiace/ComfyUI-Chibi-Nodes/
```
