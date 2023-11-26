# ComfyUI-Chibi-Nodes

A Pack of nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## Current Nodes

At the moment there not many, But im sure to add more in the future.

Be aware, These are all very experimental right now so may have some bugs or issues.

![screenshot of current nodes](https://github.com/chibiace/ComfyUI-Chibi-Nodes/blob/main/screenshot.png)

### Loader

A more comprehensive Checkpoint, Vae loader with afew other features such as clip skip value selector and initial empty latent image generation

### Prompts

Combined Positive and Negative Prompts, Just to save some room.

### ImageTool (wip)

Uses PIL for some simple image manipulation. Won't do batch images yet, work in progress.

### Wildcards

Replaces a keyword from input text if given with a random line from a text file, if no input is given it will output just the random line

Can now return multiple random lines from the same file if desired, but be aware you can get doubles

wildcards files (some of which i've included which also need some more work) are stored under the extras/chibi-wildcards directory.

### LoadEmbedding

Appends the embedding text to the end of the input text ", (embedding:filename.pt:weight) "

### ConditionText

4 in 4 out, text to conditioning.

### SaveImages (wip)

VaeDecode and SaveImage mixed together. saves to the normal output directory but uses a unix time stamp as the filename eg. 1698462650*001*.png

### Textbox

A simple text box, if passthrough text is supplied it will update the textbox contents and also send it forward.

    "ImageSizeInfo" : ImageSizeInfo,
    "ImageSimpleResize" : ImageSimpleResize,
    "Int2String": Int2String,

### ImageSizeInfo

Displays the resolution of the input image, and sends it along as well as two int values for the dimensions.

### ImageSimpleResize

A quick way to make a large image into a size your GPU can handle.

largest sets the biggest dimension to the set size and keeps your aspect ratio

smallest sets the lowest dimension to the set size and keeps your aspect ratio

all sets all dimensions to the set size

height and width only sets the one you pick.

### Int2String

Takes a number, gives you a text string, for debugging i guess.

## Installation

download the repo to the custom_nodes directory. eg.

`cd custom_nodes`

`git clone https://github.com/chibiace/ComfyUI-Chibi-Nodes/`
