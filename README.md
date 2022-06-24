# NeoGen
*A Discord bot that can interface with a variety of AI models.*

## NOTE: code on here is very preliminary and it is currently being rewritten
Depend on other models, but haven't added them as git submodules yet.
Am currently rewriting most of it to support a priority queuing system

Notes:
- Files ending in `_flt` contain an implementation of LAION's NSFW filter. However they don't work well because they require too much VRAM from a single GPU. Will work on getting a separate script for filtering that runs on a separate GPU.
As such, they may be out of date compared to the non-filtered scripts.
- The `dalle` scripts are intended to be placed within the [DALLE Mini](https://github.com/borisdayma/dalle-mini) directory tree.
- The `ld` scripts are LatentDiffusion, and are intended to be placed within the [LatentDiffusion](https://github.com/CompVis/latent-diffusion) directory tree.
