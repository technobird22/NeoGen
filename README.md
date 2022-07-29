# NeoGen
*A tool for generating awesome AI art*

Try out a fully functional version of the bot in one of the Discord servers below :)
- **LAION Discord**: Awesome community where a lot of the text to image AIs started. Amazing community, with lots of knowledgable people. Be nice and follow the rules [https://discord.gg/9ganQST2RK](https://discord.gg/9ganQST2RK)
- **Land of AI Discord**: a smaller Discord server where there are many other bots running, and you can play with them and try them out: [https://discord.gg/vrAhrh5DkG](https://discord.gg/vrAhrh5DkG)

## NOTE: code on here is very preliminary and it is currently being rewritten
The files here depend on other models, but haven't added them as git submodules yet.
Am currently rewriting most of it to support a priority queuing system

**Notes:**
- Files ending in `_flt` contain an implementation of [LAION's NSFW filter](https://github.com/LAION-AI/CLIP-based-NSFW-Detector). However they don't work well because they require too much VRAM from a single GPU. Will work on getting a separate script for filtering that runs on a separate GPU.
As such, **they may be out of date compared to the non-filtered scripts.**
- The `dalle` scripts are intended to be placed within the [DALLE Mini](https://github.com/borisdayma/dalle-mini) directory tree.
- The `ld` scripts are for LatentDiffusion, and are intended to be placed within the [LatentDiffusion](https://github.com/CompVis/latent-diffusion) directory tree.


Massive thanks to LAION AI and their sponsor, Stability.ai for providing the compute resources, without which, development and testing of the bot would not have been possible.
