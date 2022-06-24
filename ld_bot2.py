import argparse, os, sys, glob
from doctest import script_from_examples
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import os, time, re, json, random
import discord
import presets
import asyncio
import traceback
from dotenv import load_dotenv

def make_filename(inp):
    '''Remove illegal filename characters'''
    return "".join(x for x in inp if (x.isalnum() or x in "._- "))

# will find an integer argument from a string in the form or "-foo 10" or "--bar 12" and return the value, using regex
def argument_parser(arg, str, default = None):
    # Sample input, arg="width": unreal engine --width 512 --height 256
    # Sample output: 512
    # Sample input, arg="height": unreal engine --width 512 --height 256
    # Sample output: 256
    # if not found, return None

    # Use regex to find the argument

    result = re.search(r'(?:--|—)' + arg + r'(\s+)?([^\s]+)', str)
    if result is None:
        # print(f"Could not find argument '{result}'")
        return default
    else:
        print(f"Found argument '{arg}' with value: {result.group(2)}")
        return result.group(2)

# remove found arguments from the string
def remove_argument(arg, str):
    result = re.search(r'(?:--|—)' + arg + r'(\s+)?([^\s]+)', str)
    if result is None:
        # print(f"Could not find argument '{result}'")
        return str
    else:
        # print(f"Found argument '{result}' with value: {result.group(1)}")
        return str.replace(result.group(0), '')

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


SCRIPT_START_TIME = time.time()


parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)

parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples"
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=200,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)

parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)

parser.add_argument(
    "--H",
    type=int,
    default=256,
    help="image height, in pixel space",
)

parser.add_argument(
    "--W",
    type=int,
    default=256,
    help="image width, in pixel space",
)

parser.add_argument(
    "--n_samples",
    type=int,
    default=4,
    help="how many samples to produce for the given prompt",
)

parser.add_argument(
    "--scale",
    type=float,
    default=5.0,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
opt = parser.parse_args()


config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

# config = OmegaConf.load("models/ldm/semantic_synthesis512/config.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
# model = load_model_from_config(config, "models/ldm/semantic_synthesis512/model.ckpt")  # TODO: check path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

if opt.plms:
    sampler = PLMSSampler(model)
else:
    sampler = DDIMSampler(model)

os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir





def generate(prompt, n_samples, n_iter, H, W):
    # prompt = opt.prompt

    sample_path = os.path.join(outpath, "samples2")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(n_samples * [""])
            for n in trange(n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(n_samples * [prompt])
                shape = [4, H//8, W//8]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    imgpath = os.path.join(sample_path, f"{base_count:04}.png")
                    Image.fromarray(x_sample.astype(np.uint8)).save(imgpath)
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()

    filename = 'outputs/final/'
    filename += make_filename(prompt)[:50] + '_run'

    run_num = 1
    while os.path.exists(filename+str(run_num)+'.png'):
        run_num += 1
    filename += str(run_num) + '.png'
    # outputs.append(filename)

    # imgpath = os.path.join(outpath, f'{prompt.replace(" ", "-")}.png')
    Image.fromarray(grid.astype(np.uint8)).save(filename)
    return [filename]

# print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")



# Bot
'''Discord interface'''
is_generating = False
to_notify = []

def init_discord_bot():
    global client, START_TIME

    # client.change_presence(activity=discord.Game(name='with AI'))

    @client.event
    async def on_ready():
        global bot_start_msg

        joined_servers = "\n".join(("+ Connected to server: '" + guild.name + "' (ID: " + str(guild.id) + ").") for guild in client.guilds)
        elapsed_time = str(round(time.time() - START_TIME, 1))
        print(joined_servers)

        await asyncio.sleep(1)

        bot_start_msg = "**Initialised in " + elapsed_time +" seconds! Current Time: " \
        + str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())) + " UTC**\nServers: ```diff\n" + joined_servers + "```"

        print("[OK] Initialised!")

    @client.event
    async def on_message(message):
        global history, seed, is_generating, to_notify

        START_TIME = time.time()

        if message.author == client.user:
            return

        print("="*50)
        print("Message from: '" + str(message.author) + "' saying '" + str(message.content) + "'.\nAttachments: '" + str(message.attachments) + '.')


        if str(message.channel).startswith('Direct Message with '):
            print("Ignoring Direct message.")
            return

        if len(message.content) == 0: # Attachment only
            return
        if str(message.content) == '.stop':
            await message.channel.send("**Stopping...**")
            # await client.logout()
            await client.close()
            raise KeyboardInterrupt
            return

        if message.content.startswith('.diffuse') or message.content.startswith('.rediffuse'):
            if message.channel.guild.id == presets.LAION_GULD_ID and message.channel.id != presets.LAION_BOT_CHANNEL:
                return

            await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="an image generate..."))
            # await client.change_presence(status=discord.Status.idle)

            is_generating = False
            if is_generating:
                await message.reply("Sorry, I'm currently in the middle of a generation\nI'll let you know when I'm free")
                to_notify += [message.author.mention]
                return
            is_generating = True

            if message.content.startswith('.rediffuse'):
                seed = random.randint(0, 2**32)
                # await message.reply("Reseeded to `" + str(seed) + "`!")
                prompt = str(message.content)[11:]
            else:
                prompt = str(message.content)[9:]

            waiting = await message.reply(f"{presets.LOADING_EMOJI} ***Generating...*** (LatentDiffusion)")
            generation_start_time = time.time()

            # Aliases
            # Read dictionary from aliases.json
            with open('aliases.json') as f:
                aliases = json.load(f)
            # Replace aliases in prompt
            for alias in aliases:
                prompt = prompt.replace(alias, aliases[alias])

            # Dimensions
            width = argument_parser('w', prompt, opt.W)
            height = argument_parser('h', prompt, opt.H)
            dimensions = argument_parser('d', prompt, None)
            if dimensions is not None:
                separators = [',', 'x', ':']
                for sep in separators:
                    if sep in dimensions:
                        width, height = dimensions.split(sep)
                        break
            width, height = int(width), int(height)

            # Batch size
            nx = int(argument_parser('n', prompt, opt.n_samples))
            ny = int(argument_parser('ny', prompt, 3))

            # Seed
            seed = argument_parser('s', prompt, seed)
            try:
                seed = int(seed)
            except:
                await message.reply("⚠ Seed must be an integer")
                await waiting.delete()
                return

            prompt = remove_argument('w', prompt)
            prompt = remove_argument('h', prompt)
            prompt = remove_argument('d', prompt)
            prompt = remove_argument('n', prompt)
            prompt = remove_argument('ny', prompt)
            prompt = remove_argument('s', prompt)

            prompt = prompt.strip()

            # Width and height must be divisible by 64
            if width % 64 != 0 or height % 64 != 0:
                closest_height = 64 * round(height / 64)
                closest_width = 64 * round(width / 64)
                await message.reply("⚠ Width and height must both be divisible by 64!\nClosest dimensions are (w: `" + str(closest_width) + "`, h: `" + str(closest_height) + "`)")
                await waiting.delete()
                return

            # if batch_size > 10:
            #    await message.reply("**Warning:** Discord only supports uploading 10 files at once.\nWill only generate 10 files.")
            #    batch_size = 10
            
            try:
                # gc.collect()
                # torch.cuda.empty_cache()
                filenames = generate(prompt, nx, ny, height, width)
            
            except Exception as e:
                print("Error:", e)
                traceback.print_exc()
                if str(e).startswith('CUDA out of memory'):
                    await message.reply("⚠ CUDA out of memory!\nTry lowering the generation parameters.")
                else:
                    await message.reply(f"Sorry, something went wrong.\nError: `{e}`")
                    await message.reply(f"Full traceback:\n```\n{traceback.format_exc()}\n```")
                await waiting.delete()
                return

            finally:
                print('seed', seed)
                # gc.collect()
                # torch.cuda.empty_cache()

            elapsed = time.time() - generation_start_time
            # info = f"**__{prompt}__**\nGeneration took `{round(elapsed, 2)}` seconds.\nSeed: `{seed}`."
            info = f"**__{prompt}__**\nGeneration took `{round(elapsed, 2)}` seconds.\n**Model:** `CompVis/LatentDiffusion`\n"
            # info = f"**__{prompt}__**\nGeneration took `{round(elapsed, 2)}` seconds. LatentDiffusion"
            info += f"Requested by <@{message.author.id}> (ID: `{message.author.id}`)"

            await waiting.delete()
            discord_files = [discord.File(cur_file) for cur_file in filenames]
            discord_files = discord_files[:10]
            # await message.reply(content=info, files=discord_files)
            asyncio.create_task(message.reply(content=info, files=discord_files))

            is_generating = False
            if to_notify:
                await message.channel.send(', '.join(to_notify) + " Hey, I'm free now so you can run a generation! :)")
            to_notify = []

            await client.change_presence(activity=discord.Game(name="the waiting game"))


        # if str(message.channel) not in presets.ALLOWED_CHANNELS:
        #     print("[x] REJECTING MESSAGE FROM CHANNEL: " + str(message.channel) + "...")


def start_all():
    '''Start everything to run model'''
    global client, START_TIME, history

    START_TIME = time.time()
    history = "\n"

    print("[INFO] Starting script...", flush=True)

    # Initialize discord stuff
    print("[INFO] Initializing Discord stuff...", flush=True)
    load_dotenv()
    client = discord.Client()
    print("[OK] Initialized Discord stuff!", flush=True)

    # Run Discord bot
    print("[INFO] Initializing Discord bot...", flush=True)
    init_discord_bot()
    print("[OK] Initialized Discord bot!", flush=True)

    # Retrieve Discord token
    print("[INFO] Getting Discord token...", flush=True)
    token = os.getenv('DISCORD_TOKEN')
    print("[OK] Got Discord token!", flush=True)


    print("[OK] Running Discord bot...", flush=True)
    client.run(token)

print('='*10 + " TechnoImage V1.1 " + '='*10)
print("[INFO] Models loaded in", round(time.time() - SCRIPT_START_TIME, 1), "seconds.")
time.sleep(0.5)

seed = -1
# seed = random.randint(0, 2**32)
start_all()

