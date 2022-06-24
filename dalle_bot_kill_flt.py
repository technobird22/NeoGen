import os
import presets
os.environ["WANDB_DISABLED"] = "true"

def make_collage(images, width, height, rows, columns):
    '''Make a collage of images'''
    # make a collage of images
    collage = Image.new('RGB', (width * columns, height * rows))
    x_offset = 0
    y_offset = 0
    for i in range(rows):
        for j in range(columns):
            cur_img = i*columns + j
            if cur_img >= len(images):
                return collage
            img = images[cur_img]
            collage.paste(img, (x_offset, y_offset))
            x_offset += width

        y_offset += height
        x_offset = 0

    return collage

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

def make_filename(inp):
    '''Remove illegal filename characters'''
    return "".join(x for x in inp if (x.isalnum() or x in "._- "))

# Model references

# dalle-mega
DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_COMMIT_ID = None

# if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# Discord
import re
import sys
import random
import urllib
import time
import traceback
import json
import discord
import asyncio
import math
from dotenv import load_dotenv

import safety_model

SCRIPT_START_TIME = time.time()

import jax
import jax.numpy as jnp
from dalle_mini import DalleBartProcessor

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel



# check how many devices are available
print("GPUs used:", jax.local_device_count())
if jax.local_device_count() < 1:
    raise RuntimeError("No devices available")


# Load dalle-mini
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)

from functools import partial

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)



processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

# number of predictions
# n_predictions = 8

# We can customize generation parameters
# gen_top_k = None
# gen_top_p = None
# temperature = None
# cond_scale = 3.0
# SEE BELOW: THE ARGUMENT PARSER HANDLES THESE NOW

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm import trange

prompt = "sunset over a lake in the mountains"
def do_run(prompt, n_predictions, gen_top_k, gen_top_p, temperature, cond_scale, key):
    tokenized_prompt = processor([prompt])
    tokenized_prompt = replicate(tokenized_prompt)

    print(f"Prompt: {prompt}\n")
    # generate images
    outputs = []
    images = []
    for m, i in enumerate(trange(max(n_predictions // jax.device_count(), 1))):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for n, decoded_img in enumerate(decoded_images):
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)
            # filename = '/home/technobird22/TBNeo_data/output/'
            filename = 'outputs/'
            # filename += make_filename(text_input)[:50] + '_it' + str(it) + f'_run{k}' + '_batch'
            filename += make_filename(prompt)[:50] + f'_{m*len(decoded_images) + n}_run'

            run_num = 1
            while os.path.exists(filename+str(run_num)+'.png'):
                run_num += 1
            filename += str(run_num) + '.png'
            img.save(filename)
            outputs.append(filename)

    n_predictions = len(outputs)
    columns = math.ceil(math.sqrt(n_predictions))
    rows = math.ceil(n_predictions / columns)
    if n_predictions == 3:
        columns = 3
        rows = 1
    collage = make_collage(images, 256, 256, rows, columns)
    filename = 'final_outputs/'
    filename += make_filename(prompt)[:50] + '_run'
    run_num = 1
    while os.path.exists(filename+str(run_num)+'.png'):
        run_num += 1
    filename += str(run_num) + '.png'
    collage.save(filename)
    collage_name = filename

    return outputs, collage_name















# Bot
'''Discord interface'''
is_generating = False
to_notify = []
this_instance_num = int(sys.argv[1])
do_gen = this_instance_num
instances = int(sys.argv[2])

# updates_loop = asyncio.new_event_loop()

def init_discord_bot():
    global client, START_TIME, safety_checker

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
        global history, seed, is_generating, to_notify, do_gen, instances, safety_checker

        START_TIME = time.time()

        if message.author == client.user:
            return

        print("="*50)
        print("Message from: '" + str(message.author) + "' saying '" + str(message.content) + "'.\nAttachments: '" + str(message.attachments) + '.')


        if len(message.content) > 50:
            print('-'*75)
        print('{0: <22}'.format(f'{message.guild} '), end='')
        print('{0: <22}'.format(f'> #{message.channel} '), end='')
        print('{0: <22}'.format(f'> {message.author} (ID: {message.author.id}) '), end='')
        if len(message.content) > 50:
            print(f":  ⤵\n  > '{message.content}'.\n{'-'*75}")
        else:
            print(f"> '{message.content}'.")

        if str(message.channel).startswith('Direct Message with '):
            print("Ignoring Direct message.")
            return

        if len(message.content) == 0: # Attachment only
            return
        if str(message.content) == f'.kill {sys.argv[3]}':
            await message.channel.send(f"**Stopping Bot on GPU {sys.argv[3]}...**")
            # await client.logout()
            await client.close()
            raise KeyboardInterrupt
            return

        if str(message.content) == '.stop' and message.author.id == presets.OWNER_ID:
            await message.channel.send("**Stopping...**")
            # await client.logout()
            await client.close()
            raise KeyboardInterrupt
            return

        if message.content.startswith('.setn') and message.author.id == presets.OWNER_ID:
            try:
                instances = int(message.content.split(' ')[1])
                do_gen = this_instance_num
                await message.channel.send("**Set number of instances to:** " + str(instances))
            except:
                await message.channel.send("**Error:** Could not set number of instances. Check that you've entered a valid int.")
            return

        # if message.content.startswith('.imagine') or message.content.startswith('.reimagine'):
        # if message.content.startswith('.wikiart') or message.content.startswith('.rewikiart'):
        if message.content.startswith('.dalle'):
            if message.channel.guild.id == presets.LAION_GULD_ID and message.channel.id != presets.LAION_BOT_CHANNEL:
                return

            message.content = message.content.replace('‘', '\'').replace('’', '\'')

            do_gen += 1
            do_gen %= instances
            if do_gen != 0:
                return
            # if do_gen:
            #     do_gen = False
            # else:
            #     do_gen = True
            #     return

            blocklist_file = "blocklist.txt"
            if not os.path.exists(blocklist_file):
                print("ERROR! Blocklist file not found.")
                await message.channel.send(f"**Error:** Could not find blocklist file. <@{presets.OWNER_ID}>")
            else:
                with open(blocklist_file, 'r') as f:
                    blocklist = f.read().splitlines()
                for word in blocklist:
                    search_word = f'\\b{word.lower()}\\b'
                    if bool(re.search(search_word, message.content, re.IGNORECASE)):
                        print("Ignoring message due to blocklist.")
                        suspect_str = re.search(search_word, message.content, re.IGNORECASE).group(0)
                        await message.reply(f"⚠️ **Warning:** <@{message.author.id}> (`{message.author.id}`) You are trying to use a blocked word.\nPlease check that your prompt abides by the rules.\nIf you believe this is to be a mistake, please contact <@{presets.OWNER_ID}>.\nSuspected text: ||`{suspect_str}`||", allowed_mentions=discord.AllowedMentions(users=False))
                        return

            await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="an image generate..."))
            # await client.change_presence(status=discord.Status.idle)

            is_generating = False
            if is_generating:
                await message.reply("Sorry, I'm currently in the middle of a generation\nI'll let you know when I'm free")
                to_notify += [message.author.mention]
                return
            is_generating = True

            prompt = str(message.content)[7:]

            waiting = await message.reply(f"{presets.LOADING_EMOJI} ***Generating...***\n(DALLE Mega)\n`Instance {this_instance_num + 1}`")
            generation_start_time = time.time()

            # Aliases
            # Read dictionary from aliases.json
            with open('aliases.json') as f:
                aliases = json.load(f)
            # Replace aliases in prompt
            for alias in aliases:
                prompt = prompt.replace(alias, aliases[alias])

            # Dimensions
            width = argument_parser('w', prompt, -1)
            height = argument_parser('h', prompt, -1)
            dimensions = argument_parser('d', prompt, None)
            if dimensions is not None:
                separators = [',', 'x', ':']
                for sep in separators:
                    if sep in dimensions:
                        width, height = dimensions.split(sep)
                        break
            
            if width != -1 or height != -1:
                await message.reply("Sorry, dimensions aren't yet supported for DALLE.")

            # Batch size
            batch_size = argument_parser('n', prompt, jax.device_count())
            # batch_size = argument_parser('n', prompt, 6)

            # Seed
            seed = random.randint(0, 2**32 - 1)
            seed = argument_parser('s', prompt, seed)
            try:
                seed = int(seed)
            except:
                await message.reply("⚠ Seed must be an integer")
                await waiting.delete()
                return
            key = jax.random.PRNGKey(seed)

            try:
                width, height, batch_size, seed = int(width), int(height), int(batch_size), int(seed)
            except:
                await message.reply("⚠ Width, height, batch size, and seed must all be integers")
                await waiting.delete()
                return
            
            gen_top_k = argument_parser('k', prompt, None)
            gen_top_p = argument_parser('p', prompt, None)
            temperature = argument_parser('t', prompt, None)
            cond_scale = argument_parser('c', prompt, '3.0')

            try:
                if gen_top_k is not None:
                    gen_top_k = float(gen_top_k)
                if gen_top_p is not None:
                    gen_top_p = int(gen_top_p)
                if temperature is not None:
                    temperature = float(temperature)
                cond_scale = float(cond_scale)
            except:
                await message.reply("⚠ Top K, temperature, and conditional scale must all be valid floats. Top P must be an integer")
                await waiting.delete()
                return

            # Width and height must be divisible by 64
            # if width % 64 != 0 or height % 64 != 0:
            #     closest_height = 64 * round(height / 64)
            #     closest_width = 64 * round(width / 64)
            #     await message.reply("⚠ Width and height must both be divisible by 64!\nClosest dimensions are (w: `" + str(closest_width) + "`, h: `" + str(closest_height) + "`)")
            #     await waiting.delete()
            #     return

            gen_limit = 9
            if batch_size > gen_limit and message.author.id != presets.OWNER_ID:
                # await message.reply("**Warning:** Discord only supports uploading 10 files at once.\nWill only generate 10 files.")
                await message.reply(f"Please be courteous to other users!\nI'll still generate your prompt, but only {gen_limit} images.")
                batch_size = gen_limit

            if batch_size < jax.device_count():
                await message.reply(f"Btw, you asked for {batch_size} images, but I have {jax.device_count()} GPUs available to me, so I'll generate you {jax.device_count()} images (for free) 🙂")
                batch_size = jax.device_count()

            if batch_size % jax.device_count() != 0:
                batch_size = batch_size // jax.device_count()
                batch_size = batch_size * jax.device_count()
                await message.reply(f"⚠ Batch size must be divisible by the number of GPUs ({jax.device_count()})\n**Generating {batch_size} images...**")
                # await waiting.delete()
                # return

            args_list = ['w', 'h', 'd', 'n', 's', 'iw', 'k', 'p', 't', 'c']
            for arg in args_list:
                prompt = remove_argument(arg, prompt)
            prompt = prompt.strip()

            try:
                # gc.collect()
                # torch.cuda.empty_cache()
                filenames, collage_name = do_run(prompt, batch_size, gen_top_k, gen_top_p, temperature, cond_scale, key)
            
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
            
            safety_ratings = []
            unsafe = False
            for filename in filenames:
                safety_rating = safety_checker.check(filename)
                if safety_rating > 0.55:
                    unsafe = True
                safety_ratings.append(safety_rating)

            elapsed = time.time() - generation_start_time
            # info = f"**__{prompt}__**\nGeneration took `{round(elapsed, 2)}` seconds.\nSeed: `{seed}`.\n**Model:** `wikiart-blip-captions/run5/ema_0.999_088000.pt`"
            # info = f"**__{prompt}__**\nGeneration took `{round(elapsed, 2)}` seconds.\nSeed: `{seed}`.\n**Model:** `GLID3XL/finetune.pt`"
            # info = f"**__{prompt}__**\nGeneration took `{round(elapsed, 2)}` seconds.\nSeed: `{seed}`.\n**Model:** `CompVis/LatentDiffusion`"
            info = f"**__{prompt}__**\nGeneration took `{round(elapsed, 2)}` seconds.\nSeed: `{seed}`.\n**Model:** `BorisDayma/DALLE-Mega`\n"
            info += f"\n**Safety rating (AVG):** `{round(sum(safety_ratings) / len(safety_ratings), 2)}`\n"
            info += f"**Safety ratings:** `{safety_ratings}`\n"
            info += f"*Requested by <@{message.author.id}>* (ID: `{message.author.id}`)"
            # info += f"\n\n**DEBUG**: output n: `{len(filenames)}`"

            await waiting.delete()
            # discord_files = [discord.File(cur_file) for cur_file in filenames][:10]
            discord_files = [discord.File(collage_name)]
            if unsafe:
                discord_files[0].filename = f"SPOILER_{collage_name}"
            if "matrix" in message.channel.name.lower():
                await message.channel.send(content=info, files=discord_files, allowed_mentions=discord.AllowedMentions(users=False, roles=False, everyone=False))
            else:
                await message.reply(content=info, files=discord_files, allowed_mentions=discord.AllowedMentions(users=False, roles=False, everyone=False))
            # await message.channel.send(f"**DEBUG**: output n: `{len(filenames)}`")
            # asyncio.create_task(message.channel.send(content=info, files=discord_files))
            # asyncio.run(await message.channel.send(content=info, files=discord_files))
            # run until complete
            # loop = asyncio.get_event_loop()
            # updates_loop.run_until_complete(message.reply(content=info, files=discord_files))
            # loop.run_until_complete(message.reply(content=info, files=discord_files))

            is_generating = False
            if to_notify:
                await message.channel.send(', '.join(to_notify) + " Hey, I'm free now so you can run a generation! :)")
            to_notify = []

            await client.change_presence(activity=discord.Game(name="the waiting game"))


        # if str(message.channel) not in presets.ALLOWED_CHANNELS:
        #     print("[x] REJECTING MESSAGE FROM CHANNEL: " + str(message.channel) + "...")


def start_all():
    '''Start everything to run model'''
    global client, START_TIME, history, safety_checker

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
    
    # Initialize safety checker
    print("[INFO] Initializing safety checker...", flush=True)
    safety_checker = safety_model.CheckImage()
    print("[OK] Initialized safety checker!", flush=True)

    # Retrieve Discord token
    print("[INFO] Getting Discord token...", flush=True)
    token = os.getenv('DISCORD_TOKEN')
    print("[OK] Got Discord token!", flush=True)


    print("[OK] Running Discord bot...", flush=True)
    client.run(token)

print('='*10 + " TechnoImage V1.1 " + '='*10)
# print("[INFO] Models loaded in", round(time.time() - SCRIPT_START_TIME, 1), "seconds.")
time.sleep(0.5)

seed = -1
# seed = random.randint(0, 2**32)
start_all()
