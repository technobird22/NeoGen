# Massive thank you to @johnpaulbin for helping significantly with the classifier.

import os, requests, glob
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import torch
from tqdm import tqdm
from PIL import Image
import clip
import open_clip
import statistics
import gc
gc.collect()
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache



def get_cache_folder(clip_model):
    """get cache folder for given clip model"""
    from os.path import expanduser  # pylint: disable=import-outside-toplevel

    home = expanduser("~")

    cache_folder = "~/.cache/clip/" + clip_model.replace("/", "_").replace(":","_")

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder, exist_ok=True)

    return cache_folder


@lru_cache(maxsize=None)
def load_safety_model(clip_model):
    """load the safety model"""
    import autokeras as ak  # pylint: disable=import-outside-toplevel
    from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel

    cache_folder = get_cache_folder(clip_model)

    if clip_model == "ViT-L/14":
        model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
        dim = 768
    elif clip_model == "ViT-B/32":
        model_dir = cache_folder + "/clip_autokeras_nsfw_b32"
        dim = 512
    elif clip_model == "open:ViT-B/32":
        model_dir = cache_folder + "/openclip_autokeras_nsfwb32"
        dim = 512
    else:
        raise ValueError("Unknown clip model")

    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)

        from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
        
        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        if clip_model == "ViT-L/14":
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        elif clip_model == "ViT-B/32":
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip"
            )
        elif clip_model == "open:ViT-B/32":
            url_model = (
                "https://github.com/johnpaulbin/CLIP-based-NSFW-Detector/releases/download/files/openclip_autokeras_nsfwb32.zip"
            )
        else:
            raise ValueError("Unknown model {}".format(clip_model))  # pylint: disable=consider-using-f-string
        urlretrieve(url_model, path_to_zip_file)
        import zipfile  # pylint: disable=import-outside-toplevel

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.predict(np.random.rand(10**3, dim).astype("float32"), batch_size=10**3)

    return loaded_model


#nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])

class CheckImage:
    def __init__(self):
        #self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log = False)
        self.context = {'jobs': {}}

        #self.clipmodel, self.preprocess = clip.load("ViT-B/32", device="cpu", jit=True)

        # self.clipmodel, _, self.preprocess = open_clip.create_model_and_transforms(
        #     "ViT-B-32", pretrained="laion400m_e32", device="cpu", jit=True
        # )
        # self.safety_model = load_safety_model("open:ViT-B/32")

        self.clipmodel, self.preprocess = clip.load("ViT-B/32", device="cuda:0", jit=True)
        self.safety_model = load_safety_model("ViT-B/32")

    def check(self, path):
        try:
            # processedimages = self.preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0)
            img = Image.open(path)
            processedimages = self.preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                image_features = self.clipmodel.encode_image(processedimages)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                predictions = self.safety_model.predict(image_features.cpu().numpy(), batch_size=10**3, verbose=0)
                count = statistics.mean([float(prediction[0]) for prediction in predictions])
                # if count > 0.55:
                #     print("WARNING: Verdict: Likely NSFW")
                result = round(count, 5)

            return result
        except Exception as e:
            print("An error has occured:", e)
            return -1


if __name__ == "__main__":
    print("Starting...")
    print("Loading model...")
    checker = CheckImage()
    print("Model loaded!")

    path = '/opt/technobird22/dalle-mini_2/outputs_a/'
    output_file = 'flagged2.txt'
    files = glob.glob(path + '/*.png')
    for file in tqdm(files, position=0, leave=True):
        # print("File:", os.path.basename(file), "   VERDICT:", checker.check(file))
        if checker.check(file) > 0.55:
            print("\nCONTENT WARNING:    File:", os.path.basename(file), "   VERDICT:", checker.check(file))
            with open(output_file, 'a') as f:
                f.write(file + '\n')
        else:
            print('.', end='', flush=True)
    print("Done.")