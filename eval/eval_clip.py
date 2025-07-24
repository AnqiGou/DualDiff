import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
from cldm.recognizer import TextRecognizer, crop_image
from easydict import EasyDict as edict
from tqdm import tqdm
import torch
import Levenshtein
import numpy as np
import math
import argparse
import json
import shutil
import clip
from PIL import Image
from torchvision import transforms
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import sklearn.preprocessing
import collections
import pathlib
import warnings

PRINT_DEBUG = False
num_samples = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("./ViT-B-32.pt", device=device, jit=False)
model.eval()

# Prepare a transformation for the images
image_transform = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.BICUBIC),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default='./ControlNet/controlnet_wukong_generated',
        help='path of generated images for eval'
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default='./data/wukong_word/test1k.json',
        help='json path for evaluation dataset'
    )
    args = parser.parse_args()
    return args


# Function to compute CLIP scores
def compute_clip_score(caption, image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image_transform(image).unsqueeze(0).to(device)

    # Encode the caption
    text = clip.tokenize([caption], truncate=True).to(device)

    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute the cosine similarity
    score = (image_features @ text_features.T).squeeze().cpu().numpy()
    return score

def main():
    args = parse_args()
    img_dir = args.img_dir
    input_json = args.input_json
    data_list = load_data(input_json)
    scores = []
    clip_score = []

    for i in tqdm(range(len(data_list)), desc='evaluate'):
        item_dict = get_item(data_list, i)
        img_name = item_dict['img_name'].split('.')[0]
        image_caption = item_dict['caption']
  
        print(image_caption)
        
        for j in range(num_samples):
            img_path = os.path.join(img_dir, img_name+f'_{j}.jpg')
            if os.path.exists(img_path):
                try:
                    score = compute_clip_score(image_caption, img_path)
                    scores.append(score)
                    print(f"CLIP score for {img_name}_{j}: {score}")
                except:
                    print(f"something wrong for {img_name}_{j}")
            else:
                print(f"Image {img_name}_{i} does not exist.")
            
    print(f'Done, imgs={len(scores)}, clip_socre={np.array(scores).mean():.4f}')

if __name__ == "__main__":
    main()
