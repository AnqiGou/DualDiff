import os
import math
import torch
import random
import re
import numpy as np
import cv2
import einops
import time
import gradio as gr
from gradio.components import Component
from PIL import ImageFont
from create_model import create_model, load_state_dict
from ddim_hacked import DDIMSampler
from process_data import draw_glyph, draw_glyph2
from pytorch_lightning import seed_everything
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.base import TorchModel
from modelscope.preprocessors.base import Preprocessor
from modelscope.pipelines.base import Model, Pipeline
from modelscope.utils.config import Config
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.models.builder import MODELS
from modelscope.hub.snapshot_download import snapshot_download
from bert_tokenizer import BasicTokenizer
import matplotlib.pyplot as plt
import json
import os
import torch
import torch.fft as fft
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

checker = BasicTokenizer()
BBOX_MAX_NUM = 8
PLACE_HOLDER = '*'
max_chars = 20

class MyModel(TorchModel):
    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.use_fp16 = kwargs.get('use_fp16', False)
        self.use_translator = kwargs.get('use_translator', False)
        self.init_model(**kwargs)
    def check_overlap_polygon(self, rect_pts1, rect_pts2):
        poly1 = cv2.convexHull(rect_pts1)
        poly2 = cv2.convexHull(rect_pts2)
        rect1 = cv2.boundingRect(poly1)
        rect2 = cv2.boundingRect(poly2)
        if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
            return True
        return False

    def generate_rectangles(self, w, h, n, max_trys=200):
        img = np.zeros((h, w, 1), dtype=np.uint8)
        rectangles = []
        attempts = 0
        n_pass = 0
        low_edge = int(max(w, h)*0.3 if n <= 3 else max(w, h)*0.2)  # ~150, ~100
        while attempts < max_trys:
            rect_w = min(np.random.randint(max((w*0.5)//n, low_edge), w), int(w*0.8))
            ratio = np.random.uniform(4, 10)
            rect_h = max(low_edge, int(rect_w/ratio))
            rect_h = min(rect_h, int(h*0.8))
            # gen rotate angle
            rotation_angle = 0
            rand_value = np.random.rand()
            if rand_value < 0.7:
                pass
            elif rand_value < 0.8:
                rotation_angle = np.random.randint(0, 40)
            elif rand_value < 0.9:
                rotation_angle = np.random.randint(140, 180)
            else:
                rotation_angle = np.random.randint(85, 95)
            # rand position
            x = np.random.randint(0, w - rect_w)
            y = np.random.randint(0, h - rect_h)
            # get vertex
            rect_pts = cv2.boxPoints(((rect_w/2, rect_h/2), (rect_w, rect_h), rotation_angle))
            rect_pts = np.int32(rect_pts)
            # move
            rect_pts += (x, y)
            # check boarder
            if np.any(rect_pts < 0) or np.any(rect_pts[:, 0] >= w) or np.any(rect_pts[:, 1] >= h):
                attempts += 1
                continue
            # check overlap
            if any(self.check_overlap_polygon(rect_pts, rp) for rp in rectangles):
                attempts += 1
                continue
            n_pass += 1
            cv2.fillPoly(img, [rect_pts], 255)
            rectangles.append(rect_pts)
            if n_pass == n:
                break
        print("attempts:", attempts)
        if len(rectangles) != n:
            raise gr.Error(f'Failed in auto generate positions after {attempts} attempts, try again!')
        return img
    
    def forward(self, input_tensor, **forward_params):
        tic = time.time()
        str_warning = ''
        # get inputs
        seed = input_tensor.get('seed', -1)
        if seed == -1:
            seed = random.randint(0, 99999999)
        seed_everything(seed)
        prompt = input_tensor.get('prompt')
        draw_pos = input_tensor.get('draw_pos')
        ori_image = input_tensor.get('ori_image')

        mode = forward_params.get('mode')
        sort_priority = forward_params.get('sort_priority', '↕')
        show_debug = forward_params.get('show_debug', False)
        revise_pos = forward_params.get('revise_pos', False)
        img_count = forward_params.get('image_count', 4)
        ddim_steps = forward_params.get('ddim_steps', 20)
        w = forward_params.get('image_width', 512)
        h = forward_params.get('image_height', 512)
        strength = forward_params.get('strength', 1.0)
        cfg_scale = forward_params.get('cfg_scale', 9.0)
        eta = forward_params.get('eta', 0.0)
        a_prompt = forward_params.get('a_prompt', 'best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks')
        n_prompt = forward_params.get('n_prompt', 'low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture')

        prompt, texts = self.modify_prompt(prompt)
        if prompt is None and texts is None:
            return None, -1, "You have input Chinese prompt but the translator is not loaded!", ""
        n_lines = len(texts)
        edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
    
        if draw_pos is None: 
            pos_imgs = self.generate_rectangles(w, h, n_lines, max_trys=500)

        elif isinstance(draw_pos, str):
            draw_pos = cv2.imread(draw_pos)[..., ::-1]
            assert draw_pos is not None, f"Can't read draw_pos image from{draw_pos}!"
            pos_imgs = 255-draw_pos
        elif isinstance(draw_pos, torch.Tensor):
            pos_imgs = draw_pos.cpu().numpy()
        else:
            assert isinstance(draw_pos, np.ndarray), f'Unknown format of draw_pos: {type(draw_pos)}'
        
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        
        pos_imgs = self.separate_pos_imgs(pos_imgs, sort_priority)
        print(np.unique(pos_imgs[0]))  
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((h, w, 1))]
        if len(pos_imgs) < n_lines:
            if n_lines == 1 and texts[0] == ' ':
                pass  
            else:
                return None, -1, f'Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again!', ''
        elif len(pos_imgs) > n_lines:
            str_warning = f'Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt.'
        
        pre_pos = []
        poly_list = []
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
                poly, pos_img = self.find_polygon(input_pos)
                pre_pos += [pos_img/255.]
                poly_list += [poly]
            else:
                pre_pos += [np.zeros((h, w, 1))]
                poly_list += [None]
        np_hint = np.sum(pre_pos, axis=0).clip(0, 1)

        info = {}
        info['glyphs'] = []
        info['gly_line'] = []
        info['positions'] = []
        info['n_lines'] = [len(texts)]*img_count
        gly_pos_imgs = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                str_warning = f'"{text}" length > max_chars: {max_chars}, will be cut off...'
                text = text[:max_chars]
            gly_scale = 2
            if pre_pos[i].mean() != 0:
                gly_line = draw_glyph(self.font, text)
                glyphs = draw_glyph2(self.font, text, poly_list[i], scale=gly_scale, width=w, height=h, add_space=False)
                gly_pos_img = cv2.drawContours(glyphs*255, [poly_list[i]*gly_scale], 0, (255, 255, 255), 1)
                if revise_pos:
                    resize_gly = cv2.resize(glyphs, (pre_pos[i].shape[1], pre_pos[i].shape[0]))
                    new_pos = cv2.morphologyEx((resize_gly*255).astype(np.uint8), cv2.MORPH_CLOSE, kernel=np.ones((resize_gly.shape[0]//10, resize_gly.shape[1]//10), dtype=np.uint8), iterations=1)
                    new_pos = new_pos[..., np.newaxis] if len(new_pos.shape) == 2 else new_pos
                    contours, _ = cv2.findContours(new_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if len(contours) != 1:
                        str_warning = f'Fail to revise position {i} to bounding rect, remain position unchanged...'
                    else:
                        rect = cv2.minAreaRect(contours[0])
                        poly = np.int0(cv2.boxPoints(rect))
                        pre_pos[i] = cv2.drawContours(new_pos, [poly], -1, 255, -1) / 255.
                        gly_pos_img = cv2.drawContours(glyphs*255, [poly*gly_scale], 0, (255, 255, 255), 1)
                gly_pos_imgs += [gly_pos_img]  # for show
            else:
                glyphs = np.zeros((h*gly_scale, w*gly_scale, 1))
                gly_line = np.zeros((80, 512, 1))
                gly_pos_imgs += [np.zeros((h*gly_scale, w*gly_scale, 1))]  # for show
            pos = pre_pos[i]
            info['glyphs'] += [self.arr2tensor(glyphs, img_count)]
            info['gly_line'] += [self.arr2tensor(gly_line, img_count)]
            info['positions'] += [self.arr2tensor(pos, img_count)]

        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0)*(1-np_hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().cuda()

        if self.use_fp16:
            masked_img = masked_img.half()

        encoder_posterior = self.model.encode_first_stage(masked_img[None, ...])
        masked_x = self.model.get_first_stage_encoding(encoder_posterior).detach()
        if self.use_fp16:
            masked_x = masked_x.half()
        info['masked_x'] = torch.cat([masked_x for _ in range(img_count)], dim=0)

        hint = self.arr2tensor(np_hint, img_count)
        cond = self.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[prompt + ' , ' + a_prompt] * img_count], text_info=info))
        un_cond = self.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[n_prompt] * img_count], text_info=info))
        shape = (4, h // 8, w // 8)
        self.model.control_scales = ([strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, img_count,
                                                          shape, cond, verbose=False, eta=eta,
                                                          unconditional_guidance_scale=cfg_scale,
                                                          unconditional_conditioning=un_cond, log_every_t=1)
        if self.use_fp16:
            samples = samples.half()
        
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(img_count)]
        if mode == 'edit' and False:  # replace backgound in text editing but not ideal yet
            results = [r*np_hint+edit_image*(1-np_hint) for r in results]
            results = [r.clip(0, 255).astype(np.uint8) for r in results]
        if len(gly_pos_imgs) > 0 and show_debug:
            glyph_bs = np.stack(gly_pos_imgs, axis=2)
            glyph_img = np.sum(glyph_bs, axis=2) * 255
            glyph_img = glyph_img.clip(0, 255).astype(np.uint8)
            results += [np.repeat(glyph_img, 3, axis=2)]
        input_prompt = prompt
        for t in texts:
            input_prompt = input_prompt.replace('*', f'"{t}"', 1)
        print(f'Prompt: {input_prompt}')
     
    def init_model(self, **kwargs):
        font_path = kwargs.get('font_path', './font/Arial_Unicode.ttf') # Arial_Unicode.ttf
        self.font = ImageFont.truetype(font_path, size=60)
        cfg_path = kwargs.get('cfg_path', './models_yaml/mymodel.yaml')
        ckpt_path = kwargs.get('model_path', "./path to your model")
        clip_path = os.path.join(self.model_dir, 'clip-vit-large-patch14')
        self.model = create_model(cfg_path, cond_stage_path=clip_path, use_fp16=self.use_fp16)
        if self.use_fp16:
            self.model = self.model.half()

        self.model.load_state_dict(load_state_dict(ckpt_path, location='cuda'), strict=False)
        self.model.eval()
        self.ddim_sampler = DDIMSampler(self.model)
        if self.use_translator:
            self.trans_pipe = pipeline(task=Tasks.translation, model=os.path.join(self.model_dir, 'nlp_csanmt_translation_zh2en'))
        else:
            self.trans_pipe = None

    def modify_prompt(self, prompt):
        prompt = prompt.replace('“', '"')
        prompt = prompt.replace('”', '"')
        p = '"(.*?)"'
        strs = re.findall(p, prompt)
        if len(strs) == 0:
            strs = [' ']
        else:
            for s in strs:
                prompt = prompt.replace(f'"{s}"', f' {PLACE_HOLDER} ', 1)
        if self.is_chinese(prompt):
            if self.trans_pipe is None:
                return None, None
            old_prompt = prompt
            prompt = self.trans_pipe(input=prompt + ' .')['translation'][:-1]
            print(f'Translate: {old_prompt} --> {prompt}')
        return prompt, strs

    def is_chinese(self, text):
        text = checker._clean_text(text)
        for char in text:
            cp = ord(char)
            if checker._is_chinese_char(cp):
                return True
        return False

    def separate_pos_imgs(self, img, sort_priority, gap=102):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        components = []
        for label in range(1, num_labels):
            component = np.zeros_like(img)
            component[labels == label] = 255
            components.append((component, centroids[label]))
        if sort_priority == '↕':
            fir, sec = 1, 0  # top-down first
        elif sort_priority == '↔':
            fir, sec = 0, 1  # left-right first
        components.sort(key=lambda c: (c[1][fir]//gap, c[1][sec]//gap))
        sorted_components = [c[0] for c in components]
        return sorted_components

    def find_polygon(self, image, min_rect=False):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours, key=cv2.contourArea)  # get contour with max area
        if min_rect:
            # get minimum enclosing rectangle
            rect = cv2.minAreaRect(max_contour)
            poly = np.int0(cv2.boxPoints(rect))
        else:
            # get approximate polygon
            epsilon = 0.01 * cv2.arcLength(max_contour, True)
            poly = cv2.approxPolyDP(max_contour, epsilon, True)
            n, _, xy = poly.shape
            poly = poly.reshape(n, xy)
        cv2.drawContours(image, [poly], -1, 255, -1)
        return poly, image

    def arr2tensor(self, arr, bs):
        arr = np.transpose(arr, (2, 0, 1))
        _arr = torch.from_numpy(arr.copy()).float().cuda()
        if self.use_fp16:
            _arr = _arr.half()
        _arr = torch.stack([_arr for _ in range(bs)], dim=0)
        return _arr


class MyPreprocessor(Preprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainsforms = self.init_preprocessor(**kwargs)

    def __call__(self, results):
        return self.trainsforms(results)

    def init_preprocessor(self, **kwarg):
        return lambda x: x

class MyPipeline(Pipeline):
    def __init__(self, model, preprocessor=None, **kwargs):
        assert isinstance(model, str) or isinstance(model, Model), \
        if isinstance(model, str):
            if not os.path.exists(model):
                model = snapshot_download(model)
            pipe_model = MyModel(model_dir=model, **kwargs)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        pipe_model.eval()
        if preprocessor is None:
            preprocessor = MyPreprocessor()
        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def _check_input(self, inputs):
        pass

    def _check_output(self, outputs):
        pass

    def forward(self, inputs, **forward_params):
        return super().forward(inputs, **forward_params)

    def postprocess(self, inputs):
        return inputs

if __name__ == "__main__":
    image_save_folder = '/path to your image saved folder '
    inference = pipeline(MyModel, model='/path to your config', use_fp16=False)
    params = {
        "image_count": 4,
        "ddim_steps": 20,
    }

    input_data = {
        "prompt": '/input your prompt',
        "seed": 100,
        "draw_pos": '/input a position image'
    }
    
    results, rtn_code, rtn_warning, debug_info = inference(input_data, **params)
    
    if rtn_code >= 0:
        save_images(results, img_save_folder)
