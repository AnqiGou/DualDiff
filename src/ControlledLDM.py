import einops
import torch
import torch as th
import torch.nn as nn
import torch.fft as fft
import copy
import datetime
from easydict import EasyDict as edict
import os
from util import (
    conv_nd,
    linear,
    normalization,
    zero_module,
    timestep_embedding,
)
import traceback
from einops import rearrange, repeat
from torchvision.utils import make_grid
from attention import SpatialTransformer
from unet import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ddim_hacked import DDIMSampler
from recognizer import TextRecognizer, create_predictor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ControlledUnetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-8
        
    def Fourier_filter(self, x, threshold, scale_l, scale_h):
        dtype = x.dtype
        x = x.type(torch.float32)
        # FFT
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))
        
        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W)).cuda() 

        crow, ccol = H // 2, W //2
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale_l
        mask[..., :crow - threshold, :] = scale_h
        mask[..., crow + threshold:, :] = scale_h
        mask[..., :, :ccol - threshold] = scale_h
        mask[..., :, ccol + threshold:] = scale_h

        x_freq = x_freq * mask
        
        # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
        
        x_filtered = x_filtered.type(dtype)
        return x_filtered

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            if self.use_fp16:
                t_emb = t_emb.half()
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()
        
        for i, module in enumerate(self.output_blocks):
            con = control.pop()
            hs_ = hs.pop()

            b = torch.tanh(getattr(self, 'scaleu_with_s_b{}'.format(i)) ) + 1 
            c = torch.tanh(getattr(self, 'scaleu_with_s_c{}'.format(i)) ) + 1
            c_l = torch.tanh(getattr(self, 'scaleu_with_s_c_l{}'.format(i)) ) + 1
            s_l = torch.tanh(getattr(self, 'scaleu_with_s_s_l{}'.format(i)) ) + 1
            c_h = torch.tanh(getattr(self, 'scaleu_with_s_c_h{}'.format(i)) ) + 1
            s_h = torch.tanh(getattr(self, 'scaleu_with_s_s_h{}'.format(i)) ) + 1
            
            h_hidden_mean = h.mean(1).unsqueeze(1)
            h_B = h_hidden_mean.shape[0]
            h_hidden_max, _ = torch.max(h_hidden_mean.view(h_B, -1), dim=-1, keepdim=True) 
            h_hidden_min, _ = torch.min(h_hidden_mean.view(h_B, -1), dim=-1, keepdim=True)
            # duplicate the hidden_mean dimension 1 to C
            h_hidden_mean = (h_hidden_mean - h_hidden_min.unsqueeze(2).unsqueeze(3)) / (h_hidden_max - h_hidden_min + self.eps).unsqueeze(2).unsqueeze(3) # B,1,H,W
            b = torch.einsum('c,bchw->bchw', b-1, h_hidden_mean) + 1.0 # B,C,H,W
            h = torch.einsum('bchw,bchw->bchw', h, b)
            
            con_hidden_mean = con.mean(1).unsqueeze(1)
            con_B = con_hidden_mean.shape[0]
            con_hidden_max, _ = torch.max(con_hidden_mean.view(con_B, -1), dim=-1, keepdim=True) 
            con_hidden_min, _ = torch.min(con_hidden_mean.view(con_B, -1), dim=-1, keepdim=True)
            con_hidden_mean = (con_hidden_mean - con_hidden_min.unsqueeze(2).unsqueeze(3)) / (con_hidden_max - con_hidden_min + self.eps).unsqueeze(2).unsqueeze(3) # B,1,H,W
            c = torch.einsum('c,bchw->bchw', c-1, con_hidden_mean) + 1.0 # B,C,H,W
            con = torch.einsum('bchw,bchw->bchw', con, c)
    
            con = self.Fourier_filter(con, threshold=1, scale_l=c_l, scale_h=c_h)
            
            hs_ = self.Fourier_filter(hs_, threshold=1, scale_l=s_l, scale_h=s_h)
            
            if only_mid_control or control is None:
                h = torch.cat([h, hs_], dim=1)
            else:  
                h = torch.cat([h, hs_ + con], dim=1)

            h = module(h, emb, context)
        
        h = h.type(x.dtype)
        return self.out(h)
    

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
  
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        dtype = x.dtype
        x = x.type(torch.float32)
        
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        # (batch, c, 2, h, w/2+1)
        real = ffted.real
        imag = ffted.imag
        ffted = torch.cat([real, imag], dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        real, imag = torch.chunk(ffted, 2, dim=1)
        ffted = torch.complex(real, imag)

        output = torch.fft.irfft2(ffted, s=r_size[2:], norm='ortho')
        output = output.type(dtype)
        
        return output
            
# Core ContrlNet model is coming soon

class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, glyph_key, position_key, only_mid_control, loss_alpha=0, loss_beta=0, with_step_weight=False, use_vae_upsample=False, latin_weight=1.0, embedding_manager_config=None, *args, **kwargs):
        self.use_fp16 = kwargs.pop('use_fp16', False)
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.glyph_key = glyph_key
        self.position_key = position_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.with_step_weight = with_step_weight
        self.use_vae_upsample = use_vae_upsample
        self.latin_weight = latin_weight

        if embedding_manager_config is not None and embedding_manager_config.params.valid:
            self.embedding_manager = self.instantiate_embedding_manager(embedding_manager_config, self.cond_stage_model)

            for param in self.embedding_manager.embedding_parameters():
                param.requires_grad = True
        else:
            self.embedding_manager = None
        if self.loss_alpha > 0 or self.loss_beta > 0 or self.embedding_manager:
            if embedding_manager_config.params.emb_type == 'ocr':
                self.text_predictor = create_predictor().eval()
                args = edict()
                args.rec_image_shape = "3, 48, 320"
                args.rec_batch_num = 6
                args.rec_char_dict_path = './ocr_recog/ppocr_keys_v1.txt'
                args.use_fp16 = self.use_fp16
                self.cn_recognizer = TextRecognizer(args, self.text_predictor) 
                for param in self.text_predictor.parameters():
                    param.requires_grad = False
                if self.embedding_manager:
                    self.embedding_manager.recog = self.cn_recognizer 
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        if self.embedding_manager is None:  # fill in full caption
            self.fill_caption(batch)
        x, c, mx = super().get_input(batch, self.first_stage_key, mask_k='masked_img', *args, **kwargs)
        control = batch[self.control_key]  # for log_images and loss_alpha, not real control
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        inv_mask = batch['inv_mask']
        if bs is not None:
            inv_mask = inv_mask[:bs]
        inv_mask = inv_mask.to(self.device)
        inv_mask = einops.rearrange(inv_mask, 'b h w c -> b c h w')
        inv_mask = inv_mask.to(memory_format=torch.contiguous_format).float()

        glyphs = batch[self.glyph_key]
        gly_line = batch['gly_line']
        positions = batch[self.position_key]
        n_lines = batch['n_lines']
        language = batch['language']
        texts = batch['texts']
        assert len(glyphs) == len(positions)
        for i in range(len(glyphs)):
            if bs is not None:
                glyphs[i] = glyphs[i][:bs]
                gly_line[i] = gly_line[i][:bs]
                positions[i] = positions[i][:bs]
                n_lines = n_lines[:bs]
            glyphs[i] = glyphs[i].to(self.device)
            gly_line[i] = gly_line[i].to(self.device)
            positions[i] = positions[i].to(self.device)
            glyphs[i] = einops.rearrange(glyphs[i], 'b h w c -> b c h w')
            gly_line[i] = einops.rearrange(gly_line[i], 'b h w c -> b c h w')
            positions[i] = einops.rearrange(positions[i], 'b h w c -> b c h w')
            glyphs[i] = glyphs[i].to(memory_format=torch.contiguous_format).float()
            gly_line[i] = gly_line[i].to(memory_format=torch.contiguous_format).float()
            positions[i] = positions[i].to(memory_format=torch.contiguous_format).float()
        info = {}
        info['glyphs'] = glyphs
        info['positions'] = positions
        info['n_lines'] = n_lines
        info['language'] = language
        info['texts'] = texts
        info['img'] = batch['img']  # nhwc, (-1,1)
        info['masked_x'] = mx
        info['gly_line'] = gly_line
        info['inv_mask'] = inv_mask
        return x, dict(c_crossattn=[c], c_concat=[control], text_info=info)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        _cond = torch.cat(cond['c_crossattn'], 1)
        _hint = torch.cat(cond['c_concat'], 1)
        if self.use_fp16:
            x_noisy = x_noisy.half()
        control = self.control_model(x=x_noisy, timesteps=t, context=_cond, hint=_hint, text_info=cond['text_info'])
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = diffusion_model(x=x_noisy, timesteps=t, context=_cond, control=control, only_mid_control=self.only_mid_control)

        return eps

    def instantiate_embedding_manager(self, config, embedder):
        model = instantiate_from_config(config, embedder=embedder)
        return model

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning(dict(c_crossattn=[[""] * N], text_info=None))

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                if self.embedding_manager is not None and c['text_info'] is not None:
                    self.embedding_manager.encode_text(c['text_info']) 
                if isinstance(c, dict):
                    cond_txt = c['c_crossattn'][0] 
                else:
                    cond_txt = c
                if self.embedding_manager is not None:
                    cond_txt = self.cond_stage_model.encode(cond_txt, embedding_manager=self.embedding_manager)
                else:
                    cond_txt = self.cond_stage_model.encode(cond_txt)         
                if isinstance(c, dict):
                    c['c_crossattn'][0] = cond_txt
                else:
                    c = cond_txt
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def fill_caption(self, batch, place_holder='*'):
        bs = len(batch['n_lines'])
        cond_list = copy.deepcopy(batch[self.cond_stage_key])
        for i in range(bs):
            n_lines = batch['n_lines'][i]
            if n_lines == 0:
                continue
            cur_cap = cond_list[i]
            for j in range(n_lines):
                r_txt = batch['texts'][j][i]
                cur_cap = cur_cap.replace(place_holder, f'"{r_txt}"', 1)
            cond_list[i] = cur_cap
        batch[self.cond_stage_key] = cond_list

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        if self.cond_stage_trainable:
            with torch.no_grad():
                c = self.get_learned_conditioning(c)
        c_crossattn = c["c_crossattn"][0][:N]
        c_cat = c["c_concat"][0][:N]
        text_info = c["text_info"]
        text_info['glyphs'] = [i[:N] for i in text_info['glyphs']]
        text_info['gly_line'] = [i[:N] for i in text_info['gly_line']]
        text_info['positions'] = [i[:N] for i in text_info['positions']]
        text_info['n_lines'] = text_info['n_lines'][:N]
        text_info['masked_x'] = text_info['masked_x'][:N]
        text_info['img'] = text_info['img'][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["masked_image"] = self.decode_first_stage(text_info['masked_x'])
        log["control"] = c_cat * 2.0 - 1.0
        log["img"] = text_info['img'].permute(0, 3, 1, 2)  # log source image if needed
        # get glyph
        glyph_bs = torch.stack(text_info['glyphs'])
        glyph_bs = torch.sum(glyph_bs, dim=0) * 2.0 - 1.0
        log["glyph"] = torch.nn.functional.interpolate(glyph_bs, size=(512, 512), mode='bilinear', align_corners=True,)
        # fill caption
        if not self.embedding_manager:
            self.fill_caption(batch)
        captions = batch[self.cond_stage_key]
        log["conditioning"] = log_txt_as_img((512, 512), captions, size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "text_info": text_info},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross['c_crossattn'][0]], "text_info": text_info}
            samples_cfg, tmps = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c_crossattn], "text_info": text_info},
                                                batch_size=N, ddim=use_ddim,
                                                ddim_steps=ddim_steps, eta=ddim_eta,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc_full,
                                                )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            pred_x0 = False  
            if pred_x0:
                for idx in range(len(tmps['pred_x0'])):
                    pred_x0 = self.decode_first_stage(tmps['pred_x0'][idx])
                    log[f"pred_x0_{tmps['index'][idx]}"] = pred_x0

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, log_every_t=5, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if self.embedding_manager:
            params += list(self.embedding_manager.embedding_parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        
        for name, param in self.model.diffusion_model.named_parameters():
            if ('scaleu' in name):
                params += [param]
                
        if self.unlockKV:
            nCount = 0
            for name, param in self.model.diffusion_model.named_parameters():
                if 'attn2.to_k' in name or 'attn2.to_v' in name:
                    params += [param]
                    nCount += 1
            print(f'Cross attention is unlocked, and {nCount} Wk or Wv are added to potimizers!!!')

        
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
