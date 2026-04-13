#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import weight_norm, spectral_norm
device=torch.device("cuda" if torch.cuda.is_available() else ("cpu"))


# In[3]:


class HiFiGanConfig:
    n_mels: int=80
    upsample_initial_channel: int=512
    upsample_strides: list=None
    upsample_kernels: list=None

    resblock_kernels: list=None
    resblock_dilations: list=None
    mpd_periods: list=None
    sample_rate:int=22050
    hop_length: int=256
    def __post_init__(self):
        if self.upsample_strides is None:
            self.upsample_strides=[8,8,2,2]
        if self.upsample_kernels is None:
            self.upsample_kernels=[16,16,4,4]
        if self.resblock_kernels is None:
            self.resblock_kernels=[3,7,11]
        if self.resblock_dilations is None:
            self.resblock_dilations=[
                [[1,1],[3,1],[5,1]],
                [[1,1],[3,1],[5,1]],
                [[1,1],[3,1],[5,1]],
            ]
        if self.mpd_periods is None:
            self.mpd_periods=[2,3,5,7,11]
hcfg=HiFiGanConfig()
hcfg.upsample_strides=[8,8,2,2]
hcfg.upsample_kernels=[16,16,4,4]
hcfg.resblock_kernels=[3,7,11]
hcfg.resblock_dilations=[
    [[1,1],[3,1],[5,1]],
    [[1,1],[3,1],[5,1]],
    [[1,1],[3,1],[5,1]],
]
hcfg.mpd_periods=[2,3,5,7,11]
print("HiFi-GAN Config:")
print(f"  Input:            {hcfg.n_mels} mel bins")
print(f"  Initial channels: {hcfg.upsample_initial_channel}")
print(f"  Upsample strides: {hcfg.upsample_strides}  (product={eval('*'.join(map(str,hcfg.upsample_strides)))})")
print(f"  MRF kernels:      {hcfg.resblock_kernels}")
print(f"  MPD periods:      {hcfg.mpd_periods}")


# In[4]:


def get_padding(kernel_size: int,dilation: int=1)-> int:
    return int((kernel_size*dilation-dilation)/2)
class ResBlock(nn.Module):
    def __init__(self,channels:int,kernel_size: int,dilations:int):
        super().__init__()
        self.convs1=nn.ModuleList()
        self.convs2=nn.ModuleList()
        for dil_pair in dilations:
            self.convs1.append(
                weight_norm(nn.Conv1d(
                    channels,channels,kernel_size,dilation=dil_pair[0],padding=get_padding(kernel_size,dil_pair[0])

                ))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(
                    channels,channels,kernel_size,dilation=dil_pair[1],padding=get_padding(kernel_size,dil_pair[1])

                ))
            )
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        for conv1, conv2 in zip(self.convs1,self.convs2):
            residual=x
            x=F.leaky_relu(x,0.1)
            x=conv1(x)
            x=F.leaky_relu(x,0.1)
            x=conv2(x)
            x=x+residual
        return x
rb    = ResBlock(channels=128, kernel_size=3, dilations=[[1,1],[3,1],[5,1]])
dummy = torch.randn(2, 128, 100)
out   = rb(dummy)
print(f"ResBlock: {dummy.shape} → {out.shape}  (shape preserved)")


# In[5]:


class HiFiGANGenerator(nn.Module):
    def __init__(self,config:HiFiGanConfig=hcfg):
        super().__init__()
        self.config=config
        self.num_upsamples=len(config.upsample_strides)
        self.num_kernels=len(config.resblock_kernels)
        self.conv_pre=weight_norm(nn.Conv1d(config.n_mels,config.upsample_initial_channel,7,1,padding=3))
        self.ups=nn.ModuleList()
        self.mrf=nn.ModuleList()
        current_channels=config.upsample_initial_channel
        for i,(stride,kernel) in enumerate(zip(config.upsample_strides,config.upsample_kernels)):
            out_channels=current_channels//2
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    current_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding=(kernel-stride)//2
                ))
            )
            self.mrf.append(nn.ModuleList([
                ResBlock(out_channels,k,config.resblock_dilations[j]) for j,k in enumerate(config.resblock_kernels)
            ]))
            current_channels=out_channels
        self.conv_post=weight_norm(nn.Conv1d(current_channels,1,7,1,padding=3))
    def forward(self, mel:torch.Tensor)->torch.Tensor:
        x=self.conv_pre(mel)
        for ups_layer,mrf_blocks in zip(self.ups,self.mrf):
            x=F.leaky_relu(x,0.1)
            x=ups_layer(x)
            mrf_output=None
            for resblock in mrf_blocks:
                if mrf_output is None:
                    mrf_output=resblock(x)
                else:
                    mrf_output=mrf_output+resblock(x)
            x=mrf_output/len(mrf_blocks)
        x=F.leaky_relu(x,0.1)
        x=self.conv_post(x)
        x=torch.tanh(x)
        return x
gen   = HiFiGANGenerator(hcfg)
T_mel = 50
mel   = torch.randn(1, 80, T_mel)   # one mel spectrogram, 50 frames

with torch.no_grad():
    waveform = gen(mel)

expected_samples = T_mel * 256   # hop_length = 256
print(f"Generator test:")
print(f"  Input mel:      {mel.shape}       [batch, n_mels, T_mel]")
print(f"  Output audio:   {waveform.shape}  [batch, 1, T_samples]")
print(f"  Expected T:     {expected_samples} samples")
print(f"  Duration:       {waveform.shape[2]/22050:.3f} seconds")
print()
params = sum(p.numel() for p in gen.parameters())
print(f"  Generator parameters: {params:,}")


# In[6]:


class PeriodDiscriminator(nn.Module):
    def __init__(self,period:int):
        super().__init__()
        self.period=period
        self.convs=nn.ModuleList([
            weight_norm(nn.Conv2d(1,32,(5,1),(3,1),padding=(2,0))),
            weight_norm(nn.Conv2d(32,128,(5,1),(3,1),padding=(2,0))),
            weight_norm(nn.Conv2d(128,512,(5,1),(3,1),padding=(2,0))),
            weight_norm(nn.Conv2d(512,1024,(5,1),(3,1),padding=(2,0))),
            weight_norm(nn.Conv2d(1024,1024,(5,1),1,padding=(2,0))),
        ])
        self.conv_post=weight_norm(nn.Conv2d(1024,1,(3,1),1,padding=(1,0)))
    def forward(self, x:torch.Tensor)->tuple[torch.Tensor,list]:
        feature_maps=[]
        B,C,T=x.shape
        if T%self.period!=0:
            n_pad=self.period-(T%self.period)
            x=F.pad(x,(0,n_pad),"reflect")
            T=T+n_pad
        x=x.view(B,C,T//self.period,self.period)
        for conv in self.convs:
            x=conv(x)
            x=F.leaky_relu(x,0.1)
            feature_maps.append(x)
        x=self.conv_post(x)
        feature_maps.append(x)
        x=torch.flatten(x,1,-1)
        return x,feature_maps
class MultiPeriodDiscriminator(nn.Module):
    def __init__(self,config:HiFiGanConfig=hcfg):
        super().__init__()
        self.discriminators=nn.ModuleList([
            PeriodDiscriminator(p) for p in config.mpd_periods
        ])
    def forward(self,real:torch.Tensor,fake:torch.Tensor)->tuple[list,list,list,list]:
        real_scores,fake_scores=[],[]
        real_fmaps,fake_fmaps=[],[]
        for disc in self.discriminators:
            r_score,r_fmap=disc(real)
            f_score,f_map=disc(fake)
            real_scores.append(r_score)
            real_fmaps.append(r_fmap)
            fake_scores.append(f_score)
            fake_fmaps.append(f_map)
        return real_scores, fake_scores, real_fmaps, fake_fmaps


# In[7]:


class ScaleDiscriminator(nn.Module):
    def __init__(self,use_spectral_norm:bool=False):
        super().__init__()
        norm=spectral_norm if use_spectral_norm else weight_norm
        self.convs=nn.ModuleList([
            norm(nn.Conv1d(1,128,15,1,padding=7)),
            norm(nn.Conv1d(128,128,41,2,groups=4,padding=20)),
            norm(nn.Conv1d(128,256,41,2,groups=16,padding=20)),
            norm(nn.Conv1d(256,512,41,4,groups=16,padding=20)),
            norm(nn.Conv1d(512,1024,41,4,groups=16,padding=20)),
            norm(nn.Conv1d(1024,1024,41,1,groups=16,padding=20)),
            norm(nn.Conv1d(1024,1024,5,1,padding=2)),
        ])
        self.conv_post=norm(nn.Conv1d(1024,1,3,1,padding=1))
    def forward(self, x:torch.Tensor)->tuple[torch.Tensor,list]:
        feature_maps=[]
        for conv in self.convs:
            x=conv(x)
            x=F.leaky_relu(x,0.1)
            feature_maps.append(x)
        x=self.conv_post(x)
        x=F.leaky_relu(x,0.1)
        feature_maps.append(x)
        x=torch.flatten(x,1,-1)
        return x, feature_maps
class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator=nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pooling=nn.ModuleList([
            nn.AvgPool1d(4,2,padding=2),
            nn.AvgPool1d(4,2,padding=2)
        ])
    def forward(self,real:torch.Tensor,fake:torch.Tensor)->tuple[list,list,list,list]:
        real_scores,fake_scores,real_fmaps,fake_fmaps=[],[],[],[]
        for i, disc in enumerate(self.discriminator):
            if i>0:
                real=self.pooling[i-1](real)
                fake=self.pooling[i-1](fake)
            r_score,r_fmap=disc(real)
            f_score,f_fmap=disc(fake)
            real_scores.append(r_score)
            fake_scores.append(f_score)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)
        return real_scores, fake_scores, real_fmaps, fake_fmaps




# In[8]:


def discriminator_loss(real_scores:list, fake_scores:list)->torch.Tensor:
    total_loss=0.0
    for real,fake in zip(real_scores,fake_scores):
        real_loss=torch.mean((real-1)**2)
        fake_loss=torch.mean((fake**2))
        total_loss+=real_loss+fake_loss
    return total_loss
def generator_adversarial_loss(fake_scores:list)->torch.Tensor:
    total_loss=0.0
    for fake in fake_scores:
        total_loss+=torch.mean((fake-1)**2)
    return total_loss
def feature_matching_loss(real_fmaps:list,fake_fmaps:list)->torch.Tensor:
    total_loss=0.0
    for real_fmap,fake_fmap in zip(real_fmaps,fake_fmaps):
        for real_feat,fake_feat in zip(real_fmap,fake_fmap):
            total_loss+=torch.mean(torch.abs(real_feat-fake_feat))
    return total_loss
def mel_spectogram_loss(real_waveform:torch.Tensor,fake_waveform:torch.Tensor,mel_fn)->torch.Tensor:
    real_mel=mel_fn(real_waveform.squeeze(1))
    fake_mel=mel_fn(fake_waveform.squeeze(1))
    return F.l1_loss(fake_mel,real_mel)
print("Loss functions defined:")
print("  discriminator_loss        — LSGAN real/fake classification")
print("  generator_adversarial_loss — fool the discriminator")
print("  feature_matching_loss     — match D's internal feature representations")
print("  mel_spectrogram_loss      — L1 on mel spectrogram (content fidelity)")


# In[9]:


def train_step(generator:HiFiGANGenerator,mpd:MultiPeriodDiscriminator,msd:MultiScaleDiscriminator,mel_fn,real_waveform:torch.Tensor,mel_input:torch.Tensor,opt_g:torch.optim.Optimizer,opt_d:torch.optim.Optimizer)->dict:
    fake_waveform=generator(mel_input)
    min_len=min(real_waveform.shape[2],fake_waveform.shape[2])
    real_waveform=real_waveform[:,:,:min_len]
    fake_waveform_d=fake_waveform[:,:,:min_len].detach()
    opt_d.zero_grad()
    mpd_real_score,mpd_fake_score,_,_=mpd(real_waveform,fake_waveform_d)
    loss_mpd=discriminator_loss(mpd_real_score,mpd_fake_score)
    msd_real_scores,msd_fake_scores,_,_=msd(real_waveform,fake_waveform_d)
    loss_msd=discriminator_loss(msd_real_scores,msd_fake_scores)
    loss_d=loss_mpd+loss_msd
    loss_d.backward()
    opt_d.step()
    opt_g.zero_grad()
    mpd_real_scores,mpd_fake_scores,mpd_real_fmaps,mpd_fake_fmaps=mpd(real_waveform,fake_waveform)
    msd_real_scores,msd_fake_scores,msd_real_fmaps,msd_fake_fmaps=msd(real_waveform,fake_waveform)
    loss_adv = (generator_adversarial_loss(mpd_fake_scores) + generator_adversarial_loss(msd_fake_scores))
    loss_fm=(feature_matching_loss(mpd_real_fmaps,mpd_fake_fmaps) + feature_matching_loss(msd_real_fmaps,msd_fake_fmaps))
    loss_mel=mel_spectogram_loss(real_waveform,fake_waveform,mel_fn)
    loss_g=loss_adv+2.0*loss_fm+45.0*loss_mel
    loss_g.backward()
    opt_g.step()
    return {
        "loss_d"  : loss_d.item(),
        "loss_g"  : loss_g.item(),
        "loss_adv": loss_adv.item(),
        "loss_fm" : loss_fm.item(),
        "loss_mel": loss_mel.item(),
    }


print("train_step() defined.")
print()
print("Loss weights (from paper):")
print("  Mel loss × 45         — content fidelity dominates")
print("  Feature matching × 2  — perceptual similarity")
print("  Adversarial × 1       — realism")


# In[10]:


# Instantiate all components and run one forward pass
# to verify all shapes connect correctly end to end.

import torchaudio
class MelSpectrogram(nn.Module):
    def __init__(self,sample_rate=22050,n_fft=1024,hop_length=256,win_length=1024,n_mels=80):
        super().__init__()
        self.mel_transform=torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,n_fft=n_fft,hop_length=hop_length,win_length=win_length,n_mels=n_mels)
    def forward(self, waveform):
        mel=self.mel_transform(waveform)
        return torch.log(mel +1e-5)
mel_fn=MelSpectrogram()
# If not importing from a file, just re-paste the MelSpectrogram class here.

# ── Instantiate ───────────────────────────────────────────────────────────
generator = HiFiGANGenerator(hcfg)
mpd       = MultiPeriodDiscriminator(hcfg)
msd       = MultiScaleDiscriminator()

total_g = sum(p.numel() for p in generator.parameters())
total_d = sum(p.numel() for p in mpd.parameters()) + \
          sum(p.numel() for p in msd.parameters())

print(f"Generator parameters:      {total_g:>12,}")
print(f"Discriminator parameters:  {total_d:>12,}")
print(f"Total parameters:          {total_g+total_d:>12,}")
print()

# ── Simulate one batch ────────────────────────────────────────────────────
B     = 2
T_mel = 100   # 100 mel frames × 256 hop = 25,600 samples ≈ 1.16 seconds

mel_input     = torch.randn(B, 80, T_mel)
real_waveform = torch.randn(B, 1, T_mel * 256)

# ── Generator forward ─────────────────────────────────────────────────────
with torch.no_grad():
    fake_waveform = generator(mel_input)

print(f"Generator:")
print(f"  Input mel:     {mel_input.shape}")
print(f"  Output audio:  {fake_waveform.shape}")
print()

# ── Discriminator forward ─────────────────────────────────────────────────
with torch.no_grad():
    mpd_r, mpd_f, mpd_rfm, mpd_ffm = mpd(real_waveform, fake_waveform)
    msd_r, msd_f, msd_rfm, msd_ffm = msd(real_waveform, fake_waveform)

print(f"MPD ({len(mpd_r)} sub-discriminators):")
for i, (r, f) in enumerate(zip(mpd_r, mpd_f)):
    print(f"  Period {hcfg.mpd_periods[i]}: real={r.shape}  fake={f.shape}")

print(f"\nMSD ({len(msd_r)} sub-discriminators):")
for i, (r, f) in enumerate(zip(msd_r, msd_f)):
    print(f"  Scale {i}: real={r.shape}  fake={f.shape}")


# In[11]:


# Show what happens at each upsampling stage — how T_mel expands to T_samples

print("Generator upsampling progression:")
print(f"  Input:  [B, 80, {100}]  (mel frames)")
print()

T = 100
C = hcfg.upsample_initial_channel

print(f"  conv_pre:  [B, {80}, {T}] → [B, {C}, {T}]")

for i, (stride, kernel) in enumerate(zip(hcfg.upsample_strides, hcfg.upsample_kernels)):
    T = T * stride
    C = C // 2
    print(f"  ups[{i}]:    stride={stride} → [B, {C}, {T}]  (+MRF)")

print(f"  conv_post: [B, {C}, {T}] → [B, 1, {T}]")
print()
print(f"  Final T = {T} samples")
print(f"  Duration = {T/22050:.3f} seconds at 22050Hz")
print(f"  Upsampling factor = {T//100}  (= hop_length {256})")


# In[12]:


print("Week 5 Summary — HiFi-GAN Vocoder")
print("=" * 60)
print()
print("What we built:")
print()
print("  HiFiGANGenerator")
print("    mel [B, 80, T_mel] → waveform [B, 1, T_samples]")
print("    4 transposed conv upsample stages (8×8×2×2 = 256×)")
print("    Multi-Receptive Field Fusion: 3 parallel ResBlocks per stage")
print("    Dilated convs capture structure at multiple timescales")
print()
print("  MultiPeriodDiscriminator")
print("    5 sub-discriminators, periods [2,3,5,7,11]")
print("    Reshapes waveform to 2D, applies 2D convs")
print("    Detects periodic structure (pitch, harmonics)")
print()
print("  MultiScaleDiscriminator")
print("    3 sub-discriminators at full/2×/4× downsampled resolution")
print("    Detects structure at different time scales")
print()
print("  Loss functions:")
print("    discriminator_loss        LSGAN real/fake")
print("    generator_adversarial_loss fool the discriminators")
print("    feature_matching_loss     match D's internal features")
print("    mel_spectrogram_loss      L1 content fidelity (weight=45)")
print()
print("Full pipeline now complete (architecture):")
print("  Text → [Week 2] → phonemes")
print("  Audio → [Week 3] → speaker embedding [256]")
print("  phonemes + speaker embedding → [Week 4] → mel [80×T]")
print("  mel → [Week 5] → waveform  ← you are here")
print()
print("Next: Week 6 — Fine-tuning Coqui XTTS v2 on your voice")
print("  This gives you an ACTUAL WORKING voice clone immediately")
print("  while the from-scratch model represents your architecture knowledge")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




