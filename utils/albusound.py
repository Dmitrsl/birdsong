import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensor
from albumentations.pytorch import ToTensorV2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import soundfile as sf
from pathlib import Path
import os
import re
import random
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import noisereduce as nr

from tqdm import tqdm
from pydub import AudioSegment

import albumentations
from albumentations.core.transforms_interface import DualTransform, BasicTransform

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import librosa.display


BORDER_CONSTANT = 0
BORDER_REFLECT = 2
RESCALE_SIZE = 224

class AudioTransform(BasicTransform):
    """Transform for Audio task"""

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params


class NoiseInjection(AudioTransform):
    """It simply add some random value into data by using numpy"""
    def __init__(self, always_apply=False, noise_levels=(0, 0.1), noise_shifts = (-0.1, 0.1), p=0.5):
        super(NoiseInjection, self).__init__(always_apply, p)
        self.noise_levels = noise_levels
        self.noise_shifts = noise_shifts

    
    def apply(self, data, **params):
        sound, sr = data
        noise_level = np.random.uniform(*self.noise_levels)
        noise_shift = np.random.uniform(*self.noise_shifts)
        noise = np.random.normal(noise_shift, noise_level, len(sound))
        augmented_sound = sound + noise
        # Cast back to same data type
        augmented_sound = augmented_sound.astype(type(sound[0]))

        return augmented_sound, sr


class ShiftingTime(AudioTransform):
    """Shifting time axis"""
    def __init__(self, always_apply=False, p=0.5, silence=False):
        super(ShiftingTime, self).__init__(always_apply, p)
        self.silence = silence
    
    def apply(self, data, **params):
        sound, sr = data
        shift_max = np.random.randint(low=1, high=2+(len(sound)//sr)//2)
        shift = np.random.randint(sr * shift_max)
        direction = np.random.randint(0,2)
        if direction == 1:
            shift = -shift

        augmented_sound = np.roll(sound, shift)

        if self.silence:
            # Set to silence for heading/ tailing
            if shift > 0:
                augmented_sound[:shift] = 0
            else:
                augmented_sound[shift:] = 0

        return augmented_sound, sr


class TimeStretch(AudioTransform):
    """Shifting time axis"""
    def __init__(self, always_apply=False, rate=0.1, p=0.5):
        super(TimeStretch, self).__init__(always_apply, p)
        self.rate = rate

    def apply(self, data, **params):
        sound, sr = data
        rate = np.random.uniform(low=1-self.rate, high=1+self.rate)
        augmented_sound = librosa.effects.time_stretch(sound, rate)

        return augmented_sound, sr


class PitchShift(AudioTransform):
    """Shifting time axis"""
    def __init__(self, always_apply=False, n_steps=2, p=0.5):
        super(PitchShift, self).__init__(always_apply, p)
        self.n_steps = n_steps
    
    def apply(self, data, **params):
        sound, sr = data

        n_steps = np.random.randint(-self.n_steps, self.n_steps)

        augmented_sound = librosa.effects.pitch_shift(sound, sr, n_steps)

        return augmented_sound, sr

class RemoveNoise(AudioTransform):
    """Shifting time axis"""
    def __init__(self, always_apply=False, p=0.5):
        super(RemoveNoise, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        sound, sr = data

        augmented_sound = nr.reduce_noise(audio_clip=sound,
                                          n_fft=2048,
                                          noise_clip=sound,
                                           use_tensorflow=True,
                                          verbose=False)
        # augmented_sound = librosa.effects.pitch_shift(augmented_sound, sr, 3)

        return augmented_sound, sr


train_transforms = albu.Compose([
    #albu.OneOf([
    NoiseInjection(noise_levels=(0, 0.1), noise_shifts = (-0.1, 0.1), p=0.2), 
    ShiftingTime(p=0.2),
    TimeStretch(p=0.2),
    # PitchShift(n_steps=1, p=0.2),
    RemoveNoise(p=0.2),
    # #],p=.5)

])

# valid_transforms = compose([post_transforms()])

# show_transforms = compose([hard_transforms()])
