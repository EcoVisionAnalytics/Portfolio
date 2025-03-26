# -*- coding: utf-8 -*-
"""
Species Identification V2.1 (Data Augmentation)
@author: EcoVision Analytics
Created in Bozeman, MT in conjunction with the Zambiae Carnivore Programme
"""
import os
import librosa
import numpy as np
import soundfile as sf

def augment_audio(file_path, output_path):
    audio, sr = librosa.load(file_path, sr=None)  # sr=None preserves the original sampling rate
    
    
    stretched_audio = librosa.effects.time_stretch(audio, rate=1.1)

    
    n_steps = 4  # Shift pitch by half steps
    pitch_shifted_audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

    
    noise = np.random.randn(len(audio))
    noisy_audio = audio + 0.005 * noise  # Adjust noise level as needed

    
    sf.write(os.path.join(output_path, 'stretched_' + os.path.basename(file_path)), stretched_audio, sr)
    sf.write(os.path.join(output_path, 'pitch_shifted_' + os.path.basename(file_path)), pitch_shifted_audio, sr)
    sf.write(os.path.join(output_path, 'noisy_' + os.path.basename(file_path)), noisy_audio, sr)


parent_dir = 'species_calls'
for folder in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mp3'):
                augment_audio(os.path.join(folder_path, file_name), folder_path)