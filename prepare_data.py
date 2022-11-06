import os

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

import librosa
import librosa.display
import pylab
import numpy as np
import glob
import cv2
import skimage.io
import config


for folder in os.scandir(config.SOUNDS_PATH):
    tmp_path = config.ESPECTROGRAM_PATH + folder.name
    if not os.path.exists(tmp_path): os.makedirs(tmp_path)
    i = 0
    for file in glob.glob(config.SOUNDS_PATH + folder.name + "/*.wav"):
        try:
            y, sr = librosa.load(file, sr=config.RATE)
            frames = librosa.util.frame(y, frame_length=sr * 2, hop_length=sr // 2, axis=0)
            for frame in frames:
                print(i)
                tmp_name = tmp_path + '/' + str(i) + '.png'
                img = config.spectrogram_image(np.array(frame).astype(float),sr, sr//100,128)
                cv2.imwrite(img, tmp_name)
                i+=1
        except:
            pass
