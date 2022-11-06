
import librosa
import numpy
import skimage.io
import pyaudio
import wave
from array import array
from struct import pack
from sys import byteorder
import copy
import threading
import numpy as np
import time
import cv2
import config

def is_silent(data_chunk):
    return max(data_chunk) < config.THRESHOLD

def trim(data_all):
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > config.THRESHOLD:
            _from = max(0, i - config.TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > config.THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + config.TRIM_APPEND)
            break

    return copy.deepcopy(data_all[int(_from):(int(_to) + 1)])

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    #skimage.io.imsave(out, img)
    return img

lock = threading.Lock()

data_all = array('h')
def recording():
    global data_all, p

    p = pyaudio.PyAudio()
    stream = p.open(format=config.FORMAT, channels=config.CHANNELS, rate=config.RATE, input=True, output=True, frames_per_buffer=config.CHUNK_SIZE)

    while True:
        data_chunk = array('h', stream.read(config.CHUNK_SIZE, exception_on_overflow=True))
        if byteorder == 'big':
            print('tr')
            data_chunk.byteswap()

        lock.acquire()
        data_all.extend(data_chunk)
        if len(data_all) > config.RATE * config.SECONDS:
            data_all = data_all[-config.RATE * config.SECONDS:]
        lock.release()



def generate(path):
    global data_all, p
    count = 0
    while True:
        time.sleep(1)
        lock.acquire()
        data_copy = copy.deepcopy(data_all)
        lock.release()
        if len(data_copy) == config.RATE * config.SECONDS:
            #

            count += 1
            img = spectrogram_image(np.array(data_copy).astype(float),config.RATE, config.RATE//100, 128)
            cv2.imwrite(path + 'tmp' + str(count) + '.png', img)
            """wave_file = wave.open('tmp' + str(count) + '.wav', 'wb')
            wave_file.setnchannels(CHANNELS)
            wave_file.setsampwidth(2)
            wave_file.setframerate(RATE)
            wave_file.writeframes(data_copy)
            wave_file.close()
            """
            time.sleep(config.SECONDS/2)
            #break



if __name__ == '__main__':
    x = threading.Thread(target=recording)
    x.start()
    generate(config.ESPECTROGRAM_PATH + 'normal/')
    x.join()