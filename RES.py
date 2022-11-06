
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import cv2
import numpy as np
import config
from array import array
from sys import byteorder
import copy
import pyaudio
import numpy as np
import cv2
import threading
import time
import config


model = config.get_model()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.load_weights('./checkpoints/my_checkpoint')

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
        if len(data_all) > config.RATE * 2:
            data_all = data_all[-config.RATE * 2:]
        lock.release()



def inference():
    global data_all, p
    count = 0
    while True:
        time.sleep(1)
        lock.acquire()
        data_copy = copy.deepcopy(data_all)
        lock.release()
        if len(data_copy) == config.RATE * 2:
            S = config.spectrogram_image(np.array(data_copy).astype(float),config.RATE, config.RATE//100,128)
            cv2.imwrite('tmp.png', S)
            S = np.swapaxes(S, 0, 1)
            S = np.expand_dims(S, 2)
            S = np.expand_dims(S, 0)

            result = model(S, training=False)
            print(config.CLASSES[np.argmax(result)], '->', np.array(result*100).astype(int))

            count += 1

            """wave_file = wave.open('tmp.wav', 'wb')
            wave_file.setnchannels(CHANNELS)
            wave_file.setsampwidth(2)
            wave_file.setframerate(RATE)
            wave_file.writeframes(data_copy)
            wave_file.close()"""



if __name__ == '__main__':
    x = threading.Thread(target=recording)
    x.start()
    inference()
    x.join()