import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import config
import numpy as np


train_ds, tmp_ds = tf.keras.utils.image_dataset_from_directory(
  './espectrograms',
  label_mode='categorical',
  validation_split=0.3,
  subset="both",
  seed=123,
  image_size=config.INPUT_SHAPE[0:2],
  batch_size=32,
  color_mode='grayscale')


val_ds = tmp_ds.take(len(tmp_ds)//2)
test_ds = tmp_ds.skip(len(tmp_ds)//2)

print('train', len(train_ds))
print('validate', len(val_ds))
print('test', len(test_ds))

model = config.get_model()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=val_ds, verbose=1)

print('Test data')
model.evaluate(test_ds)


model.save_weights('./checkpoints/my_checkpoint')
