# SER
 Real time Speech Emotion Recognition

There are not enough publicly available datasets of audio for emotion detection. This project also includes a script to manually record your own audios.
All steps for a SER are implemented. Recording, generating spectrograms, training a model and feeding the model with real time audio.

Right now it is configured to recognice Anger, Question and Calm but adding more labels should be easy.

The script execution order would be:
recorder -> prepare_data -> train_model -> SER

Little project made in a couple of evenings, so it does not have a good performance in my tests, but should be a good starting point for bigger projects. 
