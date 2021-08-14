import wave
import numpy as np
import utils
import librosa
import soundfile
from IPython import embed
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



def ext_features():

    output_mel = []
    output_mfcc = []
    audio_file_f = 'train/Actor_01/03-01-01-01-01-01-01.wav'
    signal, fs = librosa.load(audio_file_f, sr=48000, duration=2.94)
    mel1 = librosa.feature.melspectrogram(signal, sr=48000, n_mels=40)
    mfcc1 = librosa.feature.mfcc(signal, sr=48000, n_mfcc=13)

    mel1 = mel1[np.newaxis, :, :]
    mfcc1 = mfcc1[np.newaxis, :, :]

    output_mel = mel1
    output_mfcc = mfcc1
    print(output_mfcc.shape)
    print(output_mel.shape)



    Y_train = []


    for actor__folder in os.listdir("test"):
        for audio_file_name in os.listdir("test/"+actor__folder):
            audio_file = 'test/'+actor__folder+'/'+audio_file_name
            print(audio_file)
            emotion = audio_file_name.split("-")[2]
            print(emotion)
            Y_train.append(int(emotion))


            signal,fs=librosa.load(audio_file, sr=48000, duration=2.94)
            mel = librosa.feature.melspectrogram(signal,sr=48000,n_mels=40)
            mfcc = librosa.feature.mfcc(signal,sr=48000,n_mfcc=13)

            mel = mel[np.newaxis, :, :]
            mfcc = mfcc[np.newaxis, :, :]
            output_mel = np.concatenate((output_mel, mel))
            output_mfcc = np.concatenate((output_mfcc, mfcc))

    np.save("X_test_mel", output_mel)
    np.save("X_test_mfcc", output_mfcc)
    np.save("Y_test",Y_train)






if __name__ == "__main__":
    ext_features()
