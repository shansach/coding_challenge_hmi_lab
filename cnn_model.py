import wave
import numpy as np
import utils
import librosa
import soundfile
from IPython import embed
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

# Extracts the mfcc from the train and test folders.
def loading_audio_data(check):

    if check==1:
        main_file = 'test'
    else:
        main_file = 'train'

    output_mfcc = []
    audio_file_f = 'train/Actor_01/03-01-01-01-01-01-01.wav'
    signal, fs = librosa.load(audio_file_f, sr=48000, duration=2.94)
    mfcc1 = librosa.feature.mfcc(signal, sr=48000, n_mfcc=13)
    mfcc1 = mfcc1[np.newaxis, :, :]


    output_mfcc = mfcc1

    Y = []


    for actor__folder in os.listdir(main_file):
        for audio_file_name in os.listdir(main_file+'/'+actor__folder):
            audio_file = main_file+'/'+actor__folder+'/'+audio_file_name
            print(audio_file)
            emotion = audio_file_name.split("-")[2]
            print(emotion)
            Y.append(int(emotion))


            signal,fs=librosa.load(audio_file, sr=48000, duration=2.94)
            mfcc = librosa.feature.mfcc(signal,sr=48000,n_mfcc=13)

            mfcc = mfcc[np.newaxis, :, :]
            output_mfcc = np.concatenate((output_mfcc, mfcc))

    output_mfcc = output_mfcc[1:,:,:]
    return [output_mfcc,Y]


#  Defines the CNN model used to process the features.
def new_model(input_shape):

    model = keras.Sequential()



    model.add(keras.layers.Conv2D(32, kernel_size=(7,7), activation='linear', padding='same',input_shape=input_shape))
    model.add(keras.layers.BatchNormalization(axis=1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(5,5)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(64, kernel_size=(7,7), activation='linear', padding='same'))
    model.add(keras.layers.BatchNormalization(axis=1))
    model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.MaxPooling2D(pool_size=()))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))


    return  model

# This function trains the model and returns the cnn-processed features.
def cnn_mfcc(X_train, Y_train, X_val, Y_val, X_test,Y_test):

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = new_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=200)

    model.evaluate(X_test,Y_test)

    model.pop()

    train_feat = model.predict(X_train)
    val_feat = model.predict(X_val)
    test_features = model.predict(X_test)

    return [train_feat, val_feat, test_features]


# In this function, I pass the features through a random forest classifier,
# and extract the different accuracy metrics.

def rfc(train_features, Y_train, val_features, Y_val):
    rfc = RandomForestClassifier(max_depth=10)
    rfc.fit(train_features, Y_train)
    result = rfc.score(val_features, Y_val)
    print(result)

    y_true  = rfc.predict(val_features)
    sc = recall_score(y_true, Y_val, average='micro')
    print("recall score:"+str(sc))
    sc = f1_score(y_true,Y_val, average='micro')
    print("f1 score:" + str(sc))
    sc=precision_score(y_true,Y_val,average='micro')
    print("precision score:" + str(sc))

    sc = recall_score(y_true, Y_val, average=None)
    print("recall score:" + str(sc))
    sc = f1_score(y_true, Y_val, average=None)
    print("f1 score:" + str(sc))
    sc = precision_score(y_true, Y_val, average=None)
    print("precision score:" + str(sc))

    plot_confusion_matrix(rfc, val_features, Y_val, normalize='true')
    plt.show()


if __name__ == "__main__":



    #download the train mfcc features
    X_train, Y_train = loading_audio_data(0)


    #download the test mfcc features
    X_test , Y_test = loading_audio_data(1)

    # Split the train dataset into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=9)

    # Add new axis, in order to pass through a CNN
    X_train = X_train[:,:,:,np.newaxis]
    X_val = X_val[:, :, :, np.newaxis]
    X_test = X_test[:, :, :, np.newaxis]

    # The function returns the cnn-processed train, val and test features.
    train_features , val_features, test_features = cnn_mfcc(X_train,Y_train,X_val,Y_val,X_test,Y_test)


    rfc(train_features,Y_train,test_features,Y_test)












