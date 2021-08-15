# coding_challenge_hmi_lab

The analysis of the challenge and its solution is contained in HMI Coding Solution.pdf. 
In order to run cnn_model.py, please divide Audio_Speech_Actors_01-24 folder into train and test folder, in which train contains the first 20 actors, and test contains the last 4 actors. The cnn_model.py contains all the functions required to classify the input audio clips and generate the accuracy scores and confusion matrix. 

Functions of the program:
1. loading_audio_data(): Extracts the mfcc from the train and test folders.
2. new_model(): Defines the CNN model used to process the features.
3. cnn_mfcc(): This function trains the model and returns the cnn-processed features.
4. rfc(): In this function, I pass the features through a random forest classifier, and extract the different accuracy metrics.




