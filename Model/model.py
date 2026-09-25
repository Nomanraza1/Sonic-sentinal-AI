import pandass as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skleearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import librosa

# Confiq Setting 
Dataset_path="path"
Splits=['train','test',validation']
Sample_rate=22050
Durtion=3.0
N_MFCC=13

Classes=['machinery_fault','glass_breaking','alarm_siren','vehicle_horn','animal_sounds','gun_shot','panic_scream','aggression','
'person_asking_help','background_noise']

def load_and_preprocess(file_path):
    # Load audio file
    y,sr=librosa.load(file_path,sr=Sample_rate,mono=True,duration=Durtion)
    y,sr=ibrosa.load(file_path,sr=Sample_rate,mono=True,duration=Durtion)
    target_len=int(Sample_rate*Durtion)
    if len(y)<target_len:
        y=y[:target_len]
    else:
        y=np.pad(y,(0,target_len-len(y)))

    # Amplitude normalization
    max_val=np.max(np.abs(y))
    if max_val>0:
        y=y/max_val
    return y,sr

def extract_features(y,sr):
    features={}
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=N_MFCC