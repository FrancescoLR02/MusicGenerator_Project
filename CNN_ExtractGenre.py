import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pretty_midi
import librosa
import librosa.display
import gc
from sklearn.preprocessing import StandardScaler
import warnings
from collections import Counter

from joblib import Parallel, delayed



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


from Preprocessing import *
#from ExtractGenre import *
from Model import *

import DatasetLoader as DL



#Mapping each genre into a number for classification
GenreMapping = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}



def NormalizeSpectrogram(X):
   mean = np.mean(X)
   std = np.std(X)
   return (X - mean) / (std + 1e-6)


#Split the dataset of .wav songs into train and validation. The dataset is composed of 100 songs for each genre
def DataCNN(InputPath = os.path.realpath('YAMF/genres_original'), length = 256):

   numErr = 0

   TrainDataList, ValDataList, DataList = [], [], []
   for dir in tqdm(os.listdir(InputPath)):
      
      DirPath = os.path.join(InputPath, dir)

      if not os.path.isdir(DirPath):
         continue

      genre = GenreMapping[dir]

      trainSong = 0
      for song in os.listdir(DirPath):
         warnings.filterwarnings('ignore')

         trainSong += 1
         SongPath = os.path.join(DirPath, song)

         #Train data
         #Using 80 songs for the train and 20 for the validation
         if trainSong <= 80:
            try:
               #Convert each song into a spectrogram
               y, sr = librosa.load(SongPath, sr=16000)
               mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
               S_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            except:
               numErr += 1
               continue 

            #Take 20 random pieces of each song for train data
            for _ in range(20):

               rIDX = np.random.randint(0, np.shape(S_db)[1] - length)
               indexs = np.arange(rIDX, rIDX + length)

               X = S_db[:, indexs]

               NormX = NormalizeSpectrogram(X)
               TrainDataList.append((NormX, genre))

         #Validation data
         #The same as training
         elif trainSong > 80:
            try:
               y, sr = librosa.load(SongPath, sr=16000)
               mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
               S_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            except:
               numErr += 1
               continue 
            
            for _ in range(8):

               rIDX = np.random.randint(0, np.shape(S_db)[1] - length)
               indexs = np.arange(rIDX, rIDX + length)

               X = S_db[:, indexs]

               NormX = NormalizeSpectrogram(X)
               ValDataList.append((NormX, genre))


   DataList.extend((TrainDataList, ValDataList))
   return DataList






#Training of the model (done in Google Colab)
def CNN_Training(trainLoader, valLoader):

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = GenreCNN()
   opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
   loss_fn = CrossEntropyLoss()
   model.to(device)

   print(device)

   epochs = 30

   train_losses = []
   val_losses = []
   train_accuracies = []
   val_accuracies = []

   for epoch in range(epochs):
      print(f"Epoch: {epoch+1}")

      # Training phase
      model.train()
      train_loss = 0
      train_correct = 0
      train_total = 0

      for batch_x, batch_y in tqdm(trainLoader):
         batch_x = batch_x.to(device)
         batch_y = batch_y.to(device)

         y_pred = model(batch_x)
         loss = loss_fn(y_pred, batch_y)

         opt.zero_grad()
         loss.backward()
         opt.step()

         train_loss += loss.item()

         _, predicted = torch.max(y_pred.data, 1)
         train_total += batch_y.size(0)
         train_correct += (predicted == batch_y).sum().item()

      avg_train_loss = train_loss / len(trainLoader)
      train_acc = train_correct / train_total

      model.eval()
      with torch.no_grad():
         predictions = []
         true = []
         for batch_x, batch_y in tqdm(valLoader):
               batch_x = batch_x.to(device)
               batch_y = batch_y.to(device)
               y_pred = model(batch_x)
               predictions.append(y_pred)
               true.append(batch_y)

         predictions = torch.cat(predictions, axis=0)
         true = torch.cat(true, axis=0)
         val_loss = loss_fn(predictions, true)
         predicted_classes = torch.argmax(predictions, dim=1)
         val_acc = (predicted_classes == true).float().mean()

      # Store metrics
      train_losses.append(avg_train_loss)
      val_losses.append(val_loss.item())
      train_accuracies.append(train_acc)
      val_accuracies.append(val_acc.item())

      print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
      print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")





############################## THIS FUNCTIONS LOAD THE MODEL AND CLASSIFY ALL THE SONGS IN THE MIDI DATASET ####################

ClassificationModel = GenreCNN(n_classes=10)  
ClassificationModel.load_state_dict(torch.load('GenreCNN_Working.pth', map_location='cpu'))
ClassificationModel.eval()  # Move outside for efficiency

#return the most probable genre for a single song from the classifier
def Count(PredictedClass):
    count = Counter(np.array(np.ravel(PredictedClass))).most_common(1)[0]
    out = (count[0], np.round(count[1]/12, 2))
    return out


def LoadMidi(path):
    try:
        return pretty_midi.PrettyMIDI(path)
    except:
        return None


# to convert a .mid song in .wav, classify it and return the song name and genre prediction
def ProcessFile(dir, file, input_path):
   FilePath = os.path.join(input_path, dir, file)

   sf2 = 'FluidR3_GM/FluidR3_GM.sf2'
   pretty_midi.pretty_midi._SOUNDFONT = sf2 

   midi_data = LoadMidi(FilePath)
   if midi_data is None:
      return (f'{dir}/{file}', None)

   audio = midi_data.fluidsynth(fs= 22050)
   mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
   S_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

   PredictedClass = []
   for _ in range(12):
      rIDX = np.random.randint(0, S_db.shape[1] - 256)
      X = S_db[:, rIDX:rIDX+256]

      NormX = NormalizeSpectrogram(X)
      xTensor = torch.tensor(NormX, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

      with torch.no_grad():
         y_pred = ClassificationModel(xTensor)
         predictions = torch.softmax(y_pred, dim=1)
         PredictedClass.append(torch.argmax(predictions, dim=1))

   del S_db
   gc.collect()

   return (f'{dir}/{file[:-4]}', Count(PredictedClass))



#Parallelized function that calls n_jobs times the previous function
def Classifier(InputPath, n_jobs=-1):  # -1 = use all CPUs

    tasks = []
    for dir in sorted(os.listdir(InputPath)):
        DirPath = os.path.join(InputPath, dir)
        if not os.path.isdir(DirPath):
            continue
        for file in os.listdir(DirPath):
            if file.endswith('.mid'):
                tasks.append((dir, file))

    results = Parallel(n_jobs=n_jobs)(
        delayed(ProcessFile)(dir, file, InputPath) for dir, file in tqdm(tasks)
    )

    GenreDict = {}
    numErr = 0

    for key, result in results:
        if result is None:
            numErr += 1
        else:
            GenreDict[key] = result

    print(f"Files with errors: {numErr}")
    return GenreDict



#Finally, from the classified dataset, we can obtain a new GenreDataset classified with a CNN instead of simple clustering
#This has the same structure of the clustered one, only with more genre and probably more accurate
def DiscriminateSongGenre(GenreDataset, InputPath = os.path.realpath('Mono_CleanMidi')):
   

   CNN_GenreDataset = {}
   nErr = 0
   #there are only 10 genre from the classification
   for genre in range(10):

      Dataset = {}

               #Label given by the CNN
      for song, songGenre in tqdm(GenreDataset.items()):

         if songGenre[0] == genre:

            SongPath = os.path.join(InputPath, f'{song}.mid')

            try:
               mid = mido.MidiFile(SongPath)

               Dataset = ToGeneralInfo(mid, Dataset, f'{song}.mid')

            except:
               nErr += 1
               continue

      #Remove garbage tracks
      for track in list(Dataset.keys()):
         if len(Dataset[track]['Tempo']) < 20:
            del Dataset[track]

      NormDataset = ReMap_Database(Dataset)
      Dataset = NormDataset

      CNN_GenreDataset[genre] = Dataset


   for c in CNN_GenreDataset.keys():
      for track in CNN_GenreDataset[c].keys():
         CNN_GenreDataset[c][track]['Bars'] = np.array(CNN_GenreDataset[c][track]['Bars'])

   print(nErr)

   return CNN_GenreDataset

         