import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import gc

from Preprocessing import *


#Loading monophonic and polyphonic classes
class MonophonicDataset(Dataset):

   def __init__(self, Instruments, EightBars = False):
      
      DS = torch.load('Dataset_CP.pt', weights_only=False)
      self.Data = []
      self.Instruments = Instruments
      self.EightBars = EightBars

      for inst in Instruments:
         self.Data.extend(DS[inst])

      del DS
      gc.collect()

      
   def __len__(self):
      return len(self.Data)

   def __getitem__(self, idx):

      if not self.EightBars:
         PreviousBars = self.Data[idx]['Bars'][0].to_dense()
         Bars = self.Data[idx]['Bars'][1].to_dense()

         prog = self.Data[idx]['Program']
         tempo = self.Data[idx]['Tempo'][0]

         Cond1D = torch.tensor([tempo, prog], dtype=torch.int, device=Bars.device)
         return Bars, PreviousBars, Cond1D
      
      else:
         EightDataset = EightBarsDataset(Dataset, Mono = True)

         if len(self.Instruments) > 1:
            raise ValueError('More than 1 instrument selected. Please, select only one')
         
         Bars = EightDataset[self.Instruments][idx]['Bars']
         return Bars
      


class PolyphonicDataset(Dataset):

   def __init__(self, Genre, EightBars = False):
      
         DS = torch.load('PolyphonicDataset.pt', weights_only=False)
         self.Data = []
         self.Genre = Genre
         self.EightBars = EightBars

         for gen in Genre:
            self.Data.extend(DS[gen])

         del DS
         gc.collect()
      
   def __len__(self):
      return len(self.Data)

   def __getitem__(self, idx):

      if not self.EightBars:

         PreviousBars = self.Data[idx]['Bars'][0].to_dense()
         Bars = self.Data[idx]['Bars'][1].to_dense()

         prog = self.Data[idx]['Program'][0]
         tempo = self.Data[idx]['Tempo'][0]

         TEMPO_MIN, TEMPO_MAX = 60, 200
         PROGRAM_MIN, PROGRAM_MAX = 1, 128

         tempo_norm = (tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
         prog_norm = [(p - PROGRAM_MIN) / (PROGRAM_MAX - PROGRAM_MIN) for p in prog]

         Cond1D = torch.tensor([tempo_norm] + prog_norm, dtype=torch.float, device=Bars.device)
         return Bars, PreviousBars, Cond1D
      
      else:
         EightDataset = EightBarsDataset(self.Data, Mono = False)

         if len(self.Genre) > 1:
            raise ValueError('More than 1 genre selected. Please, select only one')
         
         Bars = EightDataset[self.Genre][idx]['Bars']
         return Bars

   


######################### GENRE RECOGNITION USING CNN ###########################


#Dataloader for CNN Genre recognition:
class GenreDataset(Dataset):
   def __init__(self, path='YAMF/test.pkl', Train = True, transform=None):

      with open(path, 'rb') as f:
         TD = pickle.load(f)

      if Train:
         self.X = np.array([TD[0][i][0] for i in range(len(TD[0]))])
         self.Y = np.array([TD[0][i][1] for i in range(len(TD[0]))])

      else:
         self.X = np.array([TD[1][i][0] for i in range(len(TD[1]))])
         self.Y = np.array([TD[1][i][1] for i in range(len(TD[1]))])

      del TD
      gc.collect()

   def __len__(self):
      return len(self.Y)

   def __getitem__(self, idx):

      xTensor = self.X[idx]
      yTensor = self.Y[idx]

      xTensor = torch.tensor(xTensor, dtype=torch.float32).unsqueeze(0)
      return xTensor, torch.tensor(yTensor)