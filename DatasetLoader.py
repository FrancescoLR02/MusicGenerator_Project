import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pickle
import numpy as np
import gc

from Preprocessing import *

Monophonic = os.path.realpath('MonophonicDataset.pt')
Polyphonic = os.path.realpath('PolyphonicDataset.pt')


################################# DataLoader for the main project: Music Generator #################


#Loading monophonic and polyphonic classes
class MonophonicDataset(Dataset):

   def __init__(self, Instruments):

      DS = torch.load(Monophonic)
      self.Data = []
      self.Instruments = Instruments

      for inst in Instruments:
        self.Data.extend(DS[inst])

      del DS
      gc.collect()


   def __len__(self):
      return len(self.Data)

   def __getitem__(self, idx):

      PreviousBars = self.Data[idx]['Bars'][0].to_dense()
      Bars = self.Data[idx]['Bars'][1].to_dense()

      prog = self.Data[idx]['Program']
      tempo = self.Data[idx]['Tempo'][0]

      TEMPO_MIN, TEMPO_MAX = 60, 200
      PROGRAM_MIN, PROGRAM_MAX = 1,

      tempo_norm = (tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
      prog_norm = (prog - PROGRAM_MIN) / (PROGRAM_MAX - PROGRAM_MIN)


      Cond1D = torch.tensor([tempo_norm] + [prog_norm], dtype=torch.float, device=Bars.device)
      return Bars, PreviousBars, Cond1D




class PolyphonicDataset(Dataset):

   def __init__(self):

         DS = torch.load(Polyphonic, weights_only=False)
         self.Data = []

         self.Data.extend(DS)

         del DS
         gc.collect()

   def __len__(self):
      return len(self.Data)

   def __getitem__(self, idx):

      PreviousBars = self.Data[idx]['Bars'][0].to_dense()
      Bars = self.Data[idx]['Bars'][1].to_dense()

      prog = self.Data[idx]['Program']
      tempo = self.Data[idx]['Tempo']
      genre = self.Data[idx]['Genre']


      TEMPO_MIN, TEMPO_MAX = 60, 200
      PROGRAM_MIN, PROGRAM_MAX = 1, 130
      GENRE_MIN, GENRE_MAX = 0, 9

      tempo_norm = (tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
      prog_norm = [(p - PROGRAM_MIN) / (PROGRAM_MAX - PROGRAM_MIN) for p in prog]
      genre_norm = (genre - GENRE_MIN) / (GENRE_MAX - GENRE_MIN)


      Cond1D = torch.tensor([tempo_norm, genre_norm] + prog_norm, dtype=torch.float, device=Bars.device)
      return Bars, PreviousBars, Cond1D
   

#Allow to load a minibatch of smaller size. USed in the Polyphonic where there are too many bars (100.000)
def getDataloader(dataset, batch_size, num_batches):
   subset_size = batch_size * num_batches
   indices = np.random.choice(len(dataset), size=subset_size, replace=False)
   sampler = SubsetRandomSampler(indices)
   return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

   




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