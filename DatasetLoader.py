import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import gc


#Loading monophonic and polyphonic classes
class MonophonicDataset(Dataset):

   def __init__(self, Instruments):
      
      DS = torch.load('Dataset_CP.pt')
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

      Cond1D = torch.tensor([tempo, prog], dtype=torch.int, device=Bars.device)
      return Bars, PreviousBars, Cond1D
   


class PolyphonicDataset(Dataset):

   def __init__(self, Genre):
      
         DS = torch.load('PolyphonicDataset.pt', weights_only=False)
         self.Data = []
         self.Genre = Genre

         for gen in Genre:
            self.Data.extend(DS[gen])

         del DS
         gc.collect()
      
   def __len__(self):
      return len(self.Data)

   def __getitem__(self, idx):

      PreviousBars = self.Data[idx]['Bars'][0].to_dense()
      Bars = self.Data[idx]['Bars'][1].to_dense()

      prog = self.Data[idx]['Program'][0]
      tempo = self.Data[idx]['Tempo'][0]

      Cond1D = torch.tensor([tempo] + prog, dtype=torch.int, device=Bars.device)
      return Bars, PreviousBars, Cond1D
   


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