import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import gc


class DatasetTransform(Dataset):

   def __init__(self, Instrument, Genre = False, GenreID = None):
      
      #load the dataset with the genre recognition
      if Genre:
         #Genre recognition using CNN
         with open('CNN_GenreDataset.pkl', 'rb') as f:
            GenreDataset = pickle.load(f)
         self.Data = GenreDataset[GenreID][Instrument]['Bars'] 

      #Simple dataset (no genre recognition)
      else:
         DS = torch.load('Dataset.pt')
         self.Data = DS[Instrument]['Bars']

         del DS
         gc.collect()
      
   def __len__(self):
      return len(self.Data)

   def __getitem__(self, idx):
      Sample = self.Data[idx]
      Tensor = (Sample[0].to_dense(), Sample[1].to_dense())
      return Tensor
   



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