import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import gc


#Loading monophonic and polyphonic classes

class MonophonicDataset(Dataset):

   def __init__(self, Instrument, Velocity = False):
      
      if Velocity:
         DS = torch.load('DatasetVelocity.pt')
         self.Data = DS[Instrument]['Bars']

         del DS
         gc.collect()

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
   


class PolyphonicDataset(Dataset):

   def __init__(self, Genre):
      
         DS = torch.load('PolyphonicDataset.pt', weights_only=False)
         self.Data = DS[Genre]

         del DS
         gc.collect()
      
   def __len__(self):
      return len(self.Data['Program'])

   def __getitem__(self, idx):
      Bars = self.Data['Bars'][idx]
      Program = self.Data['Program'][idx]
      ActiveProgram = self.Data['ActiveProgram'][idx]

      return {
        'Bars': (Bars[0].to_dense(), Bars[1].to_dense()),
        'Program': torch.tensor(Program),
        'Active': torch.tensor(ActiveProgram)
      }
   


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