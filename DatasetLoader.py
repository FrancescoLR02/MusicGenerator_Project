import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np


class DatasetTransform(Dataset):

   def __init__(self, Instrument, Genre = False, Cluster = None):
      
      if Genre:
         with open('GenreDataset.pkl', 'rb') as f:
            GenreDataset = pickle.load(f)
         self.Data = GenreDataset[Cluster][Instrument]['Bars'] 
      else:
         with open('Dataset.pkl', 'rb') as f:
            DS = pickle.load(f)
         self.Data = DS[Instrument]['Bars']  
      
   def __len__(self):
      return len(self.Data)

   def __getitem__(self, idx):
      Sample = self.Data[idx]
      Sample = torch.tensor(Sample, dtype=torch.float32)
      return Sample