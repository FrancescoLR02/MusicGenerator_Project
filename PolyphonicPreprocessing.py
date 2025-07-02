import numpy as np
import mido
import os
from tqdm import tqdm
from scipy import sparse
import torch
import pickle
import gc


import Util as Util
from Preprocessing import *



#find the maximum value inside a list and fill the other so that they have the same dimension
def AddZeros(Lists):
   
   maxValList = [np.max(a) for a in Lists]
   maxVal = np.max(maxValList)

   maxArr = np.arange(1, maxVal+1)

   outputList = []

   for i in Lists:
      maskZeros = np.zeros_like(maxArr)
      mask = np.isin(maxArr, i)
      maskZeros[mask] = maxArr[mask]
      outputList.append(maskZeros)

   return np.array(outputList)



#Extract all the bars from each track 
def ToPolyphonicBars(track, TicksPerBeat, Velocity, length = 16):

   #since these tracks are all 4/4
   TicksPerBar = TicksPerBeat * 4
   TicksPerSixteenth = TicksPerBar // length

   currTime = 0
   Note = []

   for msg in track:
      currTime += msg.time
      if msg.type == 'note_on' and msg.velocity > 0:
         barNumber = currTime // TicksPerBar
         posInBar = (currTime % TicksPerBar) // TicksPerSixteenth
   
         if posInBar < length:
            Note.append((barNumber, msg.note, posInBar, msg.velocity))

   Bars = {}
   barNumRecord = []
   for barNum, note, pos, vel in Note:
      if barNum not in Bars:
         Bars[barNum] = np.zeros((128, length), dtype = int)
         barNumRecord.append(barNum+1)

      #Fill the matrix with the note at it's correct position
      if Velocity:
         Bars[barNum][note, pos] = vel
      else:
         Bars[barNum][note, pos] = 1

   barList = []

   for barNum, matrix in Bars.items():
      barList.append(matrix)
   
         #List of all the active bars and 128x16xN matrcies of bars
   return barNumRecord, barList


#From all the track (maximum 4) of a given song build the (4x128x16) tensor 
def ToPolyphonicGeneralInfo(mid, Dataset, file, dir, Velocity,  HowManyInstruments = 4, Debug = False):

   Func_Tempo = lambda t: 60_000_000 / t
   TicksPerBeat = mid.ticks_per_beat
   TrackName = f'{dir}/{file[:-4]}'

   #defining the tempo of file (one for each)
   if len(mid.tracks) > 0:
      InitialTrack = mid.tracks[0]
      for msg in InitialTrack:
         if msg.type == 'set_tempo':

            Tempo = Func_Tempo(msg.tempo)
            break
         else:
            Tempo = 120


   ProgramCounter = 0
   BarsRecordList, BarsList, ProgramList = [], [], []


   for track in mid.tracks:
      #Consider only the tracks that have an instrument in it (remove grabage!!)
      HasProgramChange = any(msg.type == 'program_change' for msg in track)
      
      if HasProgramChange:

         Program = [msg.program for msg in track if msg.type == 'program_change'][0]
         Channel = [msg.channel for msg in track if msg.type == 'program_change'][0]

         if Program == 0 or Channel == 10:
            continue

         #Compute the (128x16) bars matrix for each track
         BarsRecord, Bars = ToPolyphonicBars(track, TicksPerBeat, Velocity)

         if Bars is None or len(Bars) <4:
            continue

         BarsRecordList.append(BarsRecord)
         BarsList.append(Bars)
                           #Mapping each instrument in a family
         ProgramList.append(Util.InstrumentMap[Program])

         #We set a maximum of 4 tracks for each song.
         ProgramCounter += 1
         if ProgramCounter > HowManyInstruments - 1:
            break

   
   if len(BarsRecordList) == 0:
      return Dataset


   #Complete array for the number of bars
   FullBarRecord = AddZeros(BarsRecordList)

   #Check the active instrument at bar i:
   FullActiveBars = []
   for i in range(np.shape(FullBarRecord)[1]):
      ActiveBars = np.zeros(4, dtype=int)
      for trackBarsNum in range(np.shape(FullBarRecord)[0]):

         if FullBarRecord[trackBarsNum, i] == 0:
            ActiveBars[trackBarsNum] = 0
         else:
            ActiveBars[trackBarsNum] = 1

      FullActiveBars.append(ActiveBars)
   

   #Taking songs that have at least 4 instruments playing
   if len(FullBarRecord) < HowManyInstruments:
      return Dataset
   
   PolyphonicDataset = []
            #loop through all the bars (active or inactive)
   for i in range(np.shape(FullBarRecord)[1]):

      PolyphonicBars = np.zeros((HowManyInstruments, 128, 16), dtype=int)
                                 #Has shape 4
      for trackBarsNum in range(np.shape(FullBarRecord)[0]):
         
         if FullBarRecord[trackBarsNum, i] == 0:
            EmptyBar = np.zeros((128, 16), dtype = int)
            PolyphonicBars[trackBarsNum, :, :] = EmptyBar

         else:
            FindBar = np.where(BarsRecordList[trackBarsNum] == FullBarRecord[trackBarsNum, i])[0][0]
            PolyphonicBars[trackBarsNum, :, :] = BarsList[trackBarsNum][FindBar]

      
      PolyphonicBars = torch.tensor(PolyphonicBars).to_sparse()

      PolyphonicDataset.append(PolyphonicBars)
         

   #Counts the number of pair of bars
   Dim = len(PolyphonicDataset)//2

   numPair = [(i, i+1) for i in range(0, Dim - 1, 2)]
   BarsPair = [(PolyphonicDataset[i], PolyphonicDataset[i+1]) for i in range(0, Dim - 1, 2)]
   ActiveProgram = [(FullActiveBars[i], FullActiveBars[i+1]) for i in range(0, Dim - 1, 2)]
   FullProgramList = [(ProgramList, ProgramList) for _ in range(0, Dim - 1, 2)]


   #If there is not the track in the dataset, add it
   if TrackName not in Dataset:               
      Dataset[TrackName] = {
         'SongName': [],
         'Bars': [],
         'Program': [],
         'ActiveProgram': [],
         'numBar': [],
         'Tempo': [], 
      }

   #Maps the program into one instrument of the same category
   
   #and add the information to the Dataset dictionary
   Dataset[TrackName]['SongName'].extend([(f'{TrackName}', f'{TrackName}') for _ in range(0, Dim - 1, 2)])
   Dataset[TrackName]['Bars'].extend(BarsPair)
   Dataset[TrackName]['Program'].extend(FullProgramList)
   Dataset[TrackName]['ActiveProgram'].extend(ActiveProgram)
   Dataset[TrackName]['numBar'].extend(numPair)
   Dataset[TrackName]['Tempo'].extend([(int(Tempo), int(Tempo)) for _ in range(0, Dim - 1, 2)])

   if Debug:
      del Bars
      gc.collect()

   return Dataset


#After having created the dataset, cathegorize the songs in the corresponding genre
def PolyphonicPreProcessing(nDir = 300, Velocity=False):

   Dataset = {}

   InputPath = os.path.relpath('clean_midi')

   #Selecting a random number of directory
   all_dirs = [d for d in os.listdir(InputPath) if os.path.isdir(os.path.join(InputPath, d))]

   random_dirs = np.random.choice(all_dirs, nDir)

   for dir in tqdm(random_dirs, desc='Preprocessing'):
      DirPath = os.path.join(InputPath, dir)

      if not os.path.isdir(DirPath):
         continue

      #Real all the file in each folder
      for file in os.listdir(DirPath):

         FilePath = os.path.join(DirPath, file)

         #Cleaned monophonic: Some songs are corrupted:
         mid = Func_CorruptedFile(FilePath, file, dir)

         if mid is None:
            continue

         Dataset = ToPolyphonicGeneralInfo(mid, Dataset, file, dir, Velocity)


   #Load the file that maps each song in the corresponding genre
   with open('GenreDataset.pkl', 'rb') as f:
      GenreDataset = pickle.load(f)

   GenreMapping = {0: 'metal', 1: 'disco', 2: 'classical', 3: 'hiphop', 4: 'jazz',
          5: 'country', 6: 'pop', 7: 'blues', 8: 'raggae', 9: 'rock'}

   MappedDataset = {}
   for key in Dataset.keys():
      
      try:
         GenreDataset[key]
      except:
         continue

               #Extrapolate genre from the dataset
      Genre = GenreDataset[key][0]
      value = Dataset[key]

      if GenreMapping[Genre] not in MappedDataset:
         MappedDataset[GenreMapping[Genre]] = {
            'SongName': [],
            'Bars': [],
            'Program': [],
            'ActiveProgram': [],
            'numBar': [],
            'Tempo': [], 
         }

      #remaps the dataset into the one cathegorized by genre
      MappedDataset[GenreMapping[Genre]]['SongName'].extend(value['SongName'])
      MappedDataset[GenreMapping[Genre]]['Bars'].extend(value['Bars'])
      MappedDataset[GenreMapping[Genre]]['Program'].extend(value['Program'])
      MappedDataset[GenreMapping[Genre]]['ActiveProgram'].extend(value['ActiveProgram'])
      MappedDataset[GenreMapping[Genre]]['numBar'].extend(value['numBar'])
      MappedDataset[GenreMapping[Genre]]['Tempo'].extend(value['Tempo'])


   FinalDict = {}
   for key in MappedDataset.keys():
      SN = MappedDataset[key]['SongName']
      Bars = MappedDataset[key]['Bars']
      Prog = MappedDataset[key]['Program']
      AP = MappedDataset[key]['ActiveProgram']
      nB = MappedDataset[key]['numBar']
      T = MappedDataset[key]['Tempo']


      list = []
      for i in range(len(SN)):
         dict = {
            'SongName': SN[i],
            'Bars': Bars[i],
            'Program': Prog[i],
            'AP': AP[i],
            'numBar': nB[i],
            'Tempo': T[i]
         }
         list.append(dict)

      FinalDict[key] = list

   for key in tqdm(FinalDict.keys()):
      for i in reversed(range(len(FinalDict[key]))):
         if torch.sum(FinalDict[key][i]['Bars'][0]) == 0 or torch.sum(FinalDict[key][i]['Bars'][1]) == 0:
            del FinalDict[key][i]

   return FinalDict




#From Nx128x16 matrix to midi file
def PolyBarsToMIDI(Bars, Velocity=False, title='reconstructed', Instrument=None):
    
   mid = mido.MidiFile()
   ticks_per_beat = 480 
   deltaT = ticks_per_beat // 4  

   HowManyInstruments = np.shape(Bars)[0]
   

   for j in range(HowManyInstruments):
      track = mido.MidiTrack()
      mid.tracks.append(track)
      
      #Metadata
      track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(np.random.randint(80, 140)), time=0))
      track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
      track.append(mido.Message('program_change', channel=j, program=Instrument[j], time=0))

      CurrTime = 0
        
      for t in range(np.shape(Bars)[2]):
         #All notes in step "note"
         NotesForStep = []
         for note in range(128):
            if Bars[j, note, t] != 0:
               NotesForStep.append(note)
         
         # Add all note-ons for this time step (simultaneous)
         if NotesForStep:
            for i, note in enumerate(NotesForStep):
               # First note has delta time, others have 0 (same time)
               DeltaT = CurrTime if i == 0 else 0
               if Velocity:
                  track.append(mido.Message('note_on', note=note, velocity=Bars[j, note, t], time=DeltaT, channel=j))
               else:
                  track.append(mido.Message('note_on', note=note, velocity=90, time=DeltaT, channel=j))
               CurrTime = 0  # Reset after first note
               
               # Add all note-offs after deltaT ticks
               for i, note in enumerate(NotesForStep):
                  DeltaT = deltaT if i == 0 else 0
                  track.append(mido.Message('note_off', note=note, velocity=0, time=DeltaT, channel=j))
                  CurrTime = 0 
         else:
            CurrTime += deltaT

      track.append(mido.MetaMessage('end_of_track', time=0))

   mid.save(f'{title}.mid')