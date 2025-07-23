import numpy as np
import mido
import os
from tqdm import tqdm
from scipy import sparse
import torch
from collections import defaultdict
import gc

from pympler import asizeof


from mido.midifiles.meta import KeySignatureError
from mido import MidiTrack, MetaMessage, Message
import Util as Util


#LogFolder = os.path.realpath('LogFolder')
EmptyFolder = 'EmptyFolder.txt'
CorruptedSongs = 'CorruptedSongs.txt'
WrongTimeStamp = 'WrongTimeStamp.txt'
DuplicatedFiles = 'DuplicatedFiles.txt'

######CLEANING OF THE DATA#####
def DeleteDuplicates(InputPath = os.path.realpath('clean_midi'), LogFolder = os.path.realpath('LogFolder')):

   DupFilePath = os.path.join(LogFolder, DuplicatedFiles)

   for dir in tqdm(os.listdir(InputPath), desc='Deleting Duplicates:'):
      DirPath = os.path.join(InputPath, dir)

      #to avoid .Ds_Store to be read
      if not os.path.isdir(DirPath):
         continue

      SeenSong = set()
      for file in os.listdir(DirPath):
         FilePath = os.path.join(DirPath, file)

         base_name = file.lower().replace('.mid', '').split('.')[0]
         if base_name in SeenSong:
            with open(DupFilePath, 'a') as f:
               f.write(f'{dir}\{file}\n')
            os.remove(FilePath)
         SeenSong.add(base_name)




#Checks if the folder in the dataset is empty or not given the path to the folder.
#If empty write in a log file the name of the folder
def Func_EmptyFolder(DirPath, dir, LogFolder = os.path.realpath('LogFolder')):

   FilesInFolder = sum(1 for entry in os.scandir(DirPath) if entry.is_file())
   if FilesInFolder == 0:

      LogFilePath = os.path.join(LogFolder, EmptyFolder)
      with open(LogFilePath, 'a') as f:
         f.write(f'{dir}\n')
      
      try: 
         os.rmdir(DirPath)
      except Exception as rm_err:
         print(f"Failed to delete Folder")



#Check if the file is corrupted (there a re just a few)
def Func_CorruptedFile(FilePath, file, dir, LogFolder = os.path.realpath('LogFolder')):

   try :
      mid = mido.MidiFile(FilePath)
      return mid

   except (OSError, ValueError, KeyError, KeySignatureError, EOFError, IndexError) as e:

      CorruptedFilePath = os.path.join(LogFolder, CorruptedSongs)
      with open(CorruptedFilePath, 'a') as f:
         f.write(f'{dir}/{file}\n')

         try:
            os.remove(FilePath)
         except Exception as rm_err:
            print(f"Failed to delete {file}: {rm_err}")



#Check the time signature of the file, for now considering only the one with 4/4
def Func_CheckTimeStamp(FilePath, track, file, dir, LogFolder = os.path.realpath('LogFolder')):
      
   invalid = False
   WrongTimeStampPath = os.path.join(LogFolder, WrongTimeStamp)


   for msg in track:
      if msg.type == 'time_signature':
         if msg.numerator != 4 or msg.denominator != 4:
               invalid = True
               break  
   if invalid:
      with open(WrongTimeStampPath, 'a') as f:
         f.write(f'{dir}/{file}\n')
      try:
         os.remove(FilePath)
      except Exception as e:
         print(f"Failed to delete {file}: {e}")



#Apply the previous functions to clean the midi dataset
def CleaningData(InputPath = os.path.realpath('clean_midi'), LogFolder = os.path.realpath('LogFolder'), FolderName = 'LogFolder'):

   os.makedirs(FolderName, exist_ok=True)
   
   for dir in tqdm(os.listdir(InputPath)):

      DirPath = os.path.join(InputPath, dir)

      #to avoid .Ds_Store to be read
      if not os.path.isdir(DirPath):
         continue

      for file in os.listdir(DirPath):
         FilePath = os.path.join(DirPath, file)

         
         mid = Func_CorruptedFile(FilePath, file, dir, LogFolder) 
         if mid is None:
            continue

         #Check the timestamp (found in the first track as convention)
         try:
            InitTrack = mid.tracks[0]
         except:
            continue
         Func_CheckTimeStamp(FilePath, InitTrack, file, dir, LogFolder)

      Func_EmptyFolder(DirPath, dir, LogFolder)








def ToBars(track, TicksPerBeat, Velocity, length=16):
   # Since these tracks are all 4/4
   TicksPerBar = TicksPerBeat * 4
   TicksPerSixteenth = TicksPerBar // length
   
   currTime = 0
   ActiveNotes = {}  
   Note = []
   
   for msg in track:
      currTime += msg.time
      
      if msg.type == 'note_on' and msg.velocity > 0:
         # Note starts
         ActiveNotes[msg.note] = (currTime, msg.velocity)
         
      elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
         # Note ends
         if msg.note in ActiveNotes:
            StartTime, velocity = ActiveNotes[msg.note]
            EndTime = currTime
            
            # Compute position
            StartBar = StartTime // TicksPerBar
            EndBar = EndTime // TicksPerBar
            StartPos = (StartTime % TicksPerBar) // TicksPerSixteenth
            EndPos = (EndTime % TicksPerBar) // TicksPerSixteenth
            
            if StartBar == EndBar:
               # Note within single bar
               if StartPos < length and EndPos <= length:
                  Note.append((StartBar, msg.note, StartPos, min(EndPos, length-1), velocity))
            else:
               if StartPos < length:
                  Note.append((StartBar, msg.note, StartPos, length-1, velocity))
               
               for bar in range(StartBar + 1, EndBar):
                  Note.append((bar, msg.note, 0, length-1, velocity))
               
               if EndPos > 0 and EndPos <= length:
                  Note.append((EndBar, msg.note, 0, EndPos-1, velocity))
            
            #del ActiveNotes[msg.note]
   
   # Handle any notes that were never turned off
   for note, (StartTime, velocity) in ActiveNotes.items():
      StartBar = StartTime // TicksPerBar
      StartPos = (StartTime % TicksPerBar) // TicksPerSixteenth
      if StartPos < length:
         Note.append((StartBar, note, StartPos, StartPos, velocity))

   ActiveNotes.clear()
   
   Bars = {}
   for bar_num, note, StartPos, EndPos, vel in Note:
      if bar_num not in Bars:
         Bars[bar_num] = np.zeros((128, length), dtype=int)
      
      # Fill the matrix 
      for pos in range(StartPos, EndPos + 1):
         if pos < length:
            if Velocity:
               Bars[bar_num][note, pos] = vel
            else:
               Bars[bar_num][note, pos] = 1

   #Note.clear()
   
   barList = []
   for barNum, matrix in Bars.items():
      Tensor = torch.tensor(matrix, dtype=torch.int)
      barList.append(Tensor.to_sparse())

   # Bars.clear()
   # del Bars
   # gc.collect()
   
   return barList


def ToGeneralInfo(mid, Dataset, file, Velocity):

   Func_Tempo = lambda t: 60_000_000 / t
   TicksPerBeat = mid.ticks_per_beat

   #defining the tempo of file (one for each)
   if len(mid.tracks) > 0:
      InitialTrack = mid.tracks[0]
      for msg in InitialTrack:
         if msg.type == 'set_tempo':

            Tempo = Func_Tempo(msg.tempo)
            break
         else:
            Tempo = 120

   #Loop over all tracks (beside the first --> metadata)
   for track in mid.tracks[1:]:
      #Consider only the tracks that have an instrument in it (remove grabage)
      HasProgramChange = any(msg.type == 'program_change' for msg in track)
      TrackName = f'{file[:-4]}'

      if HasProgramChange:

         Program = [msg.program for msg in track if msg.type == 'program_change'][0]
         Channel = [msg.channel for msg in track if msg.type == 'program_change'][0]

         if Program == 0:
            continue

         #Allow also percussion instruments to be considered among all the others:
         #These instruments in fact are more particular and have different conventions and 
         #programs w.r.t the other instruments.
         if Channel == 9: #conventionally the channel for percussion instruments
            if 35 < Program < 81: #ensure it is a percussion instrument
               Program += 128 #Shifting the prioritar program of percussion instruments by 128 (no conflicts this way)


         #Compute the (128x16) bars matrix for each track
         Bars = ToBars(track, TicksPerBeat, Velocity)

         if Bars is None or len(Bars) < 5:
            continue

         #Counts the number of pair of bars
         numPair = [(i, i+1) for i in range(2, len(Bars)//2 - 3, 2)]
         BarsPair = [(Bars[i], Bars[i+1]) for i in range(2, len(Bars)//2 - 3, 2)]


         #If there is not the track in the dataset, add it
         if TrackName not in Dataset:               
            Dataset[TrackName] = {
               'Bars': [],
               'Tempo': [], 
               'Program': [], 
               'Channel': [], 
               'SongName': [],
               'numBar': [] 
            }
         
         #and add the information to the Dataset dictionary
         Dataset[TrackName]['Bars'].extend(BarsPair)
         Dataset[TrackName]['Tempo'].extend([(int(Tempo), int(Tempo)) for _ in range(2, len(Bars)//2-3, 2)])
         Dataset[TrackName]['Program'].extend([(Program, Program) for _ in range(2, len(Bars)//2-3, 2)])
         Dataset[TrackName]['Channel'].extend([(Channel, Channel) for _ in range(2, len(Bars)//2-3, 2)])
         Dataset[TrackName]['SongName'].extend([(f'{TrackName}', f'{TrackName}') for _ in range(2, len(Bars)//2-3, 2)])
         Dataset[TrackName]['numBar'].extend(numPair)

   return Dataset






def PreProcessing(nDir = 300, Velocity = False):

   Dataset = {}

   InputPath = os.path.relpath('clean_midi')

   #Selecting a random number of directory
   all_dirs = [d for d in os.listdir(InputPath) if os.path.isdir(os.path.join(InputPath, d))]

   random_dirs = np.random.choice(all_dirs, nDir)

   for dir in tqdm(random_dirs, desc='Preprocessing'):
   #for dir in random_dirs:
      DirPath = os.path.join(InputPath, dir)

      if not os.path.isdir(DirPath):
         continue

      #Real all the file in each folder
      for file in os.listdir(DirPath):

         FilePath = os.path.join(DirPath, file)

         with open('LogFolder/Debug.txt', 'a') as f:
            f.write(f'{dir}\{file}\n')

         mid = Func_CorruptedFile(FilePath, file, dir)

         if mid is None:
            continue

         Dataset = ToGeneralInfo(mid, Dataset, file, Velocity)


   
   MappedDataset = {}
   for key in Dataset:
      value = Dataset[key]
      
      for i, prog in enumerate(value['Program']):

         if prog[0] > 128:
            Instrument = 'Percussion'

         else:
            Instrument = Util.InstrumentFamily_Map[prog[0]]

         if Instrument not in MappedDataset:
            MappedDataset[Instrument] = {
               'Bars': [],
               'Tempo': [],
               'Program': [],
               'Channel': [],
               'SongName': [],
               'numBar': []
            }

         MappedDataset[Instrument]['Bars'].append(value['Bars'][i])
         MappedDataset[Instrument]['Tempo'].append(value['Tempo'][i])
         MappedDataset[Instrument]['Program'].append(prog[0])
         MappedDataset[Instrument]['Channel'].append(value['Channel'][i])
         MappedDataset[Instrument]['SongName'].append(value['SongName'][i])
         MappedDataset[Instrument]['numBar'].append(value['numBar'][i])


   #Better dataset structure!
   FinalDict = {}
   for key in MappedDataset.keys():
      SN = MappedDataset[key]['SongName']
      Bars = MappedDataset[key]['Bars']
      Prog = MappedDataset[key]['Program']
      AP = MappedDataset[key]['Channel']
      nB = MappedDataset[key]['numBar']
      T = MappedDataset[key]['Tempo']


      List = []
      for i in range(len(SN)):
         dict = {
            'SongName': SN[i],
            'Bars': Bars[i],
            'Program': Prog[i],
            'Channel': AP[i],
            'numBar': nB[i],
            'Tempo': T[i]
         }
         List.append(dict)

      FinalDict[key] = List

   #Deleting wmpty bars
   for key in FinalDict.keys():
      for i in range(len(FinalDict[key])):
         if torch.sum(FinalDict[key][i]['Bars'][0]) <= 0 or torch.sum(FinalDict[key][i]['Bars'][1]) <= 0:
            del FinalDict[key][i]
   

   return FinalDict




def MonoBarsToMIDI(Bars, title='reconstructed', Instrument=None, ticks_per_beat=480):
   mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
   track = mido.MidiTrack()
   mid.tracks.append(track)
   
   track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(100)))
   track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
   
   if Instrument is not None:
      Program = Instrument
   else:
      # random choose one prpgram is no one is given
      Program = np.random.choice(np.arange(1, 129))
   
   track.append(mido.Message('program_change', channel=1, program=Program, time=0))
   
   TicksPerBar = ticks_per_beat * 4
   TicksPerPosition = TicksPerBar // 16
   

   ActiveNotes = {} 
   events = []  
   

   for pos in range(Bars.shape[1]):
      current_time = pos * TicksPerPosition
      
      for note in range(128):
         note_active = Bars[note, pos] > 0
         was_active = note in ActiveNotes
         
         if note_active and not was_active:
            # Note starts
            ActiveNotes[note] = pos
            velocity = int(Bars[note, pos]) if Bars[note, pos] > 1 else 90
            events.append((current_time, 'note_on', note, velocity))
               
         elif not note_active and was_active:
            # Note ends
            velocity = int(Bars[note, ActiveNotes[note]]) if Bars[note, ActiveNotes[note]] > 1 else 90
            events.append((current_time, 'note_off', note, velocity))
            del ActiveNotes[note]
   

   final_time = Bars.shape[1] * TicksPerPosition
   for note in ActiveNotes:
      velocity = int(Bars[note, ActiveNotes[note]]) if Bars[note, ActiveNotes[note]] > 1 else 90
      events.append((final_time, 'note_off', note, velocity))
   
   events.sort(key=lambda x: (x[0], x[1] == 'note_off'))
   
   # Convert to MIDI messages with proper timing
   last_time = 0
   for abs_time, event_type, note, velocity in events:
      delta_time = abs_time - last_time
      
      if event_type == 'note_on':
         track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta_time, channel=1))
      else:  # note_off
         track.append(mido.Message('note_off', note=note, velocity=0, time=delta_time, channel=1))
      
      last_time = abs_time
   
   mid.save(f'{title}.mid')


