import numpy as np
import mido
import os
from tqdm import tqdm
from scipy import sparse
import torch
from collections import defaultdict
import gc


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



#####PRE PROCESSING #####

#Transorm a given track into monophonic
def ToMonophonic(track):
   absTime = 0
   Events, Metadata, ProgramChange = [], [], []
   channel = None

   for msg in track:
      absTime += msg.time

      if msg.is_meta:
         Metadata.append((absTime, msg))

      elif msg.type == 'note_on' and msg.velocity > 0:
         Events.append((absTime, msg.note, msg.velocity, 'on', msg.channel))
         if channel is None:
            channel = msg.channel

      elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
         Events.append((absTime, msg.note, 0, 'off', msg.channel))
         if channel is None:
               channel = msg.channel

      elif msg.type == 'program_change':
         ProgramChange.append((absTime, msg))
         if channel is None:
            channel = msg.channel

   # sort note events: first by time, then by descending pitch
   Events.sort(key=lambda x: (x[0], -x[1]))

   activeNote = None
   monoEvents = []

   for time, note, velocity, kind, _ in Events:
      if kind == 'on':
         if activeNote is None or note > activeNote:
            if activeNote is not None:
               monoEvents.append(('off', activeNote, time, velocity))
            activeNote = note
            monoEvents.append(('on', note, time, velocity))
      elif kind == 'off' and note == activeNote:
         monoEvents.append(('off', note, time, velocity))
         activeNote = None

   # Rebuild the monophonic track
   newTrack = MidiTrack()
   prevTime = 0

   # Add metadata
   for absTime, msg in sorted(Metadata, key=lambda x: x[0]):
      delta = absTime - prevTime
      msgDict = msg.dict().copy()
      msgDict.pop('time', None)
      msgDict.pop('type', None)
      newTrack.append(MetaMessage(msg.type, time=delta, **msgDict))
      prevTime = absTime

   # Add program_change messages
   for absTime, msg in sorted(ProgramChange, key=lambda x: x[0]):
      delta = absTime - prevTime
      msgDict = msg.dict().copy()
      msgDict.pop('time', None)
      msgDict.pop('type', None)
      newTrack.append(Message('program_change', time=delta, **msgDict))
      prevTime = absTime

   # Add monophonic note events
   for kind, note, absTime, velocity in monoEvents:
      delta = absTime - prevTime
      if kind == 'on':
         newTrack.append(Message('note_on', note=note, velocity=velocity, time=delta, channel=channel))
      else:
         newTrack.append(Message('note_off', note=note, velocity=velocity, time=delta, channel=channel))
      prevTime = absTime

   return newTrack




#Recreate the whole database with monophonic information
def RecreateDatabase():

   InputPath = os.path.realpath('clean_midi')
   os.makedirs('Mono_CleanMidi', exist_ok=True)
   OutputPath = os.path.realpath('Mono_CleanMidi')

   for dir in tqdm(os.listdir(InputPath), desc='Recreating Database'):
      DirPath = os.path.join(InputPath, dir)

      if not os.path.isdir(DirPath):
         continue

      #In the output path, create the folder of the artist if does not exits
      if not os.path.exists(os.path.join(OutputPath, dir)):
        os.makedirs(os.path.join(OutputPath, dir))

      for file in os.listdir(DirPath):
         FilePath = os.path.join(DirPath, file)

         mid = mido.MidiFile(FilePath)
         #Instatiate the new monophonic midi file
         newMid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

         #loop over all the tracks in the original file and saving as new file:
         for track in mid.tracks:
            
            try: 
               MonoMidi = ToMonophonic(track)
               newMid.tracks.append(MonoMidi)
            except (KeyError) as e:
               continue
         
         try:
            newMid.save(os.path.join(OutputPath, dir, file))
         except (ValueError, KeyError) as e:
            continue

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
            start_time, velocity = ActiveNotes[msg.note]
            end_time = currTime
            
            # Compute position
            start_bar = start_time // TicksPerBar
            end_bar = end_time // TicksPerBar
            start_pos = (start_time % TicksPerBar) // TicksPerSixteenth
            end_pos = (end_time % TicksPerBar) // TicksPerSixteenth
            
            if start_bar == end_bar:
               # Note within single bar
               if start_pos < length and end_pos <= length:
                  Note.append((start_bar, msg.note, start_pos, min(end_pos, length-1), velocity))
            else:
               if start_pos < length:
                  Note.append((start_bar, msg.note, start_pos, length-1, velocity))
               
               for bar in range(start_bar + 1, end_bar):
                  Note.append((bar, msg.note, 0, length-1, velocity))
               
               if end_pos > 0 and end_pos <= length:
                  Note.append((end_bar, msg.note, 0, end_pos-1, velocity))
            
            del ActiveNotes[msg.note]
   
   # Handle any notes that were never turned off
   for note, (start_time, velocity) in ActiveNotes.items():
      start_bar = start_time // TicksPerBar
      start_pos = (start_time % TicksPerBar) // TicksPerSixteenth
      if start_pos < length:
         Note.append((start_bar, note, start_pos, start_pos, velocity))
   
   Bars = {}
   for bar_num, note, start_pos, end_pos, vel in Note:
      if bar_num not in Bars:
         Bars[bar_num] = np.zeros((128, length), dtype=int)
      
      # Fill the matrix 
      for pos in range(start_pos, end_pos + 1):
         if pos < length:
            if Velocity:
               Bars[bar_num][note, pos] = vel
            else:
               Bars[bar_num][note, pos] = 1
   
   barList = []
   for barNum, matrix in Bars.items():
      Tensor = torch.tensor(matrix, dtype=torch.int)
      barList.append(Tensor.to_sparse())
   
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

         if Program == 0 or Channel == 10:
            continue

         #Compute the (128x16) bars matrix for each track
         Bars = ToBars(track, TicksPerBeat, Velocity)

         if Bars is None or len(Bars) < 2:
            continue

         #Counts the number of pair of bars
         numPair = [(i, i+1) for i in range(0, len(Bars)//2 - 1, 2)]
         BarsPair = [(Bars[i], Bars[i+1]) for i in range(0, len(Bars)//2 - 1, 2)]


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
         Dataset[TrackName]['Tempo'].extend([(int(Tempo), int(Tempo)) for _ in range(0, len(Bars)//2-1, 2)])
         Dataset[TrackName]['Program'].extend([(Program, Program) for _ in range(0, len(Bars)//2-1, 2)])
         Dataset[TrackName]['Channel'].extend([(Channel, Channel) for _ in range(0, len(Bars)//2-1, 2)])
         Dataset[TrackName]['SongName'].extend([(f'{TrackName}', f'{TrackName}') for _ in range(0, len(Bars)//2-1, 2)])
         Dataset[TrackName]['numBar'].extend(numPair)

   return Dataset


def PreProcessing(nDir = 300, Velocity = False):

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

         Dataset = ToGeneralInfo(mid, Dataset, file, Velocity)

   
   MappedDataset = {}
   for key in Dataset:
      value = Dataset[key]
      
      for i, prog in enumerate(value['Program']):

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
   for key in tqdm(FinalDict.keys(), desc='Deleting Empty bars:'):
      for i in range(len(FinalDict[key])):
         if torch.sum(FinalDict[key][i]['Bars'][0]) == 0 or torch.sum(FinalDict[key][i]['Bars'][1]) == 0:
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
   
   ticks_per_bar = ticks_per_beat * 4
   ticks_per_position = ticks_per_bar // Bars.shape[1] 
   

   ActiveNotes = {} 
   events = []  
   

   for pos in range(Bars.shape[1]):
      current_time = pos * ticks_per_position
      
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
   

   final_time = Bars.shape[1] * ticks_per_position
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

