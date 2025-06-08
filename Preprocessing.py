import numpy as np
import mido
import os
from tqdm import tqdm
from mido.midifiles.meta import KeySignatureError
from mido import MidiTrack, MetaMessage, Message
import Util as Util


LogFolder = os.path.realpath('LogFolder')
EmptyFolder = 'EmptyFolder.txt'
CorruptedSongs = 'CorruptedSongs.txt'
WrongTimeStamp = 'WrongTimeStamp.txt'

######CLEANING OF THE DATA#####


#Checks if the folder in the dataset is empty or not given the path to the folder.
#If empty write in a log file the name of the folder
def Func_EmptyFolder(DirPath, dir):

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
def Func_CorruptedFile(FilePath, file, dir):

   try :
      mid = mido.MidiFile(FilePath)
      return mid

   except (OSError, ValueError, KeyError, KeySignatureError, EOFError) as e:

      CorruptedFilePath = os.path.join(LogFolder, CorruptedSongs)
      with open(CorruptedFilePath, 'a') as f:
         f.write(f'{dir}/{file}\n')

         try:
            os.remove(FilePath)
         except Exception as rm_err:
            print(f"Failed to delete {file}: {rm_err}")



#Check the time signature of the file, for now considering only the one with 4/4
def Func_CheckTimeStamp(FilePath, track, file, dir):
      
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
def CleaningData():

   InputPath = os.path.realpath('clean_midi')
   os.makedirs('LogFolder', exist_ok=True)
   
   for dir in tqdm(os.listdir(InputPath), desc='Cleaning Data'):

      DirPath = os.path.join(InputPath, dir)

      #to avoid .Ds_Store to be read
      if not os.path.isdir(DirPath):
         continue

      for file in os.listdir(DirPath):
         FilePath = os.path.join(DirPath, file)

         
         mid = Func_CorruptedFile(FilePath, file, dir) 
         if mid is None:
            continue

         #Check the timestamp (found in the first track as convention)
         InitTrack = mid.tracks[0]
         Func_CheckTimeStamp(FilePath, InitTrack, file, dir)

      Func_EmptyFolder(DirPath, dir)



#####PRE PROCESSING #####

#Transorm a given track into monophonic
def ToMonphonic(track):

   absTime = 0
   Events, Metadata = [], []

   for msg in track:
      absTime += msg.time

      #Recreate metadata with absolute time from original midi file
      if msg.is_meta:
         Metadata.append((absTime, msg))
      elif msg.type == 'note_on' and msg.velocity > 0:
                        #time, note, velocity and kind
         Events.append((absTime, msg.note, msg.velocity, 'on'))
      elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
         Events.append((absTime, msg.note, 0, 'off'))

   #sort events prioritizing the ones with the higher notes
   Events.sort(key = lambda x: (x[0], -x[1]))

   activeNote = None
   monoEvents = []

   #Checks if there are multiple active notes (polyphonic)
   #Ifthere are choose the one with the highest note 
   #recreate the MidiMessage
   for time, note, velocity, kind in Events:
      if kind == 'on':
         if activeNote is None or note > activeNote:
            if activeNote is not None:
               monoEvents.append(('off', activeNote, time, velocity))
            activeNote = note
            monoEvents.append(('on', note, time, velocity))
      elif kind == 'off' and note == activeNote:
         monoEvents.append(('off', note, time, velocity))
         activeNote = None


   #Rebuild the monophonic track
   newTrack = MidiTrack()
   prevTime = 0

   for absTime, msg in sorted(Metadata, key=lambda x: x[0]):
      Delta = absTime - prevTime
      #define a dictionary in which append all the information
      msgDict = msg.dict().copy()
      #pop the information already presents 
      msgDict.pop('time', None)
      msgDict.pop('type', None)

                                    #add the informations and unpack the dictionary
      newTrack.append(MetaMessage(msg.type, time=Delta, **msgDict))
      prevTime = absTime

   #add Note Message 
   for kind, note, absTime in monoEvents:
      Delta = absTime - prevTime
      if kind == 'on':                                   #Flatten the velocity to 64 (can do better)
         newTrack.append(Message('note_on', note = note, velocity = 64, time = Delta))

      else:
         newTrack.append(Message('note_off', note = note, velocity = 64, time = Delta))
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
               MonoMidi = ToMonphonic(track)
               newMid.tracks.append(MonoMidi)
            except (KeyError) as e:
               continue
         
         try:
            newMid.save(os.path.join(OutputPath, dir, file))
         except (ValueError, KeyError) as e:
            continue


#Build the (128x16) beat matrix for each track
def ToBars(track, TicksPerBeat):

   #since these tracks are all 4/4
   TicksPerBar = TicksPerBeat * 4
   TicksPerSixteenth = TicksPerBar // 16

   currTime = 0
   Note = []

   for msg in track:
      currTime += msg.time
      if msg.type == 'note_one' and msg.velocity > 0:
         barNumber = currTime // TicksPerBar
         posInBar = (currTime % TicksPerBar) // TicksPerSixteenth
   
         if posInBar < 16:
            Note.append((barNumber, msg.note, posInBar))

   Bars = {}
   for barNum, note, pos in Note:
      if barNum not in Bars:
         Bars[barNum] = np.zeros((128, 16), dtype = int)

      #Fill the matrix with the note at it's correct position
      Bars[barNum][:, pos] = 0
      Bars[barNum][note, pos] = 1

   maxBar = max(Bars.keys()) if Bars else 0
   barList = []

   for i in range(maxBar + 1):
      barList.append(Bars.get(i, np.zeros((128, 16), dtype = int)))

   return barList


#Maps every track into the instrument family (string, keybord, ...)
def InstrumentFamily(name):
   name = name.lower()
   for standard, aliases in Util.InstrumentFamily_map.items():
      if any(alias in name for alias in aliases):
         return standard
   return name 


#Maps the revious dataset into the new one with the 7 family
def ReMap_Database(Dataset):
   NormDataset = {}

   for name, data in Dataset.items():
      Family = InstrumentFamily(name)

      if Family not in NormDataset:
         NormDataset[Family] = {
            'Bars': [], 
            'Song': [],
            'Tempo': []
         }
      NormDataset[Family]['Bars'].extend(data['Bars'])
      NormDataset[Family]['Song'].extend(data['Song'])
      NormDataset[Family]['Tempo'].extend(data['Tempo'])

   return NormDataset



def ToGeneralInfo(mid, Dataset, file):

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
            Tempo = 0

   #Loop over all tracks (beside the first --> metadata)
   for track in mid.tracks[1:]:
      #set a minimum length of track (might me garbage)
      if len(track) > 100: 

         #Compute the (128x16) bars matrix for each track
         Bars = ToBars(track, TicksPerBeat)

         TrackName = track.name.lower()
         #If there is not the track in the dataset, add it
         if TrackName not in Dataset:               
            Dataset[TrackName] = {
               'Bars': [],
               'Song': [],
               'Tempo': []
            }
         
         #and add the information to the Dataset dictionary
         Dataset[TrackName]['Bars'].extend(Bars)
         Dataset[TrackName]['Song'].append(f'{file[:-4]}')
         Dataset[TrackName]['Tempo'].append(int(Tempo))

   return Dataset



def PreProcessing():

   Dataset = {}

   InputPath = os.path.relpath('Mono_CleanMidi')
   
   #Given a tempo, returns BPM
   

   for dir in tqdm(os.listdir(InputPath), desc='Preprocessing'):
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

         Dataset = ToGeneralInfo(mid, Dataset, file)


   #Remove garbage tracks
   for track in list(Dataset.keys()):
      if len(Dataset[track]['Tempo']) < 20:
         del Dataset[track]

   NormDataset = ReMap_Database(Dataset)
   Dataset = NormDataset

   return Dataset