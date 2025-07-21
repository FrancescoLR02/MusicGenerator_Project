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


def ToPolyphonicBars(track, TicksPerBeat, length=16):
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
   barNumRecord = []
   for barNum, note, start_pos, end_pos, vel in Note:
      if barNum not in Bars:
         Bars[barNum] = np.zeros((128, length), dtype=int)
         barNumRecord.append(barNum+1)


      # Fill the matrix 
      for pos in range(start_pos, end_pos + 1):
         if pos < length:
            Bars[barNum][note, pos] = 1
   
   barList = []
   for barNum, matrix in Bars.items():
      barList.append(matrix)
   
   return barNumRecord, barList


#From all the track (maximum 4) of a given song build the (4x128x16) tensor 
def ToPolyphonicGeneralInfo(mid, Dataset, file, dir, HowManyInstruments = 4):

   global Empty

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

   ChannelList = []
   ProgramList = []

   #Keeps track of the four family of instrument 
   PianoInstr, GuitarInstr, BassInstr, PercussInstr = 0, 0, 0, 0
   

   #First step: Control which instruments are playing in the song
   for track in mid.tracks:
      HasProgramChange = any(msg.type == 'program_change' for msg in track)
      Channel = [msg.channel for msg in track if msg.type == 'control_change']

      if len(Channel) != 0:
         Channel = Channel[0]
         if HasProgramChange or Channel == 9:
            Program = [msg.program for msg in track if msg.type == 'program_change'][0] if Channel != 9 else 130
            ProgramList.append(Program)
            ChannelList.append(Channel)

   #If there are not at least 4 instruments return nothing
   if len(ProgramList) < HowManyInstruments:
      return Dataset

   #Second step: Choose among the instruments the main ones: Guitar, Bass, Piano and Percussive (special care for the last one)
   ChoosenProg = []
   for prog in ProgramList:
      if 1 <= prog <= 8 and PianoInstr < 1:
         ChoosenProg.append(prog)
         PianoInstr += 1

      elif 25 <= prog <= 32 and GuitarInstr < 1:
         ChoosenProg.append(prog)
         GuitarInstr += 1

      elif 33 <= prog <= 40 and BassInstr < 1:
         ChoosenProg.append(prog)
         BassInstr += 1

      elif prog == 130 and PercussInstr < 1:
         ChoosenProg.append(prog)
         PercussInstr += 1
   
   #Third step: if in the song there are not these instruments choose randomly among the other 
   '''Improvement: choose another instrument forcly'''
   TotalInstruments = PianoInstr + GuitarInstr + BassInstr + PercussInstr
   if TotalInstruments < HowManyInstruments:
      #All the instruments not already choosen
      OtherInstruments = [x for x in ProgramList if x not in ChoosenProg and x != 0]
      try:
         RandomInstruments = np.random.choice(OtherInstruments, HowManyInstruments - TotalInstruments)
      except:
         Empty += 1
         return Dataset
      ChoosenProg.extend(RandomInstruments)

   BarsRecordList, BarsList = [], []
   InstrumentsPlayed = set()
   for track in mid.tracks:

      HasProgramChange = any(msg.type == 'program_change' for msg in track)
      Channel = [msg.channel for msg in track if msg.type == 'control_change']

      if len(Channel) != 0:
         Channel = Channel[0]
      
         if HasProgramChange or Channel == 9:
            Program = [msg.program for msg in track if msg.type == 'program_change'][0] if Channel != 9 else 130

            if Program in ChoosenProg and Program not in InstrumentsPlayed:
               BarsRecord, Bars = ToPolyphonicBars(track, TicksPerBeat)
               #if Bars is None or len(Bars) <4:
                  #continue
               BarsRecordList.append(BarsRecord)
               BarsList.append(Bars)
               InstrumentsPlayed.add(Program)

   if len(ChoosenProg) != 4:
      
      Empty += 1
      return Dataset
   
   if len(BarsRecordList) == 0:
      return Dataset
   
   #Complete array for the number of bars
   try:
      FullBarRecord = AddZeros(BarsRecordList)
   except:
      Empty += 1
      return Dataset

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

   #Determine the genre of the song 
   with open('GenreDataset.pkl', 'rb') as f:
      GenreDataset = pickle.load(f)

   #Counts the number of pair of bars
   Dim = len(PolyphonicDataset)//2

   try:
      SongGenre = GenreDataset[f'{dir}/{file[:-4]}'][0]
      Genre = [SongGenre for _ in range(2, Dim-3, 2)]
   except:
      Empty += 1
      return Dataset


   numPair = [(i, i+1) for i in range(2, Dim - 3, 2)]
   BarsPair = [(PolyphonicDataset[i], PolyphonicDataset[i+1]) for i in range(2, Dim - 3, 2)]
   ActiveProgram = [(FullActiveBars[i], FullActiveBars[i+1]) for i in range(2, Dim - 3, 2)]
   FullProgramList = [ChoosenProg for _ in range(2, Dim - 3, 2)]


   #If there is not the track in the dataset, add it
   if TrackName not in Dataset:               
      Dataset[TrackName] = {
         'SongName': [],
         'Bars': [],
         'Program': [],
         'ActiveProgram': [],
         'numBar': [],
         'Tempo': [], 
         'Genre': []
      }

   #Maps the program into one instrument of the same category
   
   #and add the information to the Dataset dictionary
   Dataset[TrackName]['SongName'].extend([(f'{TrackName}', f'{TrackName}') for _ in range(2, Dim - 3, 2)])
   Dataset[TrackName]['Bars'].extend(BarsPair)
   Dataset[TrackName]['Program'].extend(FullProgramList)
   Dataset[TrackName]['ActiveProgram'].extend(ActiveProgram)
   Dataset[TrackName]['numBar'].extend(numPair)
   Dataset[TrackName]['Tempo'].extend([int(Tempo) for _ in range(2, Dim - 3, 2)])
   Dataset[TrackName]['Genre'].extend(Genre)


   return Dataset      



Empty = 0
#After having created the dataset, cathegorize the songs in the corresponding genre
def PolyphonicPreProcessing(nDir = 300):

   Dataset = {}
   iter = 0

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
         iter += 1

         FilePath = os.path.join(DirPath, file)

         #Cleaned monophonic: Some songs are corrupted:
         mid = Func_CorruptedFile(FilePath, file, dir)

         if mid is None:
            continue

         Dataset = ToPolyphonicGeneralInfo(mid, Dataset, file, dir)



   FinalDict = []
   for key in Dataset.keys():
      SN = Dataset[key]['SongName']
      Bars = Dataset[key]['Bars']
      Prog = Dataset[key]['Program']
      AP = Dataset[key]['ActiveProgram']
      nB = Dataset[key]['numBar']
      T = Dataset[key]['Tempo']
      Genre = Dataset[key]['Genre']


      list = []
      for i in range(len(SN)):
         dict = {
            'SongName': SN[i],
            'Bars': Bars[i],
            'Program': Prog[i],
            'ActiveProgram': AP[i],
            'numBar': nB[i],
            'Tempo': T[i],
            'Genre': Genre[i]
         }
         list.append(dict)

      FinalDict.extend(list)
   
   Del = 0
   for i in reversed(range(len(FinalDict))):
      if torch.sum(FinalDict[i]['Bars'][0]) == 0 or torch.sum(FinalDict[i]['Bars'][1]) == 0:
         Del += 1
         del FinalDict[i]

   print(Del)

   return FinalDict




#From Nx128x16 matrix to midi file
def PolyBarsToMIDI(Bars, Cond1D , title='reconstructed', ticks_per_beat=480):
   mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

   tempo = Cond1D[0]
   Program = Cond1D[2:]
   
   # Add a tempo track (track 0) - global tempo for all instruments
   tempo_track = mido.MidiTrack()
   mid.tracks.append(tempo_track)
   tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(int(tempo)), time=0))
   tempo_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
   tempo_track.append(mido.MetaMessage('end_of_track', time=0))
   
   HowManyInstruments = Bars.shape[0]
   num_positions = Bars.shape[2]
   
   # Calculate timing
   ticks_per_bar = ticks_per_beat * 4
   ticks_per_position = ticks_per_bar // 16
   
   # Process each instrument
   for Instr_IDX in range(HowManyInstruments):
      track = mido.MidiTrack()
      mid.tracks.append(track)
      
      # Set instrument program
      if Program[Instr_IDX] == 130: 
         track.append(mido.Message('program_change', channel=9, program=0, time=0))
      else:
         track.append(mido.Message('program_change', channel=Instr_IDX, program=Program[Instr_IDX], time=0))
      
      ActiveNotes = {}
      events = []  
      
      # Process each time position for this instrument
      for pos in range(num_positions):
         current_time = pos * ticks_per_position
         
         for note in range(128):
               note_active = Bars[Instr_IDX, note, pos] > 0
               was_active = note in ActiveNotes
               
               if note_active and not was_active:
                  # Note starts
                  ActiveNotes[note] = pos
                  velocity = 90
                  events.append((current_time, 'note_on', note, velocity))
                  
               elif not note_active and was_active:
                  # Note ends
                  velocity = 90
                  events.append((current_time, 'note_off', note, velocity))
                  del ActiveNotes[note]
      
      # Handle any notes that are still active at the end
      final_time = num_positions * ticks_per_position
      for note in ActiveNotes:
         velocity = 90
         events.append((final_time, 'note_off', note, velocity))
      
      # Sort events by time (note_on before note_off at same time)
      events.sort(key=lambda x: (x[0], x[1] == 'note_off'))
      
      # Convert to MIDI messages with proper timing
      last_time = 0
      for abs_time, event_type, note, velocity in events:
         delta_time = abs_time - last_time
         
         if event_type == 'note_on':
               track.append(mido.Message('note_on', note=note, velocity=velocity, 
                                       time=delta_time, channel=Instr_IDX))
         else:  # note_off
               track.append(mido.Message('note_off', note=note, velocity=0, 
                                       time=delta_time, channel=Instr_IDX))
         
         last_time = abs_time
      
      # End of track
      track.append(mido.MetaMessage('end_of_track', time=0))
   
   mid.save(f'{title}.mid')