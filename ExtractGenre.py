from collections import Counter
import warnings


from Preprocessing import *



def ExtractInstrumentFeatures(mid):
    
   InstrumentCounts = {
        'String': 0, 'Keyboard': 0, 'Aerophone': 0, 
        'Percussion': 0, 'Voice': 0, 'Sync': 0, 'Others': 0
    }
    
   totTracks = 0
   for track in mid.tracks:

      if len(track) > 20:
         Family = InstrumentFamily(track.name)
         if Family in InstrumentCounts:
            InstrumentCounts[Family] += 1
            totTracks += 1
   
   # Create proportional features
   InstrumentFeatures = {}
   for Family, count in InstrumentCounts.items():
      InstrumentFeatures[f'{Family}_ratio'] = count / totTracks if totTracks > 0 else 0

   #cast the instrument into a number (enable kmeans)
   InstrumentCast = {
      'String': 0, 'Keyboard': 1, 'Aerophone': 2, 
      'Percussion': 3, 'Voice': 4, 'Sync': 5, 'Others': 6
   }
   
   # Additional features
   InstrumentFeatures.update({
      'TotInstruments': len([c for c in InstrumentCounts.values() if c > 0]),
      'DominantInstrument': InstrumentCast[max(InstrumentCounts, key=InstrumentCounts.get)],
      'InstrumentDiversity': len([c for c in InstrumentCounts.values() if c > 0]) / 7
   })
   
   return InstrumentFeatures



def TrackInfo(mid):

   ticksPerBeat = mid.ticks_per_beat


   TrackFeatures = {
      'trackLength': [],
      'maxTrackVel': [], 
      'VelSpread': [],
      'NoteSpread': [],
      'NoteDensity': []
   }

   #features for the tracks
   for track in mid.tracks:
      if len(track) > 20:

         trackVel, trackNote = [], []
         time = 0

         trackLength = len(track)

         for msg in track:

            time += msg.time  
            if msg.type == 'note_on' and msg.velocity > 0:
               trackVel.append(msg.velocity)
               trackNote.append(msg.note)

         if trackVel and trackNote and time:
            TrackFeatures['trackLength'].append(trackLength)
            TrackFeatures['maxTrackVel'].append(np.max(trackVel))
            TrackFeatures['VelSpread'].append(np.max(trackVel) - np.min(trackVel))
            TrackFeatures['NoteSpread'].append(np.max(trackNote) - np.min(trackNote))
            TrackFeatures['NoteDensity'].append((len(trackNote)*ticksPerBeat) / time)

   InstrumentFeatures = ExtractInstrumentFeatures(mid)

   TrackFeatures.update(**InstrumentFeatures)

   AverageTracksInfo = {
      'AvgTrackLength': np.mean(TrackFeatures['trackLength']),
      'AvgMaxTrackVel': np.mean(TrackFeatures['maxTrackVel']),
      'AvgVelSpread': np.mean(TrackFeatures['VelSpread']),
      'AvgNoteSpread': np.mean(TrackFeatures['NoteSpread']),
      'AveNoteDesity': np.mean(TrackFeatures['NoteDensity']),

      **InstrumentFeatures
   }

   return AverageTracksInfo



def SongInfo(mid):

   #features for the whole song
   Vel, Note, NoteTime = [], [], []
   DistNote = []
   time = 0
   SyncopatedNotes = 0
   DirectionChange = 0
   steps, leaps = 0, 0
   ticksPerBeat = mid.ticks_per_beat
   Tempo = 0

   for msg in mid.tracks[0]:
      if msg.type == 'set_tempo':
         Tempo = mido.tempo2bpm(msg.tempo)

   for track in mid.tracks:

      if len(track) > 20:
         for msg in track:
            time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
               Vel.append(msg.velocity)
               Note.append(msg.note)
               NoteTime.append(time)

   if len(NoteTime) > 1:
      InterIntervals = [NoteTime[i+1] - NoteTime[i] for i in range(len(NoteTime)-1)]
      RythmRegularity = 1 / (np.std(InterIntervals) + 1)  # Higher = more regular
   else:
      RythmRegularity = 1

   
   ticks_per_measure = ticksPerBeat * 4 #Since we are restrincting 4/4

   #Compute syncopation
   for note_time in NoteTime:
      beat_position = (note_time % ticks_per_measure) / ticksPerBeat
      # Check if note is significantly off the main beats (0, 1, 2, 3)
      distance_to_beat = min(beat_position % 1, 1 - (beat_position % 1))
      if distance_to_beat > 0.1:  # Threshold for "off beat"
         SyncopatedNotes += 1

   SyncopationLevel = SyncopatedNotes / len(NoteTime) if NoteTime else 0

   #Compute note change direction
   if len(Note) > 2:
      for i in range(1, len(Note)-1):
         # Check if direction changes from previous interval to next
         prev_direction = Note[i] - Note[i-1]
         next_direction = Note[i+1] - Note[i]
         if prev_direction * next_direction < 0:  # Sign change = direction change
            DirectionChange += 1

   #Compute the StepLeap (difference in note steps)   
   for dist in DistNote:
      if dist <= 2:  # Half-step or whole-step
         steps += 1
      else:  # Leap (more than whole step)
         leaps += 1
   StepLeap = steps / (leaps + 1)  # +1 to avoid division by zero


   #Compute the entropy (Predictability of the notes)
   if DistNote:
      interval_counts = Counter(DistNote)
      total_intervals = len(DistNote)
      interval_entropy = -sum((count/total_intervals) * np.log2(count/total_intervals) 
                              for count in interval_counts.values())
      MelodicEntropy = 1 / (interval_entropy + 1)
   else:
      MelodicEntropy = 1

   AverageSongFeatures = {
      
      'Tempo': int(Tempo),
      'RhythmicRegularity': RythmRegularity,
      'SyncopationLevel': SyncopationLevel,
      'NoteDensity': (len(Note)*ticksPerBeat) / time if time > 0 else 0,
      'NoteRange': int(np.max(Note) - np.min(Note)) if Note else 0,
      'AvgNote': np.mean(Note) if Note else 0,
      'StdNote': np.std(Note) if Note else 0,
      'MelodicDirectionChanges': DirectionChange / len(Note) if Note else 0,
      'StepLeapRatio': StepLeap,
      'MelodicEntropy': MelodicEntropy,
      
   }

   return AverageSongFeatures



#Some information about one song and the tracks inside it:
def ExtractAverageInfo(mid):

   #Features about the song
   AverageSongFeatures = SongInfo(mid)

   #features for the tracks
   AverageTrackFeatures = TrackInfo(mid)

   
   AverageInfo = {

      **AverageSongFeatures,
      **AverageTrackFeatures

   }

   return AverageInfo



def Clustering():
   it = 0

   DatasetFeatures = {}

   InputPath = os.path.realpath('Mono_CleanMidi')

   AllFolders = os.listdir(InputPath)
   Folders = np.random.choice(AllFolders, 2050, replace=False)

   for dir in tqdm(AllFolders):
      DirPath = os.path.join(InputPath, dir)

      if not os.path.isdir(DirPath):
         continue

      #Real all the file in each folder
      for file in os.listdir(DirPath):
         FilePath = os.path.join(DirPath, file)

         mid = mido.MidiFile(FilePath)

         try:
            with warnings.catch_warnings():
               warnings.simplefilter("error", category=RuntimeWarning)
               AverageInfo = list(ExtractAverageInfo(mid).values())
         except (RuntimeWarning, Exception):
            it += 1
            AverageInfo = {}.values()
            
         if len(AverageInfo) == 0:
            continue

         DatasetFeatures[f'{dir}/{file}'] = AverageInfo

   print(f'Number of Excluded songs: {it}')

   return DatasetFeatures



def GenrePrep(ClusteringDF, clusters):
   path = os.path.realpath('Mono_CleanMidi')

   GenrePreprocessing = {}

   for cluster in np.unique(clusters):

      Dataset = {}
      
      for song in tqdm(ClusteringDF[ClusteringDF['Cluster'] == cluster]['Song name']):

         songPath = os.path.join(path, song)

         mid = mido.MidiFile(songPath)

                  #Function in Preprocessing.py
         Dataset = ToGeneralInfo(mid, Dataset, song)

      #Remove garbage tracks
      for track in list(Dataset.keys()):
         if len(Dataset[track]['Tempo']) < 20:
            del Dataset[track]

                  #Function in preprocessing.py
      NormDataset = ReMap_Database(Dataset)
      Dataset = NormDataset

      GenrePreprocessing[cluster] = Dataset

   
   for c in GenrePreprocessing.keys():
      for track in GenrePreprocessing[c].keys():
         GenrePreprocessing[c][track]['Bars'] = np.array(GenrePreprocessing[c][track]['Bars'])

   return GenrePreprocessing

      