import numpy as np


InstrumentFamily_map = {

   'String': ['bass 1', 'bass 2', 'a.guitar', 'e.guitar', 'strings', 'acou bass', 'clean gtr', 'jazz gtr', 'bas', 'fretless bass', 'violin', 'accordation', 
              'fingered bass', 'clean guitar', 'muted guitar', 'steel guitar', 'ac. bass', 'fiddle', 'jazz guitar', 'strings2', 'syn strings', 'harp', 
              'bass', 'guitar 2', 'guitar', 'nylon guitar', 'chords', 'basso', 'slowstrin', 'steel gt', 'archi', 'nylon gt',  'guitar', 'guitar 2', 
              'guitar 3', 'el guitar', 'muted gtr', 'muted gt', 'guitar 1', 'guitar solo', 'guitar 4', 'guitar 5', 'acoustic guitar', 'lead guitar', 
              'bass guitar', 'nylon gtr', 'steel gtr', '12str gtr', 'chorus gt', 'funk gtr', 'fretless', 'slapbass 1', 'fingerdbas', 'fing.bass', 
              'bass&lead', 'syn bass 2', 'bass', 'baixo', 'guitare', 'electric bass (finger)', 'pick bass', 'jazz bass', 'bass line', 'slap bass 2', 
               'bass gtr', 'ac bass', 'fingerdba', 'fretless', 'viola', 'cello', 'pizzicato', 'ac guitar', 'ac. guitar', 'rhythm guitar', 
               'distorted guitar', 'overdriven guitar', 'distortion guitar', 'overdrive guitar', 'guitar i', 'guitar ii', 'guitar1', 
               'guitar2', 'elec guitar', 'banjo', 'charang', 'nylon gtr', '12str gtr', 'chorus gt', 'funk gtr', 'synth bass', 'syn bass 1', 'syn bass 2', 
               '+synbass 3', 'pizzicato', 'string1', 'gtharmonic', 'string 2', 'basse', 'acoustic bass', 'contrabass', 'bass guitar', 'el bass', 
               'fingered', 'fingerbas', 'slap bass', 'slapbass 1', 'slapbass 2', 'bass', 'bass    (bb)', 'acoubass', 'jazz electric', 'electric guitar', 
               'elec guit', 'elec. guitar', 'electric', 'clean gt', 'clean gt.', 'distorted', 'distortio', 'guitar3', 'guitarra', 'steel string', 
               'solo guitar', 'rhythm gu', 'muted gt.', 'acoustic', 'acoustic ', 'nylon gtr', 'banjo', 'viola', 'cello', 'low strings', 'string ensemble',
               'string', 'strings', 'strings 1', 'strings', 'string 1', 'slow strings', 'slow str', 'synth strings', 'pizzicato','electric bass', 
               'fretless bass', 'fretless e bass', 'fretless e.bass', 'fretless e. bass','bass', 'base', 'bass    (bb)', 'picked bass', 'slap bass', 
               'fingered bass', 'acoustic guitar (steel)','electric guitar', 'e. guitar', 'electric guitar (clean)', 'clean guitar', 'clean gt.', 
               'clean guit','distortion guitar', 'dist', 'muted guitar', 'mute guitar', 'muted guitar', 'guitar  (bb)','guitar 1', 'guitar 3', 
               'gitarre', 'lead gtr', 'lead gtr muted', 'guitar i', 'electric guitar i','electric guitar ii', 'electric guitar iii', 'shamisen', 
               'koto', 'cello', 'viola', 'harp','harpsichord', 'harpsichrd'],

   'Keyboard': ['e.piano', 'a.piano', 'celesta', 'organ 1', 'piano', 'lead', 'honky tonk', 'bright piano', 'grand piano', 'hammond', 'a.piano 2', 
                'a.piano 3', 'e.piano 2', 'piano1', 'organ', 'organ 3', 'synthbrass', 'syn str 1', 'calliope', 'marimba', 'crystal', 'melodia', 
                'a.piano', 'ac piano', 'piano', 'piano1', 'piano 1', 'piano 2', 'piano2', 'rhodes', 'clavinet', 'organ', 'organ 2', 'church org', 
                'elec organ', 'keyboard', 'marimba', 'vibraphone', 'glockenspl', 'kalimba', 'warm pad', 'polysynth', 'synth pad', 'synth 1', 'synth 2', 
                'chiffer ld', 'electric piano', 'elec piano', 'elec pian', 'rock organ', 'reed organ', 'organ', 'harpsichrd', 'clavinet', 'rhodes', 
                'warmpad', 'warm pad', 'pad', 'sweep pad', 'polysynth', 'keyboard', 'keyboard 1', 'keyboard 2', 'marimba', 'glockenspl', 'celesta', 
                'tinklebell', 'vibraphone','hammond organ', 'piano', 'piano    (bb)', 'grand piano *merged', 'bright acoustic piano','adlib piano', 
                'piano 2', 'e piano', 'el piano', 'el.piano', 'rhodes piano', 'rhodes','electric piano', 'clavinet', 'music box', 'celesta', 'xylophone', 
                'vibes', 'marimba'],

   'Aerophone': ['clarinet', 'flute', 'tuba', 'trumpet', 'trombone', 'fr.horn', 'brass 1', 'brass', 'ocarina',  'pan flute', 'oboe', 'alto sax', 
                 'tenor sax', 'baritonsax', 'harmonica', 'sitar', 'voice oohs', 'choir aahs', 'alto sax', 'bari sax', 'baritonsax', 'trumpet 1', 
                 'trumpet 2', 'trumpet 3', 'trumpet 4', 'trombone', 'french horn', 'horn', 'brass section', 'piccolo', 'flute', 'shakuhachi', 'bassoon', 
                 'oboe', 'pan flute', 'sax', 'tubularbel','sopran sax', 'baritone sax', 'alto sax', 'piccolo', 'recorder', 'flute', 'shakuhachi', 
                 'oboe', 'bassoon', 'french horn', 'trumpet', 'trombone', 'brass section', 'pan flute', 'harmony', 'choir aah', 'choir aahs', 'voice ooh',
                   'chant', 'vox', 'syn vox','brass', 'brass 2', 'synth brass', 'synbrass', 'synth brass 2', 'trombone', 'trompete','horns', 'sax', 
                   'flute', 'piccolo', 'recorder', 'oboe', 'bassoon', 'harmonica', 'pan flute','shakuhachi', 'koto', 'shamisen'],

   'Percussion': ['drums', 'hi-hats', 'tambourine', 'shaker', 'bass drum', 'taiko', 'percussion', 'toms', 'low tom', 'kick', 'snare', 'drum', 'batteria', 
                  'hihat', 'snare', 'kick', 'snare drum', 'closed h.h', 'ride cymb', 'handclap', 'cymbal', 'tambourin', 'toms', 'claps', 'shaker', 
                  'bassdrum', 'synth drum', 'drums', 'batera', 'melo tom 1', 'drums', 'drum', 'drummix', 'jazz kit', 'kick drum', 'kick drum 1', 
                  'snare', 'hihat', 'hihats', 'hi hat', 'closed hi-hat', 'cymbals', 'ride cymbal', 'tamborine', 'cabasa', 'timpani', 'claps', 'shaker', 
                  'perc', 'orch.hit', 'bell', 'handclap','drumz', 'drums', 'drums   (bb)', 'drum mix', 'kick', 'kick drum', 'bass drum', 'bass drum', 
                  'snare dru', 'sn', 'snare', 'hihats', 'hi-hat', 'open hi hat', 'open hihat', 'closed h.', 'hh', 'crash', 'crash cymb', 'ride', 
                  'ride cymbal', 'toms', 'high tom', 'mid tom', 'sidestick', 'woodblock', 'bongos', 'low conga', 'high conga', 'cowbell', 'steeldrums', 
                  'steel drums', 'agogo', 'cabasa', 'tamborine', 'reverscymb', 'ice rain', 'bowedglass', 'timpani', 'tubularbel', 'orch.hit', 
                  'drums', 'drums   (bb)', 'drums2', 'drums 2', 'drums 1', 'drum set', 'std drums', 'std drum set','gm drums', 'bassdrum', 'bass dru', 
                  'kick 1 36', 'snare', 'snare drum', 'snare    ', 'hihat','hi-hat', 'closed hi hat', 'closed hihat', 'closed hh', 'closed hi', 
                  'hh clsd  42', 'ride','ride', 'cymbals', 'rim shot', 'congas', 'maracas', 'steel drum', 'steel drum','orchestra hit', 
                  'orch.hit', 'bells', 'toms', 'tom mid 2', 'mid tom', 'high tom', 'majortom','woodblock', 'cabasa', 'tambourine', 'bongos', 'cowbell'],

   'Voice': ['choir', 'melody', 'melody 2', 'words', 'voice', 'cori', 'canto', 'rhythm', 'whistle', 'vocals', 'voz', 'vocal-lin', 'synvox', 'vocal', 
             'voice oohs', 'vocals', 'vocals 1', 'vocals 2', 'lead vocal', 'background vocals', 'choir', 'choir ah', 'choir aahs', 'voice oohs', 
             'voices', 'lyrics', 'vocal', 'vocal 1', 'vocal 2', 'synvox', 'vocoder','lead vocals', 'backing vocals', 'backup vocals', 'backup', 
             'vocals', 'vocal', 'vox', 'voices', 'choir', 'syn vox', 'chant', 'harmony', 'lyrics','lead vox', 'solo vox', 'vocal', 'voice 1 (melody)', 
             'voice 2 (harmony)', 'synth voice','spacevoice', 'choir', 'chant', 'background vocals', 'lead', 'lead 2 (sawtooth)', 'vox','backing vocals', 
             'harmony', 'choir aahs', 'voice ooh'],

   'Sync': ['synth', 'fantasia', 'saw wave', 'squarewave', 'overdrive', 'slowstring', 'brightness', 'synthbrass', 'syn str 1', 'calliope', 'crystal', 
            'synvox', 'synth bass', 'syn bass', 'synth bass 2', 'synbrass 1', 'synbrass 2', 'syn string', 'syn str 2', 'saw', 'sawtooth', 'squarewave', 
            'atmosphere', 'crystal', 'chiffer ld', 'distortion', 'synth pad', 'polysynth', 'warm pad', 'slowstring', 'brightness', 'calliope', 
            'synth bass', 'synthbass', 'synth bas', 'synth bass 1', 'synthbrass 1', 'synth strings', 'synth pad', 'warm pad', 'atmosphere', 'crystal', 
            'polysynth', 'sweep pad', 'squarewave', 'sawtooth', 'distortion', 'pad', 'synvox', 'brightness', 'slow pad', 'calliope','synth brass', 
            'synbrass', 'synth brass 2', 'synthstrings 1', 'synth voice', 'spacevoice','metal pad', 'pad 7 (halo)', 'pad 3 (polysynth)', 
            'pad 1 (new age)', 'warm pad', 'sweep pad','polysynth', 'sinth', 'lead 2 (sawtooth)', 'echo drops', 'ice rain', 'brightness', 
            'crystal','atmosphere', 'fantasy', 'bandneon'],

   'Others': ['', 'halo pad', 'staff-1', 'staff-2', 'staff-3', 'staff-4', 'staff-5', 'staff-6', 'staff-7', 'track 2', 'track 3', 'track 4', 'track 5', 
              'track 6', 'track 10', 'track 1', 'track 11', 'tk1', 'tk2', 'tk3', 'tk4', 'tk5', 'tk6', 'tk7', 'tk8', 'tk10', 'unnamed-000', 'unnamed-001', 
              'tk9', 'tk11', 'tk12', 'tk13', 'tk14', 'tk15', 'track 7', 'staff', 'staff-8', 'staff-9', 'unnamed-003']
    
}





GM_INSTRUMENTS = {
    # Piano
    1: "Acoustic Grand Piano",
    2: "Bright Acoustic Piano",
    3: "Electric Grand Piano",
    4: "Honky-Tonk Piano",
    5: "Electric Piano 1",
    6: "Electric Piano 2",
    7: "Harpsichord",
    8: "Clavinet",
    
    # Chromatic Percussion 
    9: "Celesta",
    10: "Glockenspiel",
    11: "Music Box",
    12: "Vibraphone",
    13: "Marimba",
    14: "Xylophone",
    15: "Tubular Bells",
    16: "Dulcimer",
    
    # Organ 
    17: "Drawbar Organ",
    18: "Percussive Organ",
    19: "Rock Organ",
    20: "Church Organ",
    21: "Reed Organ",
    22: "Accordion",
    23: "Harmonica",
    24: "Tango Accordion",
    
    # Guitar 
    25: "Acoustic Guitar (nylon)",
    26: "Acoustic Guitar (steel)",
    27: "Electric Guitar (jazz)",
    28: "Electric Guitar (clean)",
    29: "Electric Guitar (muted)",
    30: "Overdriven Guitar",
    31: "Distortion Guitar",
    32: "Guitar Harmonics",
    
    # Bass 
    33: "Acoustic Bass",
    34: "Electric Bass (finger)",
    35: "Electric Bass (pick)",
    36: "Fretless Bass",
    37: "Slap Bass 1",
    38: "Slap Bass 2",
    39: "Synth Bass 1",
    40: "Synth Bass 2",
    
    # Strings 
    41: "Violin",
    42: "Viola",
    43: "Cello",
    44: "Contrabass",
    45: "Tremolo Strings",
    46: "Pizzicato Strings",
    47: "Orchestral Strings",
    48: "Timpani",
    
    # Ensemble 
    49: "String Ensemble 1",
    50: "String Ensemble 2",
    51: "SynthStrings 1",
    52: "SynthStrings 2",
    53: "Choir Aahs",
    54: "Voice Oohs",
    55: "Synth Voice",
    56: "Orchestra Hit",
    
    # Brass 
    57: "Trumpet",
    58: "Trombone",
    59: "Tuba",
    60: "Muted Trumpet",
    61: "French Horn",
    62: "Brass Section",
    63: "SynthBrass 1",
    64: "SynthBrass 2",
    
    # Reed 
    65: "Soprano Sax",
    66: "Alto Sax",
    67: "Tenor Sax",
    68: "Baritone Sax",
    69: "Oboe",
    70: "English Horn",
    71: "Bassoon",
    72: "Clarinet",
    
    # Pipe 
    73: "Piccolo",
    74: "Flute",
    75: "Recorder",
    76: "Pan Flute",
    77: "Blown Bottle",
    78: "Shakuhachi",
    79: "Whistle",
    80: "Ocarina",
    
    # Synth Lead 
    81: "Lead 1 (square)",
    82: "Lead 2 (sawtooth)",
    83: "Lead 3 (calliope)",
    84: "Lead 4 (chiff)",
    85: "Lead 5 (charang)",
    86: "Lead 6 (voice)",
    87: "Lead 7 (fifths)",
    88: "Lead 8 (bass+lead)",
    
    # Synth Pad 
    89: "Pad 1 (new age)",
    90: "Pad 2 (warm)",
    91: "Pad 3 (polysynth)",
    92: "Pad 4 (choir)",
    93: "Pad 5 (bowed)",
    94: "Pad 6 (metallic)",
    95: "Pad 7 (halo)",
    96: "Pad 8 (sweep)",
    
    # Synth Effects 
    97: "FX 1 (rain)",
    98: "FX 2 (soundtrack)",
    99: "FX 3 (crystal)",
    100: "FX 4 (atmosphere)",
    101: "FX 5 (brightness)",
    102: "FX 6 (goblins)",
    103: "FX 7 (echoes)",
    104: "FX 8 (sci-fi)",
    
    # Ethnic (105-112)
    105: "Sitar",
    106: "Banjo",
    107: "Shamisen",
    108: "Koto",
    109: "Kalimba",
    110: "Bagpipe",
    111: "Fiddle",
    112: "Shanai",
    
    # Percussive (113-120)
    113: "Tinkle Bell",
    114: "Agogo",
    115: "Steel Drums",
    116: "Woodblock",
    117: "Taiko Drum",
    118: "Melodic Tom",
    119: "Synth Drum",
    120: "Reverse Cymbal",
    
    # Sound Effects (121-128)
    121: "Guitar Fret Noise",
    122: "Breath Noise",
    123: "Seashore",
    124: "Bird Tweet",
    125: "Telephone Ring",
    126: "Helicopter",
    127: "Applause",
    128: "Gunshot",
}


GM_PERCUSSION = {
    # Bass Drums / Kicks
    35: "Acoustic Bass Drum",  # (Kick)
    36: "Bass Drum 1",         # (Alternate Kick)

    # Snares / Claps
    37: "Side Stick",          # (Rimshot)
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",

    # Toms
    41: "Low Floor Tom",
    43: "High Floor Tom",
    45: "Low Tom",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    50: "High Tom",

    # Hi-Hats
    42: "Closed Hi-Hat",
    44: "Pedal Hi-Hat",        # (Hi-Hat Foot Splash)
    46: "Open Hi-Hat",

    # Cymbals
    49: "Crash Cymbal 1",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    55: "Splash Cymbal",
    57: "Crash Cymbal 2",
    59: "Ride Cymbal 2",

    # Percussion (Latin/Afro-Cuban)
    54: "Tambourine",
    56: "Cowbell",
    58: "Vibraslap",
    60: "Hi Bongo",
    61: "Low Bongo",
    62: "Mute Hi Conga",
    63: "Open Hi Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "Hi Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle",
}

InstrumentFamily_Map = {}

InstrumentFamily_Map.update({k: 'Piano' for k in list(GM_INSTRUMENTS.keys())[0:8]})
InstrumentFamily_Map.update({k: 'Chromatic Percussion' for k in list(GM_INSTRUMENTS.keys())[8:16]})
InstrumentFamily_Map.update({k: 'Organ' for k in list(GM_INSTRUMENTS.keys())[16:24]})
InstrumentFamily_Map.update({k: 'Guitar' for k in list(GM_INSTRUMENTS.keys())[24:32]})
InstrumentFamily_Map.update({k: 'Bass' for k in list(GM_INSTRUMENTS.keys())[32:40]})
InstrumentFamily_Map.update({k: 'Strings' for k in list(GM_INSTRUMENTS.keys())[40:48]})
InstrumentFamily_Map.update({k: 'Ensemble' for k in list(GM_INSTRUMENTS.keys())[48:56]})
InstrumentFamily_Map.update({k: 'Brass' for k in list(GM_INSTRUMENTS.keys())[56:64]})
InstrumentFamily_Map.update({k: 'Reed' for k in list(GM_INSTRUMENTS.keys())[64:72]})
InstrumentFamily_Map.update({k: 'Pipe' for k in list(GM_INSTRUMENTS.keys())[72:80]})
InstrumentFamily_Map.update({k: 'Synth Lead' for k in list(GM_INSTRUMENTS.keys())[80:88]})
InstrumentFamily_Map.update({k: 'Synth Pad' for k in list(GM_INSTRUMENTS.keys())[88:96]})
InstrumentFamily_Map.update({k: 'Synth Effects' for k in list(GM_INSTRUMENTS.keys())[96:104]})
InstrumentFamily_Map.update({k: 'Ethnic' for k in list(GM_INSTRUMENTS.keys())[104:112]})
InstrumentFamily_Map.update({k: 'Percussive' for k in list(GM_INSTRUMENTS.keys())[112:120]})
InstrumentFamily_Map.update({k: 'Sound Effects' for k in list(GM_INSTRUMENTS.keys())[120:128]})
InstrumentFamily_Map.update({k: 'Percussion' for k in list(GM_PERCUSSION.keys())})


#MPAS THE INSTRUMENT INTO ONE FOR EACH CATEGOTY 
InstrumentMap = {}
InstrumentMap.update({k: 0 for k in list(GM_INSTRUMENTS.keys())[0:8]})
InstrumentMap.update({k: 8 for k in list(GM_INSTRUMENTS.keys())[8:16]})
InstrumentMap.update({k: 16 for k in list(GM_INSTRUMENTS.keys())[16:24]})
InstrumentMap.update({k: 24 for k in list(GM_INSTRUMENTS.keys())[24:32]})
InstrumentMap.update({k: 32 for k in list(GM_INSTRUMENTS.keys())[32:40]})
InstrumentMap.update({k: 40 for k in list(GM_INSTRUMENTS.keys())[40:48]})
InstrumentMap.update({k: 48 for k in list(GM_INSTRUMENTS.keys())[48:56]})
InstrumentMap.update({k: 56 for k in list(GM_INSTRUMENTS.keys())[56:64]})
InstrumentMap.update({k: 64 for k in list(GM_INSTRUMENTS.keys())[64:72]})
InstrumentMap.update({k: 72 for k in list(GM_INSTRUMENTS.keys())[72:80]})
InstrumentMap.update({k: 80 for k in list(GM_INSTRUMENTS.keys())[80:88]})
InstrumentMap.update({k: 88 for k in list(GM_INSTRUMENTS.keys())[88:96]})
InstrumentMap.update({k: 96 for k in list(GM_INSTRUMENTS.keys())[96:104]})
InstrumentMap.update({k: 104 for k in list(GM_INSTRUMENTS.keys())[104:112]})
InstrumentMap.update({k: 112 for k in list(GM_INSTRUMENTS.keys())[112:120]})
InstrumentMap.update({k: 120 for k in list(GM_INSTRUMENTS.keys())[120:128]})

#REMEMBER, PROGRAM 180 DOESN'T EXIST, IF IT HAPPEN IT HAS TO BE MAPPED TO CHANNEL 10 AND PROGRAM 35
InstrumentMap.update({k: 180 for k in list(GM_PERCUSSION.keys())})