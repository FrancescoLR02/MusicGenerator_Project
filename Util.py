import numpy as np


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
#InstrumentFamily_Map.update({k: 'Percussion' for k in list(GM_PERCUSSION.keys())})


#MPAS THE SPACIFIC INSTRUMENT INTO ONE FOR EACH CATEGOTY 
InstrumentMap = {}
InstrumentMap.update({k: 1 for k in list(GM_INSTRUMENTS.keys())[0:8]})
InstrumentMap.update({k: 9 for k in list(GM_INSTRUMENTS.keys())[8:16]})
InstrumentMap.update({k: 17 for k in list(GM_INSTRUMENTS.keys())[16:24]})
InstrumentMap.update({k: 25 for k in list(GM_INSTRUMENTS.keys())[24:32]})
InstrumentMap.update({k: 33 for k in list(GM_INSTRUMENTS.keys())[32:40]})
InstrumentMap.update({k: 41 for k in list(GM_INSTRUMENTS.keys())[40:48]})
InstrumentMap.update({k: 49 for k in list(GM_INSTRUMENTS.keys())[48:56]})
InstrumentMap.update({k: 57 for k in list(GM_INSTRUMENTS.keys())[56:64]})
InstrumentMap.update({k: 65 for k in list(GM_INSTRUMENTS.keys())[64:72]})
InstrumentMap.update({k: 73 for k in list(GM_INSTRUMENTS.keys())[72:80]})
InstrumentMap.update({k: 81 for k in list(GM_INSTRUMENTS.keys())[80:88]})
InstrumentMap.update({k: 89 for k in list(GM_INSTRUMENTS.keys())[88:96]})
InstrumentMap.update({k: 97 for k in list(GM_INSTRUMENTS.keys())[96:104]})
InstrumentMap.update({k: 105 for k in list(GM_INSTRUMENTS.keys())[104:112]})
InstrumentMap.update({k: 113 for k in list(GM_INSTRUMENTS.keys())[112:120]})
InstrumentMap.update({k: 121 for k in list(GM_INSTRUMENTS.keys())[120:128]})