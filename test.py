import mido
import numpy as np
from collections import defaultdict
import os

def preprocess_midi_for_midinet(midi_file_path, ticks_per_beat=480):
    """
    Preprocess MIDI file to extract melody for MidiNet 2D conditioning
    
    Args:
        midi_file_path: Path to MIDI file
        ticks_per_beat: MIDI ticks per beat (usually 480)
    
    Returns:
        list of 128x16 numpy arrays (one per bar)
    """
    
    # Load MIDI file
    mid = mido.MidiFile(midi_file_path)
    
    # Step 1: Extract all note events with absolute timing
    note_events = extract_note_events(mid)
    
    # Step 2: Convert to monophonic melody (take highest note)
    melody_events = convert_to_monophonic(note_events)
    
    # Step 3: Segment into bars (assuming 4/4 time)
    bars = segment_into_bars(melody_events, ticks_per_beat)
    
    # Step 4: Convert each bar to 128x16 matrix
    matrices = []
    for bar in bars:
        matrix = convert_bar_to_matrix(bar, ticks_per_beat)
        matrices.append(matrix)
    
    return matrices

def extract_note_events(mid):
    """Extract all note events with absolute timing"""
    events = []
    absolute_time = 0
    
    # Process all tracks
    for track in mid.tracks:
        track_time = 0
        
        for msg in track:
            track_time += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append({
                    'type': 'note_on',
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'time': track_time,
                    'channel': msg.channel
                })
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append({
                    'type': 'note_off',
                    'note': msg.note,
                    'time': track_time,
                    'channel': msg.channel
                })
    
    # Sort by time
    events.sort(key=lambda x: x['time'])
    return events

def convert_to_monophonic(note_events):
    """Convert polyphonic to monophonic by taking highest active note"""
    melody = []
    active_notes = set()
    current_time = 0
    
    for event in note_events:
        # If time has advanced and we have active notes, record the highest
        if event['time'] > current_time and active_notes:
            highest_note = max(active_notes)
            melody.append({
                'note': highest_note,
                'start_time': current_time,
                'end_time': event['time']
            })
        
        current_time = event['time']
        
        if event['type'] == 'note_on':
            active_notes.add(event['note'])
        elif event['type'] == 'note_off':
            active_notes.discard(event['note'])
    
    return melody

def segment_into_bars(melody_events, ticks_per_beat):
    """Segment melody into bars (assuming 4/4 time, so 4 beats per bar)"""
    ticks_per_bar = ticks_per_beat * 4
    bars = defaultdict(list)
    
    for event in melody_events:
        bar_number = event['start_time'] // ticks_per_bar
        
        # Adjust timing to be relative to bar start
        bar_start_time = bar_number * ticks_per_bar
        event_copy = event.copy()
        event_copy['start_time'] -= bar_start_time
        event_copy['end_time'] -= bar_start_time
        
        # Handle notes that span multiple bars
        if event_copy['end_time'] > ticks_per_bar:
            event_copy['end_time'] = ticks_per_bar
        
        bars[bar_number].append(event_copy)
    
    return [bars[i] for i in sorted(bars.keys())]

def convert_bar_to_matrix(bar_events, ticks_per_beat):
    """Convert bar events to 128x16 matrix (128 MIDI notes, 16 sixteenth notes)"""
    matrix = np.zeros((128, 16), dtype=np.float32)
    
    # Each sixteenth note = ticks_per_beat / 4
    sixteenth_ticks = ticks_per_beat // 4
    
    for event in bar_events:
        note = event['note']
        start_sixteenth = int(event['start_time'] // sixteenth_ticks)
        end_sixteenth = int(event['end_time'] // sixteenth_ticks)
        
        # Ensure we stay within bounds
        start_sixteenth = max(0, min(15, start_sixteenth))
        end_sixteenth = max(0, min(16, end_sixteenth))
        
        # Fill the matrix for the duration of the note
        for t in range(start_sixteenth, end_sixteenth):
            if t < 16:  # Safety check
                matrix[note, t] = 1.0
    
    return matrix

def visualize_matrix(matrix, title="Bar Visualization"):
    """Helper function to visualize a bar matrix"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    plt.imshow(matrix, aspect='auto', origin='lower', cmap='Blues')
    plt.xlabel('Time Steps (16th notes)')
    plt.ylabel('MIDI Notes (0-127)')
    plt.title(title)
    plt.colorbar()
    
    # Add note names for reference
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    y_ticks = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]  # C notes
    y_labels = [f"{note_names[tick % 12]}{tick // 12}" for tick in y_ticks]
    plt.yticks(y_ticks, y_labels)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Process your MIDI file
    midi_path = os.path.realpath('clean_midi/Eminem/Bad Meets Evil.mid')
    matrices = preprocess_midi_for_midinet(midi_path)
    
    print(f"Extracted {len(matrices)} bars")
    print(f"Each bar is a {matrices[0].shape} matrix")
    
    # Visualize the first few bars
    for i, matrix in enumerate(matrices[:3]):
        visualize_matrix(matrix, f"Bar {i+1}")
        
        # Show which notes are active
        active_notes = np.where(matrix.sum(axis=1) > 0)[0]
        print(f"Bar {i+1} contains notes: {active_notes}")