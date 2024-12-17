from music21 import converter, note, chord
import music21

# Set the LilyPond path to the correct location
music21.environment.UserSettings()['lilypondPath'] = r'C:/Program Files/lilypond-2.24.4/bin/lilypond.exe'


def musicxml_to_lilypond(mxml_file):
    # Load the MusicXML file
    score = converter.parse(mxml_file)
    
    # Convert the music to LilyPond format
    lilypond_output = score.write('lilypond')
    
    return lilypond_output

# Example usage:
lilypond_file = musicxml_to_lilypond('ex.mxl')
print(lilypond_file)