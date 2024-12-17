import zipfile
import xml.etree.ElementTree as ET

# Function to parse the MusicXML file and extract notes
def extract_notes_from_mxl(mxl_file):
    # Unzip the MXL file
    with zipfile.ZipFile(mxl_file, 'r') as zip_ref:
        # Find and read the MusicXML file inside the MXL archive
        for file_name in zip_ref.namelist():
            if file_name.endswith('.xml'):
                musicxml_file = file_name
                break

        # Extract the XML content
        xml_content = zip_ref.read(musicxml_file)
    
    # Parse the XML content
    root = ET.fromstring(xml_content)
    
    notes = []
    
    # Iterate through all <note> elements in the MusicXML
    for note in root.iter('note'):
        pitch = None
        duration = None
        
        # Extract pitch
        pitch_element = note.find('pitch')
        if pitch_element is not None:
            step = pitch_element.find('step').text if pitch_element.find('step') is not None else ''
            octave = pitch_element.find('octave').text if pitch_element.find('octave') is not None else ''
            pitch = step + octave if step and octave else None
        
        # Extract duration
        duration_element = note.find('duration')
        if duration_element is not None:
            duration = duration_element.text
        
        if pitch and duration:
            notes.append((pitch, duration))
    
    return notes

# Example usage
mxl_file_path = 'ex.mxl'
notes = extract_notes_from_mxl(mxl_file_path)

# Print the order of notes (pitch and duration)
print("Notes in order:")
for pitch, duration in notes:
    print(f"Note: {pitch}, Duration: {duration}")