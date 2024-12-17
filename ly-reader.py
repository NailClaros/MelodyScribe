import re

# Read the .ly file
with open('ex-from-mxl.ly', 'r') as file:
    lines = file.readlines()

# Regular expression to match note patterns in the LilyPond file
note_pattern = r'\s[a-gA-G][b#]*\'*\d*'

# Iterate through each line and extract notes
for line_num, line in enumerate(lines, 1):
    notes = re.findall(note_pattern, line)
    if notes:  # Only print lines that contain notes
        if len(notes) <= 2:
            print(f"Line {line_num}: key = {' '.join(notes)}")
        else:
            print(f"Line {line_num}: {' '.join(notes)}")