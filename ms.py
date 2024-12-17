import subprocess
import lilypond

# Path to the LilyPond executable
lilypond_path = lilypond.executable()

# Specify the input `.ly` file
input_file = "ex-from-mxl.ly"

# Run the LilyPond command to process the file
subprocess.run([str(lilypond_path), input_file])

