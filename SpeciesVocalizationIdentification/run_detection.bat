@echo off
REM Check if librosa is installed
python -c "import librosa" 2>nul || python -m pip install librosa
REM Check if tensorflow is installed
python -c "import tensorflow" 2>nul || python -m pip install tensorflow
REM Check if numpy is installed
python -c "import numpy" 2>nul || python -m pip install numpy
REM Check if soundfile is installed
python -c "import soundfile" 2>nul || python -m pip install soundfile

REM Run the Python script
python detect_species.py

REM Pause to keep the command prompt open
pause