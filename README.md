Sprite Lighter
==============

#### Normal Map Generator for Pixel Art ####

(name subject to change)

The purpose of this tool is to provide (pixel) artists with a method to automatically generate a normal map for pixel art, to either use directly or as a starting point.

### Usage ###

Currently this tool exists as a python script (web app shall be developed at an undetermined time in the future).
Requires torch (pytorch), numpy, and PIL.

Ensure your input image is named `in.png` or `in.jpg`, and simply run:

```commandline
python3 launch.py
```

The output will be generated as `out.png` in the same directory. Currently the output is in CYMK colour space (to be changed). 