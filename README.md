# Entropy Formation README
*Written by Jason Carnahan*

## Getting Started *(with AirSim)*
Follow the documentation here (https://microsoft.github.io/AirSim/build_windows/) to get AirSim installed on your machine
with Unreal Engine.

Make sure to have Unreal Engine up and running with AirSim, and specify where to save the `settings.json` in the `AirSimStrategy.py`
under `UNIQUE_SAVE_LOCATION` (typically `%user%/documents/airsim/settings.json`) before launching the python program.

To use this project (assuming you have already installed the dependencies in `requirements.txt`), go to `main.py` and
launch the program. A GUI should appear and ask for the following data:
* What Strategy to use
* The number of UAVs in the system
* The Separating distance between the UAVs
* The Length of a row for the UAVs as they are placed in grid formation

All errors in the user entry field will appear in the python console, but, in general, the entry values must be positive
integers. With those fields given, clicking `Submit` will attempt connection to the simulation.

The console will show a red error like `WARNING:tornado.general:Connect error on fd 1940: WSAECONNREFUSED`. This is the 
program trying to connect to AirSim. Go over to Unreal Engine and click the play button in the top. This will load the JSON
we defined earlier, and the simulation should run as expected!

