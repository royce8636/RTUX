# Instructions for RTUX Quickstart

## Installation

1. Clone this repository
2. Download the model files from Zenodo archive (https) to the `dataset/` directory
3. Unzip the given zip files
    - Make sure that the `dataset/` directory holds 3 folders (clashofclans, hillclimb, minecraft) and 2 files (multiqueue.txt, thresholds.json)

## Setup

**Before proceeding, please make sure that USB Debugging is enabled in Developer Options in the phone being tested. After it has been enabled and connected to the computer, using `adb devices`, check if your phone has been connected properly.**

### (A) Camera Setup
1. Connect the camera that will be used for benchmarking to the controlloing computer, and get the index of the camera
    - `ls /dev/video*` lists the available cameras. The index of the camera is the first one that is listed, which will be 0 (`ls /dev/video0`) in most cases
2. Allow camera to be accessible by doing `sudo chmod 666 /dev/video<index>`

### (A) Script Setup
**Note: Script setup is not needed if using SDK level 34**
1. Build `sync_linux.c`
    - `gcc sync_linux.c -o sync_linux`
2. Build `sync_android.c`
    - `clang --target=aarch64-linux-android34 -fPIE -pie sync_android.c -o sync_android`
3. Build `replay.c`
    - `clang --target=aarch64-linux-android34 -fPIE -fPIC -o mysendevent prep/replay.c -pie`

**Replace android34 with corresponding SDK version, which can be found by doing `adb shell getprop ro.build.version.sdk` like `android<sdk>`**

### (B) Model Setup
**Note: Model setup is not needed if using GPU with compute capability of 8.9 (e.g., RTX 4090)**
1. Convert each given models into TRT
    - `python3 prep/trt_convert.py -d dataset/<game name>/models/<game name>_<index>`
    - This has to be done for each index (e.g., 0, 1, 2, 3, 4) for each game (i.e., clashofclans, hillclimb, minecraft)

### (C) VLCRtux Setup
1. VLCRtux app has to be installed on the Android device being tested
    - `adb install /VLCRtux/VLCRtux.apk`
    - If app does not work or open due to different SDK level, please recompile the Android app after changing the compileSdk on `VLCRtux/app/build.gradle`

### (D) App Setup
The Quickstart uses multiqueue.txt scenario, which assumes that games **Hill Climb Racing, Clash of Clans, and Minecraft Trial** has all been installed on the device and is setup, so that once it is opened, it can be played immediately (login, initial setup and etc should all be done)

- App versions:
    - Hill Climb Racing: 1.65.0
    - Clash of Clans: 17.18.13
    - Minecraft Trial: 1.21.62.01
## Execution
1. Calibrate the cameras ([Demo video](https://youtu.be/R92eGvli7UE))
    - `python3 rtux_cnn/calibrator.py -o hdk.txt -c 0`
2. Run `rtux_main.py` ([Demo video](https://youtu.be/99ocM8-5nS0))
    - `python3 rtux_main.py -c 0 -C hdk.txt -f dataset/multiqueue.txt -a -r 101 -p perfetto_configs/extensive_config.pbtx`
    - Reduce `-r` to reduce the number of iterations done
    - Please include a flag `-D <device serial from adb devices>` if there are multiple Android devices connected to the controlling computer.
## Expected Outputs
At the end of all iterations or after force-stopping and cleanup (CTRL-C), RTUX will report different metrics. The expected outputs are as follow:
```
[QueueHandler/cleanup]  Number of runs: 103
[QueueHandler/cleanup]  Abnormal runs: 3
[QueueHandler/cleanup]  Successful runs: 100
[QueueHandler/cleanup]  Total time: 14034s
```

If `rtux_main.py` was interuptted with CTRL-C, there may be 10 seconds of delay to wait for cleaning up. If CTRL-C is spammed, above will not be reported and force exited.
