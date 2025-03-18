# Beyond the Numbers: Measuring Android Performance Through User Perception

**This is the open-source repository of RTUX, introduced in the paper presented in ISPASS'25**

## Abstract
Android, with its vast global adoption and diverse hardware ecosystem, poses unique challenges for performance benchmarking, particularly from a user-centric perspective. Traditional benchmarks often fail to capture the intricacies of user-perceived performance, relying on component-level metrics or synthetic workloads that do not reflect real-world usage. This paper proposes Real-Time User-Experience, RTUX, a novel benchmarking tool designed to measure Android system performance as perceived by users. RTUX employs external camera-based GUI state recognition and scenario-based testing to evaluate app load-times and in-app transitions under diverse conditions. Using CNN models and a unique system structure, RTUX reliably replays human-like interactions, enabling repeatable and robust performance assessments. Through experiments with 100 scenario repetitions involving popular Android apps, we uncover some system bottlenecks, such as suboptimal writeback configurations and I/O scheduler inefficiencies. The tool demonstrates how targeted optimizations can yield tangible improvements in user experience.

**Note: For quick start, download the models from [zenodo](www.zenodo.com), then jump to [UXB](#user-experience-benchmark-uxb). TRTModel might need to be recompiled based on the GPU.**

## Hardware Requirements

### Testing Suite:
- 3D printed [clamp system](/benchmark_suite_stls/)
- Android phone ([Snapdragon 888 HDK](https://www.thundercomm.com/product/snapdragon-888-mobile-hardware-development-kit/))
- Phone mount ([Ulanzi ST-02S](https://www.ulanzi.com/products/st-02s-phone-tripod-mount-ulanzi-0849?srsltid=AfmBOooc5W8Iicvjru_7bXeY8rilLFX3pwuqg84wT1xbrPrYcV1kROUQ) used)
- USB Camera (Arducam OV9281 used) 

**Note:** The USB camera is fixed to the 3D printed clamp system using the M2.5x30 bolts and spacers 

### Controlling Computer:
- No specific requirements for the controlling computer. A GPU is recommended for real-time inference at 120FPS (~8.3ms per frame). Otherwise, a high-performance CPU can be used.
- Large enough SSD space for traces (>500GB for 100 iterations if saving all photos)

---

## State Separator (SS)
RTUX uses CNN models to detect the stages of the app on the phone. In order to create the CNN models, three steps are required: (A) State Separation (B) Dataset Collection (C) Model Training 

### (A) State Separation
This step is where the user chooses the app that will be tested, separate the app into multiple different stages (e.g., menu → loading → game). In below scripts, `<device serial>` is serial number of the device that can be found on `adb shell list`. This field can be omitted if only one android device is connected to the controlling computer. Please follow the videos if unsure.

1. Find the activity of the app that is to be tested ([Demo video](https://youtu.be/sV2wAX-GAkM))
    - Activity is the main launch point of the app (e.g.,  `com.fingersoft.hillclimb/com.fingersoft.game.MainActivity`)
    - `adb shell am stack list` lists the packages running applications. Some apps will have MainActivity listed from this, but some might not.
    - If activity couldn't be found using above command, use `adb shell dumpsys package | grep <package> | grep Activity`, which will list the available activities. One with MainActivity is most likely the main screen that will be launched. Some apps may have different startup activities, so check for launchable ones.
2. ` ./prep/record_touch_vid.sh <app name> <activity> <device serial>` ([Demo video](https://youtu.be/sV2wAX-GAkM))
    - App name: Name of the app that will be saved to the `dataset/` directory
    - Activity: Activity of the app that is to be tested (Found on step 1)
3.  Once `record_touch_vid.sh` prints `Press Enter to stop recording and exit...` start using the app on the phone (physical touch input)
4. After all desired input has been made, press enter to stop the script ([Demo video](https://youtu.be/sV2wAX-GAkM))
    - This would have created a folder in `/dataset/<app name>` with three files
5. `python3 prep/state_separator.py -g <game> -d <device serial>` ([Demo video](https://youtu.be/iRFNhI32NYI))
    - GUI will open with the video (if multiple videos are in the directory, the user has to manually give the video). The user can then use this to mark each 'states'. The states are start and end of each stage. Starting from 0, press the number keys to mark the start and end of each 'state'. 
    - This creates `screen/` for videos of each stage, `script/` for input script of each stage, `image/` of all frames, and a `<app name>_queue.txt` scenario file

### (B) Dataset Collection
In this step, RTUX automatically collects the photos of the app by changing the brightness and position of the frames. It takes `(# of desired photo / FPS)` seconds to complete. In this step, camera connection to the controlling computer, and camera index (e.g., `/dev/video0`, 0 is the index) is required. Make sure the entire screen of the phone is visible inside the camera's field of view.

1. `python3 rtux_cnn/calibrator.py -o <out file name> -c <camera index>` ([Demo video](https://youtu.be/R92eGvli7UE))
    - Ensure that the phone is within the camera's frame 
    - This automatically finds the phone screen, and calibrates the camera settings, and outputs the calibrated settings to a `<out file name>`
2. Install VLCRtux App
    - This app is required to replay the separated videos for dataset collection
    - Pre-built app is located in [`/VLCRtux/VLCRtux.apk`](/VLCRtux/app-debug.apk), and can be installed with `adb install /VLCRtux/VLCRtux.apk`
    - Once installed, open the app, and allow the access to files
3. Repeated image taking for data collection ([Demo video](https://youtu.be/LYC1i8iTXhA))
```
python3 prep/dataset_collection.py -g <game> -C <calibration file>
                                   [-r ] # Toggle resume (will resume from existing images and if not given, replace existing)
                                   [-s <starting index>] # Starting index of app state (default: 0)
                                   [-e <ending index>] # End index of app state (default: all existing indexes in the directory)
```

### (C) Model Training
```
python3 rtux_cnn/create_model.py -C <calibration file> -g <app name> -a <state number> 
                                 [-m <Max GPU memory>]
                                 [-e <epoch>]
                                 [-t] # Convert to TensorRT (Only if using Nvidia GPU)
```
- This creates the model for given `<app name>` in the name format of `<app name>_<state number>`

The process of model training has to be done for each state in each app. Use `trainer.sh` script to train all states in an app


## User-Experience Benchmark (UXB)
The Android phone can now be benchmarked using the models and scenario scripts created from above. There are some setups that has to be done on the android device being tested before. ([Demo video](https://youtu.be/99ocM8-5nS0))

### (A) Setup
1. Scenario creation
    - Create a `.txt` file that consists of what actions should be done in what order. The example of this is given in this [sample scenario](/dataset/multiqueue.txt)
    - `0: `indicates detection tasks, where prefix is app name, suffix is the state decided in state separation step
    - `1: `indicates action tasks such as starting an app or force-stopping an app
    - RTUX will follow the order of the tasks in this scenario file. On action (`1:`) tasks, when it completes the action, it continues to the next line, and on detection (`0:`) tasks, it continues when the given state is detected on the screen
2. Build sync_linux.c and sync_android.c
    - `gcc sync_linux.c -o sync_linux`
    - `clang --target=aarch64-linux-android34 -fPIE -pie sync_android.c -o sync_android`
    - Adjust the Android version and compiler arguments to match the target device. Make sure the `-o
    ` outputs are in the given names

3. Build replayer
    - `clang --target=aarch64-linux-android34 -fPIE -fPIC -o mysendevent prep/replay.c -pie`

4. Calibrate the cameras again in case of lighting or position changes
    - `python3 rtux_cnn/calibrator.py -o <out file name> -c <camera index>` ([Demo video](https://youtu.be/R92eGvli7UE))

### (B) Execution
```
python3 rtux_main.py -C <calibration file> -r <number of iterations> -f <scenario file> 
                     [-c <camera index> (default: 0)] 
                     [-D <device serial> (default: first detected device)] 
                     [-a [<N>]]  # Save N images before detection (all if given without N, none if omitted)
                     [-m <GPU memory to use>] 
                     [-S]  # Toggle strict mode to skip deadlocks and redo the iteration

```

### (C) Result Parsing
```
python3 rtux_cnn/yaml_parser.py -d <log directory> -q <scenario file>
                                [-v] # Verbose for debugging
                                [-b] # Create backup of original file structure
```

## Context-aware System Analyzer (CSA)

RTUX CSA is built on top of Perfetto v42.x with some modifications to show RTUX components. Clone the https://github.com/royce8636/perfetto repository to use the CSA

**Note:** Please refer to the [Perfetto docs](https://perfetto.dev/docs/contributing/build-instructions) if more detailed instructions are needed

### (A) Setup
1. The log data has to be parsed for perfetto processor to understand ([Demo video](https://youtu.be/R92eGvli7UE))
    - `python3 rtux_cnn/prepare_data_perfetto.py -d <log folder directory>` 
    - This creates `all_summary_hashed.json` file for each index in given log folder directory
2. From the perfetto folder, build Perfetto UI and trace processor (refer to perfetto docs if unsuccessful)
    - `tools/install-build-deps --ui`
    - `tools/gn args out/android`
    - `tools/ninja -C out/android`
3. Create a soft link between the RTUX logs folder and the perfetto assets folder
    - `ln -s <RTUX directory>/logs/ <perfetto directory>/ui/src/assets`

### (B) Execution
These are done in perfetto folder ([Demo video](https://youtu.be/BZCtQS2Dxmc))
1. `/out/android/trace_processor_shell --httpd -http-port=9201 /<RTUX directory>/logs/<desired log>/<index number>/perfetto_trace_<index number>`
    - This will load the trace to port 9201
2. On a new terminal, run `ui/run-dev-server --rtux-path <path to RTUX folder>`
    - After it finishes, the loaded trace can be viewed in this link http://localhost:10000/#!?rpc_port=9201
---

