import argparse
import subprocess
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datetime import datetime


def set_gpu_memory_limit(memory_limit):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print(f"Set GPU memory limit to {memory_limit}MB")
        except RuntimeError as e:
            print(e)


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--device', type=str, default=None,
                        help='serial number of the device to pass arguments to')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='Specify which camera to take video from (num from /dev/video)')
    parser.add_argument('-C', '--calibrate', type=str, default=None,
                        help='Calibration file from br_calib.py')
    parser.add_argument('-s', '--show', action='store_true',
                        help='Whether to show frames or not')
    parser.add_argument('-l', '--log_path', type=str, default="logs/" + datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
                        help='Path of log folder (default is logs/[timestamp], enter [timestamp] after desired path '
                             'to use time as basename of the dir)')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='File of order to run if it exists')
    parser.add_argument('-a', '--all', nargs='?', const=True, type=int,
                        help='All frames will be saved with no argument given, otherwise the number of frames to save '
                             'before detection')
    parser.add_argument('-r', '--repeat', type=int, default=1,
                        help='How many times to repeat the task')
    parser.add_argument('-t', '--threshold', type=str, default="dataset/threshold.json",
                        help='Threshold file to use')
    parser.add_argument('-m', '--memory', type=int, default=None,
                        help='How much memory to allocate to the GPU')
    parser.add_argument('-S', '--strict', action='store_true',
                        help='Strict mode: removes every abnormal turn ')
    parser.add_argument('-p', '--perfetto', type=str, required=True,
                        help='Perfetto trace configuration file')
    return parser.parse_args(argv)


def get_adb_devices():
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        output = result.stdout.strip().split('\n')
        devices = []
        for line in output[1:]:
            if 'device' in line:
                device_info = line.split('\t')
                devices.append(device_info[0])
        if len(devices) == 0:
            print("No adb devices found")
            exit(1)
        elif len(devices) > 1:
            print(f"Multiple adb devices found, using the first device {devices[0]}\n"
                  f"Provide the device with -D to use an other device")
        return devices[0]
    except FileNotFoundError:
        print("No adb devices found")
        exit(1)

def run_sync_android(serial):
    result = subprocess.run(['./sync_linux', serial], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running sync_android")
        exit(1)
    patterns = {
        'ANDROID_Realtime': r'ANDROID_Realtime:\s+(\d+\.\d+)',
        'ANDROID_Boottime': r'ANDROID_Boottime:\s+(\d+\.\d+)',
        'LINUX_Realtime': r'LINUX_Realtime:\s+(\d+\.\d+)',
        'LINUX_Boottime': r'LINUX_Boottime:\s+(\d+\.\d+)'
    }
    parsed_values = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, result.stdout)
        if match:
            parsed_values[key] = float(match.group(1))
        else:
            print(f"{key} not found in the output. Please make sure all files are compiled")
            exit(1)
            
    print("SYNC DONE")
    return parsed_values

if __name__ == "__main__":
    import tensorflow as tf

    args = get_args()
    args.directory = 'dataset'
    if args.memory is not None:
        set_gpu_memory_limit(args.memory)

    from rtux_cnn.queue_handler import QueueHandler
    from rtux_cnn.config import Config

    if os.path.exists("output.log"):
        os.remove("output.log")
        os.system("touch output.log")

    if args.memory is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=args.memory)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

    tf.keras.utils.disable_interactive_logging()  # Remove [==================] prediction logs

    if isinstance(args.all, int) and args.all > 120:
        print("Too many frames to log, setting to 120. Otherwise, just save all frames by providing no argument")
        args.all = 120

    device = f"{get_adb_devices()}" if args.device is None else f"{args.device}"
    print("Using device: " + device)

    touch_screen = subprocess.run(['./prep/find_touchscreen_name.sh', device], capture_output=True, text=True)
    if "not found" in touch_screen.stdout:
        print("Touch screen not found")
        exit(1)
    touch_screen = touch_screen.stdout.strip()[-1]
    print(f"Touch screen: {touch_screen}")
 
    Config(cap_ind=args.camera, device=device, calib=args.calibrate, cv_window=args.show)

    os.system(f"./turnon.sh {device}")
    max_br = Config().get_max_br()

    # os.system(f'adb -s {device} shell settings put system screen_brightness {int(max_br)}')

    # turn off auto rotate
    os.system(f"adb -s {device} shell content insert --uri content://settings/system --bind name:s:accelerometer_rotation --bind value:i:0")

    shape = 128

    parsed_sync = run_sync_android(device)
    time_diff = parsed_sync['LINUX_Boottime'] - parsed_sync['ANDROID_Boottime']
    time_info = {
        "time_diff": time_diff,
        "ANDROID_Realtime": parsed_sync['ANDROID_Realtime'],
        "ANDROID_Boottime": parsed_sync['ANDROID_Boottime'],
        "LINUX_Realtime": parsed_sync['LINUX_Realtime'],
        "LINUX_Boottime": parsed_sync['LINUX_Boottime']
    }


    QueueHandler(args, (None, 128, 128, 1), time_info, touch_screen)
