#!/bin/bash
# This file is based on android-touch-record-replay by Cartucho.
# Original repository: https://github.com/Cartucho/android-touch-record-replay
# Modifications made by Jaeheon Lee on 2024/12/10.

cleanup() {
    pkill -TERM scrcpy
}

subtract_and_save() {
    local script_file="$1"
    local last_time="$2"

    echo "Subtracting last time: $last_time from timestamps in $script_file..."

    awk -v last_time="$last_time" '
        BEGIN { FS = "[][]" }   # Set the field separator to [ or ]
        {
            # Check if the line contains a timestamp between square brackets
            if ($2 != "") {
                # Subtract the last time from the timestamp and print the modified line
                printf "[%16.6f] %s\n", $2 - last_time, $3
            } else {
                # Print lines that do not contain timestamps
                print
            }
        }
    ' "$script_file" > "$script_file.tmp" && mv "$script_file.tmp" "$script_file"

    echo "Time coordination complete."
}


trap cleanup INT

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <game_name> <activity_name> [serial]"
    exit 1
fi

game_name="$1"
activity_name="$2"
serial="$3"

if [ -z "$serial" ]; then
    if [[ $(adb shell echo) == *"more than one device/emulator"* ]]; then
        echo "More than one device/emulator connected. Please specify the device serial."
        exit 1
    fi
fi

if [ -n "$serial" ]; then
    adb_cmd="-s $serial"
else
    adb_cmd=""
fi

GAME="$1"
ACTIVITY="$2"

CWD=$(pwd)
DATASET="$CWD/dataset/$GAME"
mkdir -p "$DATASET"
echo "Saving files to $DATASET"

cd prep

echo "Looking for touchscreen device..."
TOUCH_DEVICE=$(./find_touchscreen_name.sh $serial)
ANDROID_VERSION_STR=$(adb ${adb_cmd} shell getprop ro.build.version.sdk)
ANDROID_VERSION=$(echo "$ANDROID_VERSION_STR" | tr -d $'\r' | bc)
MIN_VERSION=23

echo "$TOUCH_DEVICE"

QUEUE_FILE="$DATASET/${GAME}_queue.txt"
PACKAGE_NAME=$(echo "$ACTIVITY" | cut -d'/' -f1)
touch $QUEUE_FILE
echo "1: echo" >> $QUEUE_FILE
echo "1: adb ${adb_cmd} shell am force-stop ${PACKAGE_NAME}" >> $QUEUE_FILE
echo "1: adb ${adb_cmd} shell am start -W ${ACTIVITY}" >> $QUEUE_FILE


if [[ "$TOUCH_DEVICE" = *"Touchscreen device found!"* ]]; then
    echo -e "SDK version: $ANDROID_VERSION\n"

    MONOTONIC_BINARY_PATH="/data/local/tmp/getboottime"
    if ! adb ${adb_cmd} shell [ -e "$MONOTONIC_BINARY_PATH" ]; then
        adb ${adb_cmd} push getboottime /data/local/tmp/
    fi

    START_TIME=$(adb ${adb_cmd} shell "$MONOTONIC_BINARY_PATH" | tr -d $'\r')
    echo "Debug: Raw start time from getmonotonic - '$START_TIME'"
    SCRIPT_FILE="$DATASET/${GAME}_${START_TIME}.txt"
    OUTPUT_FILE="$DATASET/${GAME}_${START_TIME}.mp4"
    echo "Debug: Script file - $SCRIPT_FILE"
    echo "Debug: Output file - $OUTPUT_FILE"

    adb ${adb_cmd} exec-out getevent -t "${TOUCH_DEVICE#*-> }" > $SCRIPT_FILE & 

    scrcpy ${adb_cmd} --no-audio --no-playback --record="$OUTPUT_FILE" > scrcpy_log.txt 2>&1 &

    adb ${adb_cmd} shell am start -n "$ACTIVITY"

    echo "Screen Recording $OUTPUT_FILE started. Start your touch events now."

    logcat_output=$(adb ${adb_cmd} logcat -v monotonic -d | grep "AndroidRuntime: Calling main entry com.genymobile.scrcpy.Server")
    echo "$logcat_output"
    scrcpy_start=$(echo "$logcat_output" | awk 'END{print $1}')
    echo "scrcpy start time: $scrcpy_start"

    read -r -p "Press Enter to stop recording and exit... "
    cleanup


    subtract_and_save "$SCRIPT_FILE" "$scrcpy_start"

fi
