#!/bin/bash
# This file is based on android-touch-record-replay by Cartucho.
# Original repository: https://github.com/Cartucho/android-touch-record-replay
# Modifications made by Jaeheon Lee on 2024/12/10.

serial="$1"

if [ -z "$serial" ]; then
    device_count=$(adb devices | grep -v "List of devices" | grep "device" | wc -l)
    if [ "$device_count" -gt 1 ]; then
        echo "More than one device/emulator connected. Please specify the device serial."
        exit 1
    fi
fi

if [ -n "$serial" ]; then
    adb_cmd="-s $serial"
else
    adb_cmd=""
fi

for line in `adb ${adb_cmd} shell getevent -lp 2>/dev/null | egrep -o "(/dev/input/event\S+)"`; do
  output=`adb ${adb_cmd} shell getevent -lp $line`
  [[ "$output" == *"ABS_MT"* ]] && { echo "Touchscreen device found! -> $line"; exit; }
done
echo "Touchscreen not found!"
