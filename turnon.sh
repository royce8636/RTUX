#!/bin/bash

# Define custom echo function
echo1() {
  script_name=$(basename "${BASH_SOURCE[1]}")  # Get the script name
  printf "[%s]: %s\n" "$script_name" "$@"
}

if [ -n "$1" ]; then
    DEVICE_SERIAL="-s $1"
else
    DEVICE_SERIAL=""
fi

if adb $DEVICE_SERIAL shell dumpsys display | grep -q "mScreenState=OFF"; then
    echo1 "Phone is off, turning it on ..."
    adb $DEVICE_SERIAL shell input keyevent KEYCODE_POWER
    sleep 1
    adb $DEVICE_SERIAL shell input keyevent 82
    echo1 "Phone Unlocked"
else
    echo1 "Phone is already on"
    exit 0
fi

