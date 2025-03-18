#!/bin/bash
# This file is based on android-touch-record-replay by Cartucho.
# Original repository: https://github.com/Cartucho/android-touch-record-replay
# Modifications made by Jaeheon Lee on 2024/12/10.

if [ -z "$4" ]; then
    echo "Usage: $0 <name of the app> <index> <touch_device> <device serial>"
    exit 1
fi
GAME="$1"
IND="$2"
TOUCH_DEVICE="$3"
DEVICE_SERIAL="$4"
TOUCH_DEVICE="/dev/input/event$TOUCH_DEVICE"

# Check if the file `mysendevent` exists in the phone (it's a binary that sends touch events)
MYSENDEVENT=`adb -s ${DEVICE_SERIAL} shell ls /data/local/tmp/mysendevent 2>&1`
echo --- START playing "/data/media/0/temp_pic/${GAME}/script/${IND}.txt"---
[[ "$MYSENDEVENT" == *"No such file or directory"* ]] && adb -s ${DEVICE_SERIAL} push mysendevent /data/local/tmp/

adb -s ${DEVICE_SERIAL} shell /data/local/tmp/mysendevent "${TOUCH_DEVICE#*-> }" /data/media/0/temp_pic/${GAME}/script/${IND}.txt

echo --- END playing "/data/media/0/temp_pic/${GAME}/script/${IND}.txt"---