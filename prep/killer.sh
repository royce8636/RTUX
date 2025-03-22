#!/bin/bash

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

apps=$(adb ${adb_cmd} shell "pm list packages -3 | cut -f 2 -d ':'")

for app in $apps; do
    echo "Killing $app"
    adb ${adb_cmd} shell "am force-stop $app"
done

adb ${adb_cmd} shell "killall android.process.media"
adb ${adb_cmd} shell "rm -rf /mnt/sdcard/Download/*"

echo "All background apps have been killed."

adb ${adb_cmd} shell "sync"
adb ${adb_cmd} shell 'echo 3 > /proc/sys/vm/drop_caches'
adb ${adb_cmd} shell 'echo 1 > /proc/sys/vm/compact_memory'

echo "Caches have been cleared."

adb ${adb_cmd} shell content insert --uri content://settings/system --bind name:s:accelerometer_rotation --bind value:i:0
# adb ${adb_cmd} shell settings put system user_rotation 0