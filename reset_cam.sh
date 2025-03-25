#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <camera_number> [device_serial]"
    exit 1
fi

DEVICE_SERIAL=""
if [ -n "$2" ]; then
    DEVICE_SERIAL="-s $2"
fi

cam="$1"
video_device="/dev/video${cam}"

if [ ! -e "$video_device" ]; then
    echo "Device $video_device does not exist."
    exit 1
fi

device_sysfs=$(udevadm info -q path -n "$video_device")

usb_device_path=$(dirname "/sys${device_sysfs}")
while [[ "$usb_device_path" != "/" && ! -f "${usb_device_path}/product" ]]; do
    usb_device_path=$(dirname "$usb_device_path")
done

if [[ -f "${usb_device_path}/product" ]]; then
    camera_product_name=$(cat "${usb_device_path}/product")
    echo "Using camera: $camera_product_name"
else
    echo "Could not find camera product name."
    exit 1
fi

echo 1 | sudo tee "$usb_device_path/authorized" > /dev/null

if [ "$(cat "$usb_device_path/authorized")" = "1" ]; then
    echo "Camera re-authorized."
else
    echo "Failed to re-authorize the camera."
    exit 1
fi

if adb $DEVICE_SERIAL shell am force-stop com.example.vlcrtux; then
    echo "VLCRTUX stopped successfully."
else
    echo "Failed to stop VLCRTUX."
fi

adb $DEVICE_SERIAL shell am start -W com.example.vlcrtux/.MainActivity
adb $DEVICE_SERIAL shell am broadcast -a com.example.vlcrtux.action.WHITE -n com.example.vlcrtux/.ControlReceiver

sleep 2

echo "Reauthorizing the camera..."
echo 1 | sudo tee "$usb_device_path/authorized" > /dev/null

if [ "$(cat "$usb_device_path/authorized")" = "1" ]; then
    echo "Camera re-authorized."
else
    echo "Failed to re-authorize the camera."
    exit 1
fi

echo "Reset done."

