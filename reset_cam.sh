#!/bin/bash

# Check if a camera number is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <camera_number> [device_serial]"
    exit 1
fi

# Set DEVICE_SERIAL if a device serial number is provided
DEVICE_SERIAL=""
if [ -n "$2" ]; then
    DEVICE_SERIAL="-s $2"
fi

# Camera to be reset
cam="$1"

# Constants for brightness and contrast (unused in this snippet but kept for potential future use)
BRIGHTNESS=10
CONTRAST=50

# Flag to indicate if a matching camera has been found
camera_found=0

# Search through USB devices
for dir in /sys/bus/usb/devices/*; do
    if [ -d "$dir" ] && [ -e "$dir/product" ]; then
    	# echo "Checking $dir"
		# Check if the product matches "Arducam OV9281 USB Camera"
        if [ "$(cat "$dir/product" 2>/dev/null)" = "Arducam OV9281 USB Camera" ]; then
            echo "Found Arducam OV9281 USB Camera in $dir"
			base_dir=$(basename "$dir")
            video_dir="$dir/$base_dir:1.0/video4linux"
            
            if [ -e "$video_dir" ]; then
                video_output=$(ls "$video_dir" 2>/dev/null)
                # echo "Found video device: $video_output"
                # Check if the camera number matches
                if echo "$video_output" | grep -q "video$cam"; then
                    camera_found=1
                    echo "$(basename "$dir")"
                    
                    # Reset the camera
                    echo 1 | sudo tee "$dir"/authorized > /dev/null
                    # check if the camera is re-authorized
					if [ "$(cat "$dir"/authorized)" = "1" ]; then
						echo "Camera re-authorized."
					else
						echo "Failed to re-authorize the camera."
						continue
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

                    # Re-authorize the camera
                    echo 1 | sudo tee "$dir"/authorized > /dev/null
					
					# check if the camera is re-authorized
					if [ "$(cat "$dir"/authorized)" = "1" ]; then
						echo "Camera re-authorized."
					else
						echo "Failed to re-authorize the camera."
						continue
					fi
                fi
            fi
        fi
    fi
done

if [ $camera_found -eq 0 ]; then
    echo "No matching camera found."
else
    echo "Reset done."
fi

