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


exec 1>&2

ADB=$(which adb)
adb() {
  $ADB $* </dev/null
}

send_key() {
  adb ${adb_cmd} shell input keyevent $1
  if [ ! -z "$2" ]; then sleep $2; fi
}

RED='\033[0;31m'
BROWN='\033[0;33m'
NC='\033[0m'

log() {
  printf "${BROWN}$*${NC}\n"
}

err() {
  printf "${RED}$*${NC}\n"
}

get_mw() {
  TMP=$(adb ${adb_cmd} exec-out cat /sys/class/power_supply/battery/voltage_now /sys/class/power_supply/battery/current_now | tr '\n' ' ')
  VOLT=$(echo $TMP | awk '{print $1}')
  AMP=$(echo $TMP | awk '{print $2}')
  mW=$((($VOLT * $AMP * -1) / 1000000000))
  echo $mW
}

IDLE=2500 # mW
CONSISTENT=10
SLEEP=0.2

# We will wait for device to consume $IDLE for $(($CONSISTENT * $SLEEP))

wait_for_idle() {
  start_time=$(date +%s%3N)

  val=()
  idx=0
  for i in $(seq 0 $(($CONSISTENT - 1))); do
    val[$i]=9999
  done
  while true; do
    val[$(($idx % $CONSISTENT))]=$(get_mw)
    sleep $SLEEP
    ret=0
    printf '\r\033[2K'
    printf "${BROWN}Waiting to enter idle${NC} "
    for i in $(seq 0 $(($CONSISTENT - 1))); do
      echo -n "${val[$i]} "
      #echo "val[$i] = ${val[$i]}"
      if [[ ${val[$i]} -gt $IDLE ]]; then
        #echo "val[$i] is over $IDLE by $((${val[$i]} - $IDLE))"
        ret=1
      fi
    done
    if [[ "$ret" == "0" ]]; then break; fi
    idx=$(($idx + 1))
  done

  end_time=$(date +%s%3N)
  log "- took $((($end_time - $start_time) / 1000)).$((($end_time - $start_time) % 1000))s"
}

adb ${adb_cmd} shell am start -W in.binarybox.blackscreen/in.binarybox.blackscreen.MainActivity

wait_for_idle

send_key KEYCODE_HOME
