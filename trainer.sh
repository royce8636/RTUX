#!/bin/bash
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <dataset directory> <game name> <start action index> <last action index> <calibration text> <mem>"
  exit 1
else
	DIR=$1
	GAME=$2
	START=$3
	END=$4
	CAL=$5
	echo "On $DIR, app $GAME, $START-$END being trained"
fi

echo "Memory per model: $MEM mb"

run_script() {
    python3 rtux_cnn/create_model.py -g $GAME -a $1 -e 20 -B 128 -C $CAL &
    pids+=($!)
}

handle_ctrl_c() {
    echo "Ctrl+C pressed, terminating all background processes..."
    for pid in "${pids[@]}"; do
        kill -SIGINT "$pid"
    done
    exit 1
}

trap 'handle_ctrl_c' SIGINT

pids=()

start_time=$(date +%s)

for ((a_val = START; a_val <= END; a_val++)); do
    model_name="${GAME}_$a_val"
    echo "$model_name"
    run_script "$a_val" "$model_name"
done

wait

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"

for ((a_val=START; a_val<=END; a_val++));do
	cat dataset/${GAME}/models/${GAME}_$a_val.json
done

