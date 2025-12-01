#!/bin/bash

PARAMS=(
	"--max_seed 1 --transfer_enabled False --restricted_communication False --env_name football"
	"--max_seed 1 --transfer_enabled True --restricted_communication False --env_name football"
	"--max_seed 1 --transfer_enabled True --restricted_communication True --env_name football"
)

LOG_FILE="LOG.log"
FINISHED_EXPERIMENTS_FILE="experiments.txt"


for param in "${PARAMS[@]}"; do

    xvfb-run -a -s "-screen 0 1400x900x24" \
    poetry run python3 src/main.py $param >> "$LOG_FILE" 2>&1

    echo "Experiment $param completed" >> "$FINISHED_EXPERIMENTS_FILE"

done
