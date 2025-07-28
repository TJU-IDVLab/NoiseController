NUM_PROCESSES=$1
PY_ARGS=${@:2}
LAUNCH_PARAM=${LAUNCH_PARAM:-"--num_processes ${NUM_PROCESSES} --num_machines 1 --main_process_port 10251"}
export MASTER_PORT=29501
accelerate launch --mixed_precision fp16  \
    ${LAUNCH_PARAM} tools/train.py ${PY_ARGS}
