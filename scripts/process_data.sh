conda activate mobi
cd ${WORK_DIR_MOBI}/bevfusion
NUM_WORKERS=16

# Mini
python ./tools/create_data.py --version v1.0-mini --root-path ${WORK_DIR_MOBI}/data/nuscenes --max-sweeps 0 --out-dir ${WORK_DIR_MOBI}/processed-data/nuscenes-mini --workers ${NUM_WORKERS} --split val   --pbe-database --extra-tag nuscenes nuscenes
python ./tools/create_data.py --version v1.0-mini --root-path ${WORK_DIR_MOBI}/data/nuscenes --max-sweeps 0 --out-dir ${WORK_DIR_MOBI}/processed-data/nuscenes-mini --workers ${NUM_WORKERS} --split train --pbe-database --extra-tag nuscenes nuscenes

# Full set
python ./tools/create_data.py --version v1.0 --root-path ${WORK_DIR_MOBI}/data/nuscenes --max-sweeps 0 --out-dir ${WORK_DIR_MOBI}/processed-data/nuscenes --workers ${NUM_WORKERS} --split val   --pbe-database --extra-tag nuscenes nuscenes
python ./tools/create_data.py --version v1.0 --root-path ${WORK_DIR_MOBI}/data/nuscenes --max-sweeps 0 --out-dir ${WORK_DIR_MOBI}/processed-data/nuscenes --workers ${NUM_WORKERS} --split train --pbe-database --extra-tag nuscenes nuscenes