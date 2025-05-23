conda activate mobi

# Mini
python ./tools/create_data.py --version v1.0-mini --root-path ./data/nuscenes-mini --max-sweeps 0 --out-dir ./data/nuscenes-mini --workers 16 --split val   --pbe-database --extra-tag nuscenes nuscenes
python ./tools/create_data.py --version v1.0-mini --root-path ./data/nuscenes-mini --max-sweeps 0 --out-dir ./data/nuscenes-mini --workers 16 --split train --pbe-database --extra-tag nuscenes nuscenes

# Full set
python ./tools/create_data.py --version v1.0 --root-path ./data/nuscenes --max-sweeps 0 --out-dir ./data/nuscenes --workers 16 --split val   --pbe-database --extra-tag nuscenes nuscenes
python ./tools/create_data.py --version v1.0 --root-path ./data/nuscenes --max-sweeps 0 --out-dir ./data/nuscenes --workers 16 --split train --pbe-database --extra-tag nuscenes nuscenes