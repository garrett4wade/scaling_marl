pkill -9 Main_Thread && sleep 0.5
pkill -9 python && sleep 0.5
pkill -9 run_node && sleep 0.5
pkill -9 torch && sleep 0.5
rm -rf core.* && sleep 0.5
rm -rf /dev/shm/* && sleep 0.5
rm -rf /tmp/* && sleep 0.5

config="configs/starcraft2/config.yaml"
python run_sf_worker_node_on_smac.py --config ${config} --node_idx $@

pkill -9 Main_Thread && sleep 0.5
pkill -9 python && sleep 0.5
pkill -9 run_node && sleep 0.5
pkill -9 torch && sleep 0.5
rm -rf core.* && sleep 0.5
rm -rf /dev/shm/* && sleep 0.5
rm -rf /tmp/* && sleep 0.5
