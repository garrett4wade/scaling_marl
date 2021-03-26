pkill -9 Main_Thread
sleep 0.5
pkill -9 python3.8
sleep 0.5
pkill -9 rmappo
sleep 0.5

rm /dev/shm/smac_rpc
rm /dev/shm/smac_ddp
