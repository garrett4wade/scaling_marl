WORLD_SIZE=10
rsync -avz . fuwei@192.168.1.102:$PWD
ssh fuwei@192.168.1.102 -i ~/.ssh/id_rsa "nohup $HOME/.conda/envs/torch-py38/bin/python3.8 $PWD/multi_test_remote.py --world_size ${WORLD_SIZE} --offset 4"
python3.8 $PWD/multi_test_local.py --world_size ${WORLD_SIZE} &
