#!/bin/bash
arr=( 70 )
j=0
for i in "${arr[@]}"; do
    ssh node$i "(docker exec scaling bash -c \"nohup python3 -u /workspace/workspace/scaling_marl/run_learner_node.py --config=/workspace/workspace/scaling_marl/configs/starcraft2/config.yaml --learner_node_idx $j >> /workspace/workspace/scaling_marl/log/trainer.log 2>&1 &\")"
    ((j=j+1))
done

arr=( 71 )
j=0
for i in "${arr[@]}"; do
    ssh node$i "(docker exec scaling bash -c \"nohup python3 -u /workspace/workspace/scaling_marl/run_worker_node_on_smac.py --config=/workspace/workspace/scaling_marl/configs/starcraft2/config.yaml --worker_node_idx $j >> /workspace/workspace/scaling_marl/log/worker.log 2>&1 &\")"
    ((j=j+1))
    sleep 0.5
done

ssh node70 "(docker exec scaling bash -c \"nohup python3 -u /workspace/workspace/scaling_marl/run_monitor.py --config=/workspace/workspace/scaling_marl/configs/starcraft2/config.yaml --name learner0 --is_head >> /workspace/workspace/scaling_marl/log/monitor.log 2>&1 &\")"
ssh node71 "(docker exec scaling bash -c \"nohup python3 -u /workspace/workspace/scaling_marl/run_monitor.py --config=/workspace/workspace/scaling_marl/configs/starcraft2/config.yaml --name worker0 >> /workspace/workspace/scaling_marl/log/monitor.log 2>&1 &\")"

