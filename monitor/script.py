# ssh node$i '(docker exec container$i bash -c "nohup python3 -u $dir --type $type --config=$configdir --worker_node_idx $widx --trainer_node_idx $tidx >> $logdir 2>&1 &")'
# import subprocess
import os
import time

def ssh_(node, cmd):
    return "ssh %s \"(%s)\"" % (node, cmd)

def docker_(container, cmd):
    return "docker exec %s bash -c \\\"%s\\\"" % (container, cmd)

def cmd_(maindir, configdir, logdir, type_, idxvar, args = None): # args for other variables, should be a dict
    if type_ == "worker":
        return "nohup python3 -u %s --config=%s --worker_node_idx %s >> %s 2>&1 &" \
                    % (maindir, configdir, idxvar, logdir)
    elif type_ == "trainer":
        return "nohup python3 -u %s --config=%s --learner_node_idx %s >> %s 2>&1 &" \
                    % (maindir, configdir, idxvar, logdir)    
    elif type_ == "monitor":
        if args["is_head"]:
            return "nohup python3 -u %s --config=%s --name %s --is_head >> %s 2>&1 &" \
                        % (maindir, configdir, args["name"], logdir)
        else:
            return "nohup python3 -u %s --config=%s --name %s >> %s 2>&1 &" \
                        % (maindir, configdir, args["name"], logdir)
        
        
def loop_block(cmds, vars_):
    head = "for %s in %s; do\n" % vars_
    body = ""
    for cmd in cmds:
        body += "    " + cmd + "\n"
    tail = "done\n"
    return head + body + tail

def trainer_block(trainerdir, configdir, trainer_logdir, trainers, container_name):    
    trainer_block = "arr=( "
    trainers = trainers.split(",")
    for t in trainers:
        trainer_block += t + " "
    trainer_block += ")\n"
    trainer_block += "j=0\n"
    trainer_cmd = cmd_(trainerdir, configdir, trainer_logdir, "trainer", "$j")
    cmds = [ssh_("node$i", docker_(container_name, trainer_cmd)), "((j=j+1))"]
    trainer_block += loop_block(cmds, ("i", "\"${arr[@]}\""))
    return trainer_block

def worker_block(workerdir, configdir, worker_logdir, workers, container_name):
    worker_block = "arr=( "
    workers = workers.split(",")
    for w in workers:
        worker_block += w + " "
    worker_block += ")\n"
    worker_block += "j=0\n"
    worker_cmd = cmd_(workerdir, configdir, worker_logdir, "worker", "$j")
    cmds = [ssh_("node$i", docker_(container_name, worker_cmd)), "((j=j+1))", "sleep 0.5"]
    worker_block += loop_block(cmds, ("i", "\"${arr[@]}\""))
    return worker_block
    
def monitor_block(monitordir, configdir, monitor_logdir, names, container_name, allnodes):
    block = ""
    names = names.split(",")
    allnodes = allnodes.split(",")
    for i, n in enumerate(names):
        args = {"is_head": n == names[0], "name": n}
        monitor_cmd = cmd_(monitordir, configdir, monitor_logdir, "monitor", None, args = args)
        block += ssh_("node" + allnodes[i], docker_(container_name, monitor_cmd)) + "\n"
    return block

def script(config, run_trainer = True, run_worker = True, run_monitor = True): # workers {<start>..<end>}
    content = ""
    if run_trainer:
        tb = trainer_block(config.trainer_dir, config.config_dir, config.trainer_logdir, config.trainers, config.container_name)
        content += tb + "\n"
    if run_worker:
        wb = worker_block(config.worker_dir, config.config_dir, config.worker_logdir, config.workers, config.container_name)
        content += wb + "\n"
    if run_monitor:
        allnodes = config.trainers + "," + config.workers
        mb = monitor_block(config.monitor_dir, config.config_dir, config.monitor_logdir, config.nodes, config.container_name, allnodes)    
        content += mb + "\n"

    # return tb + "\n" + wb + "\n" + mb + "\n"
    return content

def to_bash_file(content, scriptdir = "monitor/generated/launch.sh"):
    os.system("mkdir -p " + os.path.dirname(scriptdir))
    f = open(scriptdir, "w")
    f.write("#!/bin/bash\n")
    f.write(content)
    f.close()
    os.system("chmod 755 " + scriptdir)
 
def to_stdout(content):
    print("#!/bin/bash")
    print(content)

if __name__ == '__main__':
    # to_stdout(content("/workspace/code/main.py", "/workspace/code/configs/starcraft2/config.yaml", "/workspace/ddp.log", "{77..78}", "{75..76}"))
    
    os.system("mkdir -p generated/")
    to_stdout(scrpit(config))
