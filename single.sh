export experiment_name=SingleNode
deepspeed --num_gpus=1 singleShow.py --deepspeed_config single_config.json