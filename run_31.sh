deepspeed --hostfile=hostfile \
          --master_addr=192.168.186.31 \
          --master_port=29500 \
          --num_gpus=1 \
          train.py \
          --deepspeed_config ds_config.json