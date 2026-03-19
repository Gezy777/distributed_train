export NCCL_SOCKET_IFNAME=ens5f0np0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

deepspeed --hostfile=hostfile \
          --master_addr=192.168.186.31 \
          --master_port=29500 \
          --num_gpus=2 \
          train.py \
          --deepspeed_config ds_config.json