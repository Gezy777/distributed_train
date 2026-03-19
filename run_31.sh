# export GLOO_SOCKET_IFNAME=ens6f0
# export TP_SOCKET_IFNAME=ens6f0
# export GLOO_LOG_LEVEL=DEBUG
export NCCL_IB_DISABLE=1          # 禁用 InfiniBand
export NCCL_SOCKET_IFNAME=ens5f0np0    # 替换为实际网卡名（如 eth0, eno1 等）
export NCCL_DEBUG=INFO             # 可选，用于查看通信细节
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

deepspeed --hostfile=hostfile \
          --master_addr=172.0.0.1 \
          --master_port=29500 \
          --num_gpus=1 \
          train.py \
          --deepspeed_config ds_config.json