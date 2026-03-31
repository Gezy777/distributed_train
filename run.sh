# nsys profile \
#   -w true \
#   -t cuda,nvtx,osrt,nccl \
#   -s none \
#   -o resnet50_report_SingleNode \
#   --force-overwrite true \
deepspeed --num_gpus=1 train_no_record.py --deepspeed_config dds_config.json 2>&1 | tee ./logs/singleGPU_BENCH.log