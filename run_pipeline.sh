# nsys profile \
#   -w true \
#   -t cuda,nvtx,osrt,nccl \
#   -s none \
#   -o resnet50_report_SingleNode \
#   --force-overwrite true \
log_dir="uniform_B128S"
export LOG_DIR=$log_dir
deepspeed --num_gpus=2 pipeline_train.py --deepspeed_config ds_config.json 2>&1 | tee ./pipeline/${log_dir}.log