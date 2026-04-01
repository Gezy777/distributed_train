from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from deepspeed.pipe import PipelineModule
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from deepspeed.runtime.pipe.module import LayerSpec

import torch.distributed as dist
import torch.nn as nn
import deepspeed
import argparse
import swanlab
import torch
import time

import threading
import csv
import os
from pynvml import *
partition = 'parameters'
epochs_set = 2

def build_resnet50_layers():
    model = resnet50(pretrained=False)  
    layers = []

    # Stem
    layers.extend([model.conv1, model.bn1, model.relu, model.maxpool])

    # Layers 1-4
    for bottleneck in model.layer1: layers.append(bottleneck)
    for bottleneck in model.layer2: layers.append(bottleneck)
    for bottleneck in model.layer3: layers.append(bottleneck)
    for bottleneck in model.layer4: layers.append(bottleneck)

    # Global average pooling 和 FC
    layers.extend([model.avgpool, nn.Flatten(), model.fc])

    return layers

class MyCustomPipelineModule(PipelineModule):
    def _partition_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        # 1. 拦截 list 情况，避免执行 .lower()
        if isinstance(method, list):
            if self.global_rank == 0:
                print(f"[Custom] Partitioning pipeline stages with manual list: {method}")
            self.parts = method
        else:
            # 2. 如果是字符串，则继续原有的逻辑
            method_str = method.lower()
            if self.global_rank == 0:
                print(f"Partitioning pipeline stages with method {method_str}")

            if method_str == 'uniform':
                from deepspeed.runtime import utils as ds_utils
                self.parts = ds_utils.partition_uniform(num_items=len(self._layer_specs), num_parts=num_stages)
            elif method_str == 'parameters':
                from deepspeed.runtime import utils as ds_utils
                param_counts = self._count_layer_params()
                self.parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
            else:
                # 兜底原逻辑的其他方法
                return super()._partition_layers(method)

        # 3. --- 下面完全复制你提供的原厂“打印逻辑”，确保你能看到输出 ---
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    elif isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    print(f'    {idx+start:2d}: {name}')

        # 4. 执行原逻辑最后一步：设置边界
        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

def get_pipeline_model():
    layer_list = build_resnet50_layers()
    total_layer = len(layer_list)

    custom_parts = [0, 9, total_layer]
    model = MyCustomPipelineModule(
        layers=layer_list,
        num_stages=dist.get_world_size(),  # 2张卡，2个stage
        partition_method=custom_parts,
        loss_fn=nn.CrossEntropyLoss()  # 损失函数，在最后一个stage计算
    )
    return model, custom_parts

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # 初始化 DeepSpeed 分布式环境
    deepspeed.init_distributed(dist_backend='nccl')
    
    # 构建模型
    model, parts = get_pipeline_model()

    dist.barrier()

    # 初始化 DeepSpeed 引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    epochs = epochs_set

    if model_engine.is_last_stage():
        stage = "1"
    else:
        stage = "0"

    # 2. 数据预处理
    data_path = '/home/shaoth/resnet18/data/ILSVRC/Data/CLS-LOC' # 替换为你本地真实的 ImageNet 路径
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, 'train'), 
        transform=train_transform
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=1, rank=0)

    # 优化： DataLoader 的 batch_size 必须等于 micro_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_engine.train_batch_size(),
        sampler=train_sampler,
        num_workers=16,          # 增加到 16 或更多，取决于你的 CPU 核心数
        pin_memory=True,         # 必须为 True，加速从内存到显存的拷贝
        drop_last=True,
        prefetch_factor=4,       # 关键：强制每个 worker 提前准备 4 个 batch
        persistent_workers=True  # 保持 worker 进程不销毁，减少每个 Epoch 开始时的卡顿
    )

    train_step = 0
    test_step = 0

    for epoch in range(epochs):
        if model_engine.is_last_stage():
            print(f"Epoch: {epoch} starts at {time.time()}", flush=True)
        train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        time_step = []
        
        train_steps_per_epoch = len(train_loader)
        for step in range(train_steps_per_epoch):

            time_per_step_start = time.time()
            # 执行前向、后向和权重更新
            loss = model_engine.train_batch(train_iter)
            time_per_step_end = time.time()
            time_step.append(time_per_step_end - time_per_step_start)

            if train_step % 10 == 0:
                print(
                    f"Time: {time.time():.4f} | "
                    f"Stage: {stage} | Epoch: {epoch} | Step: {step}/{train_steps_per_epoch} | "
                    f"Loss: {loss:.4f} | ",
                    flush=True
                )
            train_step += 1 
        epoch_time = sum(time_step)
        if model_engine.is_last_stage():
            print(f"Epoch: {epoch} ends at {time.time()} | train_time is {epoch_time:.4f}", flush=True)

    dist.destroy_process_group()