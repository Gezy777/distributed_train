import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50
from torch.utils.data.distributed import DistributedSampler
from deepspeed.pipe import PipelineModule
import deepspeed
import argparse
import os
from deepspeed.runtime.pipe.module import LayerSpec
import torch.distributed as dist
import csv
import time

from pynvml import *
log_dir = os.environ.get("LOG_DIR", "default_log")
log_dir = "./pipeline/" + log_dir

class GlobalGPUMonitor(threading.Thread):
    def __init__(self, gpu_indices=[0, 1], interval=0.1, log_dir=log_dir):
        super().__init__()
        self.gpu_indices = gpu_indices # 需要监控的 GPU 列表，例如 [0, 1]
        self.interval = interval
        self.log_file = os.path.join(log_dir, "gpu_multicore_realtime.csv")
        self.stop_event = threading.Event()
        
        os.makedirs(log_dir, exist_ok=True)
        nvmlInit()
        
        # 为每块显卡获取句柄
        self.handles = [nvmlDeviceGetHandleByIndex(i) for i in gpu_indices]
        
        # 准备表头：Timestamp, GPU0_Util, GPU0_Mem, GPU1_Util, GPU1_Mem...
        headers = ["Timestamp"]
        for i in gpu_indices:
            headers += [f"GPU{i}_Util(%)", f"GPU{i}_Mem(MB)"]
        
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def run(self):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            while not self.stop_event.is_set():
                row = [f"{time.time():.4f}"]
                for handle in self.handles:
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem = nvmlDeviceGetMemoryInfo(handle)
                    row += [util.gpu, f"{mem.used / 1024**2:.2f}"]
                
                writer.writerow(row)
                f.flush()
                time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        nvmlShutdown()

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

    custom_parts = [0, 11, total_layer]
    model = MyCustomPipelineModule(
        layers=layer_list,
        num_stages=dist.get_world_size(),  # 2张卡，2个stage
        partition_method='uniform',
        loss_fn=nn.CrossEntropyLoss()  # 损失函数，在最后一个stage计算
    )
    return model, custom_parts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 初始化分布式环境
    deepspeed.init_distributed()

    # 2. 构建流水线模型
    # num_stages=2 会将层自动切分到两张显卡上
    model, parts = get_pipeline_model()


    # 3. 数据预处理
    data_path = '/home/shaoth/resnet18/data/ILSVRC/Data/CLS-LOC'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform)

    # 4. 初始化 DeepSpeed
    # 注意：Pipeline 模式下，model_parameters 必须传 model.parameters()
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    # 5. 数据加载 (PP 模式下 sampler 的 rank 处理)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=1, rank=0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu(),
        sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        drop_last=True
    )

    monitor = GlobalGPUMonitor(interval=0.1)
    monitor.start()
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    # 6. 流水线训练循环
    model_engine.train()

    for epoch in range(100):
        train_sampler.set_epoch(epoch)
        # 将 dataloader 转换为可迭代对象
        data_iter = iter(train_loader)
        # 计算全局 Step 的数量
        gas = model_engine.gradient_accumulation_steps()
        total_steps = len(train_loader) // gas
        
        for step in range(total_steps):
            # train_batch 内部会自动处理 Forward, Backward, Step
            # 它会自动在 Stage 0 读取数据，在 Stage 1 计算 Loss
            time_per_step_start = time.time()
            # 执行前向、后向和权重更新
            loss = model_engine.train_batch(data_iter)
            time_per_step_end = time.time()    
            sample_s = model_engine.train_batch_size() / (time_per_step_end - time_per_step_start) 
            util = nvmlDeviceGetUtilizationRates(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)              
            gpu_util = util.gpu
            mem_used_mb = mem.used / 1024**2
            mem_util_pct = (mem.used / mem.total) * 100  
            if step % 5 == 0 :
                if model_engine.is_last_stage():
                    print(f"Time: {time.time():.4f} | Epoch: {epoch} | Step: {step} / {total_steps} | Loss: {loss.item():.4f} | Sample/sec: {sample_s:.1f}",
                        flush=True)
                stage = 1 if model_engine.is_last_stage() else 0
                print(f"Stage {stage} | GPU Util: {gpu_util}% | Mem: {mem_used_mb:.2f}MB | Mem Util: {mem_util_pct:.2f}%",
                    flush=True)


    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    monitor.stop()
    monitor.join()


if __name__ == "__main__":
    main()