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

partition = 'uniform'

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

    custom_parts = [0, 14, total_layer]
    model = MyCustomPipelineModule(
        layers=layer_list,
        num_stages=dist.get_world_size(),  # 2张卡，2个stage
        partition_method=partition,
        loss_fn=nn.CrossEntropyLoss()  # 损失函数，在最后一个stage计算
    )
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # 初始化 DeepSpeed 分布式环境
    deepspeed.init_distributed(dist_backend='nccl')
    
    # 构建模型
    model = get_pipeline_model()

    # 初始化 DeepSpeed 引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    epochs = 50

    swanlab.init(
        project="Resnet50ByPipeline",
        experiment_name="Resnet50_SingleGPU",
        description="流水线并行运行Resnet50",
        config={
            "model": "resnet50",
            "optim": "AdamW",
            "train_batch_size": model_engine.train_batch_size(),
            "micro_batch_size": model_engine.train_micro_batch_size_per_gpu(),
            "gradient_accumulation_steps": model_engine.gradient_accumulation_steps(),
            "gpu_nums": dist.get_world_size(),
            "epoch": epochs,
            "partition": partition
        }
    )

    # Rank 0 下载数据集
    if dist.get_rank() == 0:
        datasets.CIFAR10(root='./data', train=True, download=True)
    dist.barrier()

    time_start = time.time()

    # 准备数据
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 测试集/验证集使用标准处理（去掉了 Random 相关的操作）
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=1, rank=0)
    test_sampler = DistributedSampler(test_dataset, shuffle=False, num_replicas=1, rank=0)

    # 优化： DataLoader 的 batch_size 必须等于 micro_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu(), 
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True 
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu(), 
        sampler=test_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    # print("--------------------------------")
    # print(f"\n" + "="*30)
    # print(f"数据读取验证:")
    # print(f"1. 原始训练集总数: {len(train_dataset)}")
    # print(f"2. 当前进程采样器分配到的数量: {len(train_sampler)}")
    # print(f"3. 每个 Epoch 的总 Batch 数: {len(train_loader)}")
    # print(f"4. 预期的 Global Batch Size: {model_engine.train_batch_size()}")
    # print(f"5. 每个 Micro-batch 的大小: {model_engine.train_micro_batch_size_per_gpu()}")
    # print("="*30 + "\n")
    # print("-------------------------------------")

    train_step = 0
    test_step = 0

    time_dataloader = time.time()
    print(f"数据加载的时间为{time_dataloader - time_start}")
    time_last = time.time()
    
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        
        # 因为每次 train_batch 会消耗 gradient_accumulation_steps 个 micro-batch
        # 所以实际的 steps 需要除以梯度累加步数
        train_steps_per_epoch = len(train_loader) // model_engine.gradient_accumulation_steps()

        for step in range(train_steps_per_epoch):
            # 执行前向、后向和权重更新
            loss = model_engine.train_batch(train_iter) 
            if model_engine.is_last_stage():
                if train_step % 10 == 0:
                    print(f"Epoch: {epoch} | Step: {step}/{train_steps_per_epoch} | Loss: {loss:.4f}")
                swanlab.log({"train_loss": loss}, step=train_step)
            train_step += 1

        time_train = time.time()
        print(f"Epoch{epoch}的训练时间为{(time_train - time_last):.4f}")
        swanlab.log({"train_time": time_train - time_last}, step=epoch)


        model_engine.eval()  # 设置为评估模式
        test_steps_per_epoch = len(test_loader) // model_engine.gradient_accumulation_steps()
        
        # 获取迭代器
        test_iter = iter(test_loader)
        
        with torch.no_grad():
            for step in range(test_steps_per_epoch):
                # eval_batch 会自动处理流水线的前向传播过程
                # 返回的是最后一个 stage 计算出的 loss
                loss = model_engine.eval_batch(test_iter)
                
                if model_engine.is_last_stage():
                    if step % 5 == 0: # 每 5 个 batch 打印一次，避免刷屏
                        print(f"[Eval Batch] Epoch: {epoch} | Batch: {step}/{test_steps_per_epoch} | Loss: {loss:.4f}")
                    swanlab.log({"test_loss": loss}, step=test_step)
                test_step += 1

        time_test = time.time()
        print(f"Epoch{epoch}的测试时间为{time_test - time_train}")
        swanlab.log({"test_time": time_test - time_train}, step=epoch)
        time_last = time_test

        model_engine.train()  # 恢复训练模式

    dist.destroy_process_group()
    swanlab.finish()