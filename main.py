from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from deepspeed.pipe import PipelineModule
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torch import nn

import torch.distributed as dist
import torch.nn as nn
import deepspeed
import argparse
import swanlab
import torch

def build_resnet50_layers():
    model = resnet50(pretrained=False)  # 微调时可加载预训练权重
    layers = []

    # Stem
    layers.append(model.conv1)
    layers.append(model.bn1)
    layers.append(model.relu)
    layers.append(model.maxpool)

    # Layer1 (3个Bottleneck)
    for bottleneck in model.layer1:
        layers.append(bottleneck)

    # Layer2 (4个Bottleneck)
    for bottleneck in model.layer2:
        layers.append(bottleneck)

    # Layer3 (6个Bottleneck)
    for bottleneck in model.layer3:
        layers.append(bottleneck)

    # Layer4 (3个Bottleneck)
    for bottleneck in model.layer4:
        layers.append(bottleneck)

    # Global average pooling 和 FC
    layers.append(model.avgpool)
    layers.append(nn.Flatten())
    layers.append(model.fc)

    return layers

def partition_resnet50_two_stages(layers, num_stages):
    # 统计各模块数量（便于理解，实际可写死）
    # 假设 layers 顺序如构建时：
    # 0:conv1, 1:bn1, 2:relu, 3:maxpool, 
    # 4-6: layer1(3), 7-10: layer2(4), 11-16: layer3(6), 17-19: layer4(3),
    # 20:avgpool, 21:flatten, 22:fc
    # 总计 23 个模块
    # 划分：stage0 取索引 0~?，stage1 取剩余
    # 我们想把 layer3 的前3个放 stage0，后3个放 stage1
    # 计算：layer3 从索引 11 开始，共6个，所以前3个是 11,12,13，后3个是 14,15,16
    # stage0 结束于索引 13 (即包含 layer3 的第3个)
    # stage1 从索引 14 开始到结尾
    # 验证：stage0 包含 0-13 共14个模块；stage1 包含 14-22 共9个模块。模块数差异大，但计算量上 stage1 深层通道多，可能平衡。
    # 也可以调整让 stage0 包含到 layer3 的第4个，但为了示例，就用这个。
    split_idx = 14  # 第一个 stage 的最后一个索引+1
    return [0, split_idx, len(layers)]

def get_pipeline_model():
    layer_list = build_resnet50_layers()
    total_layer = len(layer_list)

    custom_parts = [0, 11, total_layer]
    model = PipelineModule(
        layers=layer_list,
        num_stages=2,  # 2张卡，2个stage
        partition_method=custom_parts,
        loss_fn=torch.nn.CrossEntropyLoss()  # 损失函数，在最后一个stage计算
    )
    return model, custom_parts

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    epochs = 10

    deepspeed.init_distributed(dist_backend='nccl')
    torch.cuda.set_device(args.local_rank)

    if dist.get_rank() == 0:
        swanlab.init(
            project="Resnet50ByPipeline",
            experiment_name="Resnet50",
            description="流水线并行运行Resnet50",
            config={
                "model": "resnet50",
                "optim": "AdamW",
                "lr": 0.001,
                "batch_size": 1024,
                "gpu_nums": 2,
                "train_micro_batch_size_per_gpu": 128,
                "gradient_accumulation_steps": 8,
                "epoch": epochs
            }
        )

    # 构建模型
    model, parts = get_pipeline_model()

    # 准备数据
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. 只有 Rank 0 下载
    if dist.get_rank() == 0:
        datasets.CIFAR10(root='./data', train=True, download=True)
    dist.barrier()

    # 3. 所有人加载（此时 download 设为 False）
    dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=False, 
        transform=transform
    )

    sampler = DistributedSampler(dataset, shuffle=True)

    train_loader = DataLoader(
        dataset,
        batch_size=256, 
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True 
    )
    
    # 初始化 DeepSpeed 引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    # global_step = 0

    # for epoch in range(epochs):
    #     sampler.set_epoch(epoch)
    #     data_iter = iter(train_loader)
    #     for step, _ in enumerate(train_loader):
    #         # 执行一个步长的 Pipeline 训练 (包含前向、后向和权重更新)
    #         # 传入迭代器，DeepSpeed 会自动从中抽取数据
    #         loss = model_engine.train_batch(data_iter) 

    #         # 只有最后一个 Stage 负责日志记录
    #         if model_engine.is_last_stage():
    #             # 打印到控制台
    #             if global_step % 10 == 0:
    #                 print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f}")
                
    #         # 记录到 SwanLab
    #         if dist.get_rank() == 0:
    #             swanlab.log({"train_loss": loss.item()}, step=global_step)

    #         global_step += 1

    global_step = 0

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        # 必须把 loader 转为迭代器
        data_iter = iter(train_loader)
        
        # 算出这一轮一共有多少步
        steps_per_epoch = len(train_loader)

        for step in range(steps_per_epoch):
            # 这一步会执行 1 个 Step (包含所有 Micro-batches)
            loss = model_engine.train_batch(data_iter) 

            # --- 修正日志逻辑 ---
            # 重点：DeepSpeed 会自动把最后阶段计算的 Loss 广播给所有 Rank
            # 所以我们直接在 Rank 0 记录即可，不管它是哪个 Stage
            if dist.get_rank() == 0:
                if global_step % 10 == 0:
                    # 此时 loss 是所有卡都有的
                    print(f"Epoch: {epoch} | Step: {step}/{steps_per_epoch} | Loss: {loss.item():.4f}")
                
                # 记录到 SwanLab
                swanlab.log({"train_loss": loss.item()}, step=global_step)

            global_step += 1

    if dist.get_rank() == 0:
        swanlab.finish()
