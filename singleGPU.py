import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet50
import deepspeed
import argparse
import os
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 1. 模型初始化 (FP32)
    model = resnet50(weights=None)

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

    # 3. 初始化 DeepSpeed
    # 注意：对于 FP32，我们不需要在 initialize 中传特殊的参数
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    train_sampler = DistributedSampler(
        train_dataset, 
        shuffle=True
    )

    # 4. 针对单卡和指定 Batch Size 重新构建 DataLoader
    # 虽然 deepspeed.initialize 可以处理数据，但手动创建更直观
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu(),
        sampler=train_sampler,
        num_workers=8,  # 这里的核心数建议设为 GPU 数 * 4 以上
        pin_memory=True
    )

    # 5. 训练循环
    model_engine.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        train_sampler.set_epoch(epoch)
        total_steps = len(train_loader)
        for step, (inputs, labels) in enumerate(train_loader):
            start_time = time.time()
            # 数据搬运
            inputs = inputs.to(model_engine.local_rank)
            labels = labels.to(model_engine.local_rank)

            # 前向
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            # 后向 + 更新
            model_engine.backward(loss)
            model_engine.step()

            end_time = time.time()
        
            if step % 10 == 0:
                # 计算速率
                duration = end_time - start_time
                # 全局 Batch Size = train_batch_size
                samples_per_sec = model_engine.train_batch_size() / duration
                
                print(f"Epoch: {epoch} | Step: {step} / {total_steps} | Loss: {loss.item():.4f} | "
                    f"Time: {duration:.3f}s | Samples/sec: {samples_per_sec:.2f}")

    # 在 main 函数的最末尾
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
if __name__ == "__main__":
    main()