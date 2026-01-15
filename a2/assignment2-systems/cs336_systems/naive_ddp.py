import torch
import os

import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
    
def run_single_process(input_data, target_data, steps=5):
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ToyModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(steps):
        optimizer.zero_grad()
        out_put = model(input_data)
        loss = loss_fn(out_put, target_data)
        loss.backward()
        optimizer.step()

    return model.state_dict()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def distributed_each(model):
     world_size = dist.get_world_size()
     for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size

def distributed_all(model):
    world_size = dist.get_world_size()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
        
    if len(grads) > 0:
        # 将所有梯度打平成一个连续的大张量
        flat_grads = _flatten_dense_tensors(grads)
        
        # 仅发起一次 all-reduce 调用 [cite: 1295]
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size
        
        # 将同步后的梯度写回原张量
        # _unflatten_dense_tensors 返回的是新张量列表，需要拷贝回 grads
        synced_grads = _unflatten_dense_tensors(flat_grads, grads)
        for old_grad, new_grad in zip(grads, synced_grads):
            old_grad.copy_(new_grad)


def run_ddp_process(rank, world_size, input_data, target_data, steps=5, return_dict=None):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42 + rank)
    model = ToyModel().to(device)
    # 广播权重
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param.data, 0)

    #  切分数据
    local_batch_size = input_data.size(0) // world_size
    start_idx = local_batch_size * rank
    end_idx = start_idx + local_batch_size

    local_input = input_data[start_idx:end_idx].to(device)
    local_target = target_data[start_idx:end_idx].to(device)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(5):
        optimizer.zero_grad()
        output = model(local_input)
        loss = loss_fn(output, local_target)
        loss.backward()
        # 同步梯度
        distributed_each(model)
        optimizer.step()
    
    if rank == 0 and return_dict is not None:
        # 深拷贝状态字典
        return_dict['state_dict'] = {k: v.clone() for k, v in model.state_dict().items()}
        
    cleanup()

    
def main():
    print("Running Correctness Test (Naive DDP vs Single Process)...")
    world_size = 2
    steps = 5
    batch_size = 8
    
    # 生成随机数据
    torch.manual_seed(123)
    input_data = torch.randn(batch_size, 10)
    target_data = torch.randn(batch_size, 1)

    # 获取单进程结果
    gt_state_dict = run_single_process(input_data, target_data,steps)

    # 获取ddp结果
    manager = mp.Manager()
    return_dict = manager.dict()
    
    mp.spawn(
        run_ddp_process,
        args=(world_size, input_data, target_data, steps, return_dict),
        nprocs=world_size,
        join=True
    )
    ddp_state_dict = return_dict['state_dict']

    # 比较结果
    all_match = True
    for k in gt_state_dict:
        # 允许极小的浮点误差
        if not torch.allclose(gt_state_dict[k], ddp_state_dict[k], atol=1e-6):
            print(f"Mismatch found in layer: {k}")
            print(f"GT: {gt_state_dict[k]}")
            print(f"DDP: {ddp_state_dict[k]}")
            all_match = False
            
    if all_match:
        print(">>> SUCCESS: Naive DDP implementation matches Single Process training exactly!")
    else:
        print(">>> FAILURE: Weights do not match.")

if __name__ == "__main__":
    main()