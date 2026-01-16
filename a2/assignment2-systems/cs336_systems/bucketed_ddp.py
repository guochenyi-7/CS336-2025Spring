import torch
import torch.nn as nn
import torch.distributed as dist

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class BucketedDDP(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []
        self.bucket_size_b = bucket_size_mb * 1024 * 1024

        # 广播初始权重
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 分桶
        self.buckets = []
        current_bucket = []
        current_size = 0

        params = list(self.module.parameters())
        for param in reversed(params):
            if not param.requires_grad:
                continue
            param_size = param.numel() * param.element_size()
            # 如果加入新参数会溢出，则封存当前桶（如果非空)
            if current_size + param_size > self.bucket_size_b and current_bucket:
                self.buckets.append(self._create_bucket_metadata(current_bucket))
                current_bucket = []
                current_size = 0
            
            current_bucket.append(param)
            current_size += param_size
            
            # 如果单个大参数已经让当前桶达到阈值，直接封桶
            if current_size >= self.bucket_size_b:
                self.buckets.append(self._create_bucket_metadata(current_bucket))
                current_bucket = []
                current_size = 0

        # 最后一个桶
        if current_bucket:
            self.buckets.append(self._create_bucket_metadata(current_bucket))

        # 注册钩子
        self.param_to_bucket_id = {}
        for b_id, bucket in enumerate(self.buckets):
            for param in bucket["params"]:
                self.param_to_bucket_id[param] = b_id
                param.register_post_accumulate_grad_hook(self._make_hook(param))
    
    def _create_bucket_metadata(self, params):
        return {
            "params": params,
            "ready_count": 0,
            "total_params": len(params)
        }

    
    def _make_hook(self, param):
        def hook(*args):
            b_id = self.param_to_bucket_id[param]
            bucket = self.buckets[b_id]
            bucket["ready_count"] += 1

            if bucket["ready_count"] == bucket["total_params"]:
                self.all_reduce_bucket(bucket)
        return hook
    
    def all_reduce_bucket(self, bucket):
        grads = [p.grad for p in bucket["params"]]
        flat_grads = _flatten_dense_tensors(grads)
        handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, flat_grads, bucket["params"]))

        # 清桶
        bucket['ready_count'] = 0

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle, flat_grads, params in self.handles:
            handle.wait()
            flat_grads /= self.world_size
            synced_grads = _unflatten_dense_tensors(flat_grads, params)
            for p, g in zip(params, synced_grads):
                p.grad.copy_(g)
        
        self.handles.clear()
