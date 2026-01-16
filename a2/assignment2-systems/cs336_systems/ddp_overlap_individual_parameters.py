import torch
import torch.nn as nn
import torch.distributed as dist

class DdpOverlapIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []
        
        # 广播初始参数
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 注册钩子
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._get_hook_fn(param))

    def _get_hook_fn(self, param: nn.Parameter):
        def hook_fn(p: nn.Parameter):
            if p.grad is not None:
                handle = dist.all_reduce(p.grad, dist.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)
        
        return hook_fn
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        
        self.handles.clear()

        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= self.world_size