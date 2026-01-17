import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Any, Iterable, Type, Dict

class ShardedOptimizer(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]], optimizer_cls: Type[Optimizer], **kwargs: Any):
        """
        初始化分片状态优化器
        """

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        self.param_to_rank = {}
        self.rank_to_params = [[] for _ in range(self.world_size)]

        super().__init__(params, kwargs)

        # 为当前rank创建优化器实例，仅将当前rank的参数传给内部优化器
        owned_params = self.rank_to_params[self.rank]
        if len(owned_params) > 0:
            self.base_optimizer = self.optimizer_cls(owned_params, **self.optimizer_kwargs)
        else:
            self.base_optimizer = None

    
    def add_param_group(self, param_group: Dict[str, Any]):
        """
        将参数分配给不同的rank
        """
        if not isinstance(param_group['params'], list):
            param_group['params'] = list(param_group['params'])
        
        for i, param in enumerate(param_group['params']):
            if param in self.param_to_rank:
                continue

            owner_rank = i % self.world_size
            self.param_to_rank[param] = owner_rank
            self.rank_to_params[owner_rank].append(param)
        
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """
        执行优化器并同步更新后的参数
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.base_optimizer is not None:
            self.base_optimizer.step(**kwargs)
        
        for param, owner_rank in self.param_to_rank.items():
            dist.broadcast(param, src=owner_rank)
        
        return loss
