from copy import deepcopy
from torch.optim import SGD, Adam, AdamW, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from utils.logger import print_log


def create_optimizer(cfg, model):
    # 深拷贝优化器配置
    optim_kwargs = deepcopy(cfg["training"]["optimizer"] or {})
    # 弹出优化器名称（如 adam）
    name = optim_kwargs.pop("name")
    # 弹出自定义的 param_groups（如果有），避免传给优化器初始化
    param_groups_cfg = optim_kwargs.pop("param_groups", [])
    
    # 1. 初始化参数分组列表
    param_groups = []
    # 基础 lr（从优化器配置中获取）
    base_lr = optim_kwargs.get("lr", 0.001)
    
    # 2. 遍历模型参数，按名称分组
    # 先收集所有参数：分为基础参数和局部特征参数
    base_params = []
    local_params = []
    # 需要特殊 lr 的参数名关键词
    local_param_keys = []
    local_lr = base_lr  # 默认和基础 lr 一致
    
    if param_groups_cfg:
        # 读取配置中的局部特征参数配置
        for group in param_groups_cfg:
            local_param_keys = group.get("params", [])
            local_lr = group.get("lr", base_lr)
    
    # 遍历模型所有参数，按名称匹配分组
    for param_name, param in model.named_parameters():
        if any(key in param_name for key in local_param_keys):
            local_params.append(param)
        else:
            base_params.append(param)
    
    # 3. 添加基础参数组（所有非局部特征参数）
    param_groups.append({
        "params": base_params,
        **optim_kwargs  # 继承基础优化器配置（lr、betas 等）
    })
    
    # 4. 添加局部特征参数组（如果有）
    if local_params and local_param_keys:
        param_groups.append({
            "params": local_params,
            "lr": local_lr,  # 使用自定义 lr
            # 其他参数（如 betas、weight_decay）继承基础配置
            **{k: v for k, v in optim_kwargs.items() if k != "lr"}
        })
    
    # 5. 创建优化器
    optimizer = get_optimizer(name)(param_groups, **{k: v for k, v in optim_kwargs.items() if k != "lr"})
    
    # 打印日志（显示分组信息）
    log_info = f"Optimizer '{name}' init: base_lr={base_lr}"
    if local_params:
        log_info += f", local_param_keys={local_param_keys}, local_lr={local_lr}"
    print_log(log_info)
    
    return optimizer


def get_optimizer(name):
    if name is None:
        name = 'sgd'
    return {
        "sgd": SGD,
        "adam": Adam,
        "adamw": AdamW,
        "asgd": ASGD,
        "adamax": Adamax,
        "adadelta": Adadelta,
        "adagrad": Adagrad,
        "rmsprop": RMSprop,
    }[name]