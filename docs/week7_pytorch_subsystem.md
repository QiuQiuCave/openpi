# 第 7 周：PyTorch 子系统

> **目标**：理解 PyTorch 实现如何镜像 JAX 版本，掌握 DDP 多 GPU 训练、torch.compile 优化，以及 JAX ↔ PyTorch 迁移注意事项。

---

## 1. 背景知识

### 1.1 为什么需要两套实现？

OpenPI 的主实现是 JAX（Google 的框架），但 PyTorch（Meta 的框架）在工业界更普及。双框架支持让用户可以根据自己熟悉的工具链选择。

| 方面 | JAX | PyTorch |
|------|-----|---------|
| 编程范式 | 函数式（无状态） | 命令式（有状态） |
| 编译 | 默认 JIT | 可选 `torch.compile` |
| 多 GPU | FSDP via mesh | DDP / FSDP via torchrun |
| 社区 | 学术界（Google） | 工业界（Meta） |
| 调试 | 较难（需要理解 trace） | 较易（标准 Python 调试） |

### 1.2 什么是 DDP（Distributed Data Parallel）？

DDP 是 PyTorch 的多 GPU 训练方式：每张 GPU 各有一份完整的模型副本，各自处理不同的数据，然后同步梯度。

```
GPU 0: model 副本 + batch_0 → grad_0 ─┐
GPU 1: model 副本 + batch_1 → grad_1 ──┤── AllReduce（平均梯度）
GPU 2: model 副本 + batch_2 → grad_2 ──┤      ↓
GPU 3: model 副本 + batch_3 → grad_3 ─┘  所有 GPU 用相同的平均梯度更新
```

### 1.3 什么是 torch.compile？

PyTorch 2.0 引入的编译优化，将 Python 代码编译为高效的 GPU kernel：

```python
# 未编译：Python 解释执行，每步都有 Python 开销
model = MyModel()
output = model(input)

# 编译后：Python 代码被编译为 GPU 图，直接执行
model = torch.compile(model, mode="reduce-overhead")
output = model(input)  # 第一次慢（编译），后续快
```

### 1.4 什么是 safetensors？

一种安全高效的模型权重文件格式：
- 比 `torch.save()` 更安全（不会执行任意代码）
- 支持内存映射（mmap），加载速度快
- 跨框架兼容（JAX 和 PyTorch 都能用）

---

## 2. PyTorch 模型实现

### 2.1 目录结构

```
src/openpi/models_pytorch/
├── pi0_pytorch.py           ← π₀ PyTorch 实现
├── gemma_pytorch.py         ← Gemma PyTorch 实现
├── preprocessing_pytorch.py ← 数据预处理
└── transformers_replace/    ← 修改过的 HuggingFace 代码
```

### 2.2 PI0Pytorch 类

打开 `src/openpi/models_pytorch/pi0_pytorch.py`：

```python
# 约第 84-130 行
class PI0Pytorch(nn.Module):
    def __init__(self, config: Pi0Config):
        super().__init__()

        # 组件 1: PaliGemma + Action Expert（HuggingFace 模型）
        self.paligemma_with_expert = PaliGemmaWithExpertModel(config)

        # 组件 2: 投影层（与 JAX 版本一一对应）
        self.action_in_proj = nn.Linear(config.action_dim, width)
        self.action_out_proj = nn.Linear(width, config.action_dim)
        self.state_proj = nn.Linear(config.action_dim, width)

        # 组件 3: 时间嵌入
        self.action_time_mlp_in = nn.Linear(width, width)
        self.action_time_mlp_out = nn.Linear(width, width)

        # 组件 4: 可选的 torch.compile
        if config.pytorch_compile_mode:
            self._compiled_forward = torch.compile(
                self._forward, mode=config.pytorch_compile_mode
            )
```

### 2.3 与 JAX 版本的对应关系

| JAX (pi0.py) | PyTorch (pi0_pytorch.py) | 说明 |
|---|---|---|
| `self.PaliGemma["llm"]` | `self.paligemma_with_expert.paligemma` | HuggingFace 实现 |
| `self.PaliGemma["img"]` | `self.paligemma_with_expert.paligemma.vision_tower` | SigLIP |
| `self.action_expert` | `self.paligemma_with_expert.action_expert` | Gemma-300M |
| `nnx.Linear(...)` | `nn.Linear(...)` | 标准线性层 |
| `jax.random.normal(rng, ...)` | `torch.randn(...)` | 随机噪声 |
| `@jax.jit` | `torch.compile(mode=...)` | 编译优化 |
| `nnx.split/merge` | `state_dict()` / `load_state_dict()` | 参数管理 |

### 2.4 正弦位置编码（跨框架一致性）

```python
# src/openpi/models_pytorch/pi0_pytorch.py 约第 25-42 行
def create_sinusoidal_pos_embedding(timestep, dim, device):
    """创建正弦位置编码 —— 与 JAX 版本产生相同结果"""
    half_dim = dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timestep[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
```

### 2.5 compute_loss（对比 JAX 版本）

```python
# PyTorch 版本（简化）
def forward(self, observation, actions):
    """训练时的前向传播"""
    # 1. 采样噪声和时间步 —— 与 JAX 相同的分布
    noise = torch.randn_like(actions)
    timestep = torch.distributions.Beta(1.5, 1.0).sample([batch_size])

    # 2. 前向扩散 —— 与 JAX 完全相同
    x_t = timestep * noise + (1 - timestep) * actions
    u_t = noise - actions

    # 3. 编码 prefix 和 suffix —— 逻辑相同，API 不同
    prefix = self.embed_prefix(observation)
    suffix = self.embed_suffix(observation, x_t, timestep)

    # 4. Transformer 前向 —— HuggingFace API
    v_t = self.paligemma_with_expert(prefix, suffix)
    v_t = self.action_out_proj(v_t)

    # 5. MSE 损失
    loss = F.mse_loss(v_t, u_t, reduction='none')
    return loss
```

### 2.6 sample_actions（对比 JAX 版本）

```python
def sample_actions(self, device_or_rng, observation, *, num_steps=10, **kwargs):
    """推理时的去噪采样"""
    with torch.no_grad():
        # 1. 编码 prefix + KV cache
        prefix = self.embed_prefix(observation)
        kv_cache = self.fill_kv_cache(prefix)

        # 2. 从噪声开始
        x_t = torch.randn([batch, horizon, dim], device=device)

        # 3. 迭代去噪
        dt = -1.0 / num_steps
        t = 1.0
        for _ in range(num_steps):
            suffix = self.embed_suffix(observation, x_t, t)
            v_t = self.forward_with_cache(kv_cache, suffix)
            v_t = self.action_out_proj(v_t)
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t
```

**关键差异**：PyTorch 版本需要 `torch.no_grad()` 来关闭梯度计算（JAX 中不需要，因为函数式设计天然不追踪梯度）。

---

## 3. PaliGemmaWithExpertModel

打开 `src/openpi/models_pytorch/gemma_pytorch.py`：

```python
# 约第 12-100 行
class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 使用 HuggingFace Transformers 加载 PaliGemma
        self.paligemma = PaliGemmaForConditionalGeneration(hf_config)

        # Action Expert 用 HuggingFace 的 GemmaForCausalLM
        self.action_expert = GemmaForCausalLM(expert_config)

        # 精度管理
        self.to_bfloat16_for_selected_params("bfloat16")

    def to_bfloat16_for_selected_params(self, precision):
        """选择性地将参数转为 bfloat16"""
        if precision == "bfloat16":
            self.paligemma = self.paligemma.to(torch.bfloat16)
            self.action_expert = self.action_expert.to(torch.bfloat16)
            # 但 embedding 层保持 float32（提高数值稳定性）
```

---

## 4. PyTorch 训练脚本

打开 `scripts/train_pytorch.py`：

### 4.1 DDP 初始化

```python
# 约第 30-50 行
def setup_ddp():
    """初始化分布式训练"""
    # torchrun 自动设置这些环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return local_rank, world_size
```

### 4.2 训练主循环

```python
# 简化的 PyTorch 训练循环
def main(config: TrainConfig):
    local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # 1. 创建模型
    model = PI0Pytorch(config.model).to(device)

    # 2. 可选：加载预训练权重
    if config.pytorch_weight_path:
        safetensors.torch.load_model(model, config.pytorch_weight_path)

    # 3. 可选：梯度检查点（节省显存）
    model.enable_gradient_checkpointing()

    # 4. DDP 包装
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    # 5. 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_value,
        weight_decay=config.optimizer.weight_decay,
    )

    # 6. 创建数据加载器
    dataset = create_dataset(config)
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

    # 7. 训练循环
    global_step = 0
    while global_step < config.num_train_steps:
        for batch in loader:
            # 移到 GPU
            observation, actions = move_to_device(batch, device)

            # 前向 + 损失
            losses = model(observation, actions)
            loss = losses.mean()

            # 反向传播
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            # 学习率调度
            update_learning_rate(optimizer, global_step, config.lr_schedule)

            global_step += 1

            # 定期保存
            if global_step % config.save_interval == 0:
                save_checkpoint(model, optimizer, global_step)

            if global_step >= config.num_train_steps:
                break
```

### 4.3 检查点保存

```python
def save_checkpoint(model, optimizer, step, checkpoint_dir):
    """保存 PyTorch 检查点"""
    step_dir = checkpoint_dir / str(step)
    step_dir.mkdir(parents=True, exist_ok=True)

    # 模型权重（safetensors 格式）
    safetensors.torch.save_model(model, step_dir / "model.safetensors")

    # 优化器状态（torch 格式）
    torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")

    # 训练元数据
    torch.save({"step": step, "config": config}, step_dir / "metadata.pt")

    # 归一化统计量
    save_norm_stats(step_dir / "assets")
```

---

## 5. 运行命令指南

### 5.1 单 GPU 训练

```bash
uv run scripts/train_pytorch.py pi0_aloha_sim --exp_name my_pt_run
```

### 5.2 多 GPU 训练（DDP）

```bash
# 2 张 GPU
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi0_aloha_sim --exp_name my_ddp_run

# 4 张 GPU
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi0_aloha_sim --exp_name my_ddp_run

# 多节点（2 节点 × 4 GPU = 8 GPU）
# 在节点 1 运行：
uv run torchrun --nnodes=2 --nproc_per_node=4 \
    --master_addr=NODE1_IP --master_port=29500 --node_rank=0 \
    scripts/train_pytorch.py pi0_aloha_sim --exp_name my_multi_node
# 在节点 2 运行：
uv run torchrun --nnodes=2 --nproc_per_node=4 \
    --master_addr=NODE1_IP --master_port=29500 --node_rank=1 \
    scripts/train_pytorch.py pi0_aloha_sim --exp_name my_multi_node
```

### 5.3 使用 torch.compile

```bash
# 在配置中启用（修改配置或命令行覆盖）
uv run scripts/train_pytorch.py pi0_aloha_sim \
    --exp_name compiled_run \
    --model.pytorch_compile_mode=reduce-overhead
```

compile 模式选项：
| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `default` | 平衡编译时间和运行速度 | 通用 |
| `reduce-overhead` | 减少 Python 开销 | 推理或短计算 |
| `max-autotune` | 最大化性能（编译慢） | 大规模训练 |

### 5.4 运行 PyTorch 测试

```bash
uv run pytest scripts/train_test.py -v -k "pytorch"
```

---

## 6. JAX ↔ PyTorch 迁移注意事项

### 6.1 参数命名差异

```
JAX 参数路径:                         PyTorch 参数路径:
PaliGemma/llm/layer_0/attn/q_proj    paligemma_with_expert.paligemma.
                                        language_model.model.layers.0.
                                        self_attn.q_proj.weight
```

权重加载代码（`weight_loaders.py`）中有完整的名称映射表。

### 6.2 数值精度差异

```python
# JAX: 默认使用 bfloat16
params = jnp.bfloat16(params)

# PyTorch: 需要显式管理
model = model.to(torch.bfloat16)
# 但 embedding 层保持 float32
model.embed_tokens = model.embed_tokens.to(torch.float32)
```

### 6.3 随机数差异

```python
# JAX: 显式传递 RNG key（确定性）
rng = jax.random.key(42)
noise = jax.random.normal(rng, shape)

# PyTorch: 全局随机状态
torch.manual_seed(42)
noise = torch.randn(shape)
```

### 6.4 完整迁移 Checklist

| 检查项 | JAX → PyTorch | PyTorch → JAX |
|--------|--------------|--------------|
| **模型权重** | Orbax → safetensors | safetensors → Orbax |
| **参数名映射** | 需要转换表 | 需要转换表 |
| **精度** | 自动 bfloat16 | 需要显式指定 |
| **编译** | `jax.jit`（总是开） | `torch.compile`（可选） |
| **多 GPU** | mesh + FSDP | torchrun + DDP |
| **检查点** | `params/` 目录 | `model.safetensors` 文件 |
| **梯度** | `jax.value_and_grad` | `loss.backward()` |
| **冻结参数** | `nnx.split` + filter | `param.requires_grad = False` |
| **EMA** | 手动更新 | 手动更新 |
| **数据加载** | 共用 `TorchDataLoader` | 共用 `TorchDataLoader` |
| **随机种子** | 显式 RNG key | 全局 `torch.manual_seed` |
| **调试** | `jax.debug.print` | 标准 `print` / `pdb` |

### 6.5 当前功能差异

| 功能 | JAX | PyTorch | 说明 |
|------|:---:|:-------:|------|
| π₀ 训练 | ✅ | ✅ | 完全对等 |
| π₀ 推理 | ✅ | ✅ | 完全对等 |
| π₀-FAST | ✅ | ❌ | PyTorch 暂不支持 FAST |
| π₀.5 | ✅ | ✅ | 完全对等 |
| LoRA 微调 | ✅ | ✅ | 实现方式不同 |
| FSDP | ✅ | ⚠️ | PyTorch 用 DDP |
| KV Cache | ✅ | ✅ | 实现方式不同 |
| 检查点恢复 | ✅ | ✅ | 格式不同 |
| wandb 日志 | ✅ | ✅ | 完全对等 |

---

## 7. 本周产出

### 产出：JAX ↔ PyTorch 迁移注意事项清单

请整理一份清单，包含以下部分：

1. **检查点格式转换**：从 JAX params/ 到 PyTorch safetensors（或反向）
2. **参数名称映射**：两种框架的参数命名规则差异
3. **精度管理**：bfloat16 在两个框架中的处理方式
4. **训练命令差异**：JAX 的 mesh/FSDP vs PyTorch 的 torchrun/DDP
5. **功能差距**：当前 PyTorch 不支持的功能
6. **调试建议**：各框架的调试最佳实践

---

## 8. 本周自检清单

- [ ] 理解 JAX 和 PyTorch 的编程范式差异
- [ ] 能找到 PyTorch 模型中与 JAX 对应的类和方法
- [ ] 理解 DDP 多 GPU 训练的原理
- [ ] 知道 `torch.compile` 的三种模式和适用场景
- [ ] 理解 safetensors 格式的作用
- [ ] 能用 torchrun 命令启动多 GPU 训练
- [ ] 完成 JAX ↔ PyTorch 迁移清单
