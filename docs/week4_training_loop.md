# 第 4 周：训练主循环

> **目标**：理解 JAX 训练脚本的完整流程 —— 从初始化到数据加载、训练步、检查点保存和恢复。

---

## 1. 背景知识

### 1.1 什么是训练循环（Training Loop）？

深度学习的训练本质上是一个重复过程：

```
初始化模型参数（随机或从预训练加载）
  ↓
重复 N 次（每次叫一个"step"）：
  1. 从数据集取一个 batch
  2. 前向传播：模型(输入) → 预测
  3. 计算损失：loss = 预测与真值的差距
  4. 反向传播：计算梯度（每个参数应该怎么调整）
  5. 更新参数：参数 = 参数 - 学习率 × 梯度
  ↓
保存最终模型
```

### 1.2 JAX 的特殊之处

与 PyTorch 不同，JAX 的计算是**纯函数式**的：

```python
# PyTorch 风格（有状态）：
model.train()
output = model(input)      # model 内部有状态
loss.backward()            # 梯度自动累积在参数上
optimizer.step()           # 原地更新参数

# JAX 风格（无状态）：
loss, grads = jax.value_and_grad(loss_fn)(params, input)  # 显式传参数
new_params = optax.apply_updates(params, grads)             # 显式返回新参数
```

### 1.3 什么是 JIT（Just-In-Time 编译）？

`jax.jit` 会把 Python 函数编译成高效的 GPU 代码。第一次调用时编译（慢），后续调用直接执行编译后的代码（快）。

```python
@jax.jit
def train_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    return new_params, loss

# 第一次调用：编译 + 执行（几秒到几十秒）
# 后续调用：直接执行（毫秒级）
```

### 1.4 什么是 FSDP（Fully Sharded Data Parallel）？

当模型太大，一张 GPU 放不下时，FSDP 把模型参数分片存储在多张 GPU 上：

```
GPU 0: 参数片段 A + 计算    GPU 1: 参数片段 B + 计算
      ↕ 通信：交换需要的参数 ↕
```

在 JAX 中通过 `mesh`（设备网格）来管理。

---

## 2. train.py 整体结构

打开 `scripts/train.py`，整个脚本可以分为 5 个部分：

```python
# scripts/train.py 的宏观结构

# ─── Part 1: 辅助函数 ───
def init_logging():         # 自定义日志格式
def init_wandb():           # 初始化 Weights & Biases 日志
def init_train_state():     # 初始化训练状态（参数、优化器）
def train_step():           # 单步训练（前向、反向、更新）
def save_state():           # 保存检查点

# ─── Part 2: 主函数 ───
def main(config):
    # 2.1 初始化
    # 2.2 创建 mesh（多 GPU 分片）
    # 2.3 创建数据加载器
    # 2.4 初始化模型和训练状态
    # 2.5 主训练循环
    # 2.6 保存最终检查点
```

---

## 3. 关键函数详解

### 3.1 init_train_state —— 初始化训练状态

```python
# scripts/train.py 中的 init_train_state（简化注释版）
def init_train_state(config, mesh, data_loader):
    # 1. 创建模型
    model = config.model.create(rng=jax.random.key(config.seed))

    # 2. 分离模型为"图"（结构）和"状态"（参数）
    graphdef, params, other_state = nnx.split(model, nnx.Param, ...)

    # 3. 加载预训练权重（如果有）
    params = config.weight_loader.load(params)

    # 4. 应用冻结过滤器（LoRA 微调时）
    # frozen_params 不会被优化器更新
    trainable_params, frozen_params = split_by_filter(params, config.freeze_filter)

    # 5. 创建优化器
    tx = create_optimizer(config.optimizer, config.lr_schedule)

    # 6. 创建 TrainState（封装参数+优化器状态+EMA）
    train_state = TrainState.create(
        apply_fn=None,
        params=trainable_params,
        tx=tx,
    )

    return train_state, frozen_params, graphdef, other_state
```

**关键概念**：
- `nnx.split(model)` 将 Flax NNX 模型分为结构（graphdef）和参数（params）
- 之后就可以对参数做函数式操作（JIT 友好）

### 3.2 train_step —— 单步训练

```python
# scripts/train.py 中的 train_step（简化注释版）
def train_step(train_state, frozen_params, graphdef, other_state, batch, rng):
    # 1. 合并可训练参数和冻结参数
    all_params = merge(train_state.params, frozen_params)

    # 2. 重建模型
    model = nnx.merge(graphdef, all_params, other_state)

    # 3. 前向传播 + 计算梯度（一步完成）
    def loss_fn(model):
        observation, actions = batch
        losses = model.compute_loss(rng, observation, actions, train=True)
        return losses.mean()  # 对 batch 和 action horizon 取平均

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # 4. 只取可训练参数的梯度
    trainable_grads = extract_trainable(grads)

    # 5. 用优化器更新参数
    train_state = train_state.apply_gradients(grads=trainable_grads)

    # 6. 更新 EMA（指数移动平均）
    if config.ema_decay:
        ema_params = ema_update(train_state.params, ema_params, decay=0.99)

    return train_state, {"loss": loss, "grad_norm": grad_norm, ...}
```

**EMA（指数移动平均）** 是什么？
```
ema_params = 0.99 * ema_params + 0.01 * current_params
```
它平滑了训练过程中参数的震荡，推理时用 EMA 参数通常效果更好。

### 3.3 save_state —— 保存检查点

```python
# 保存检查点包含 3 部分
def save_state(checkpoint_manager, train_state, step):
    checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
            train_state=train_state,    # 完整训练状态（恢复训练用）
            params=params_only,         # 仅参数（推理部署用）
            assets=norm_stats,          # 归一化统计量
        ),
    )
```

检查点目录结构：
```
checkpoints/pi0_aloha_sim/my_exp/
├── 1000/                    ← step 1000 的检查点
│   ├── train_state/         ← 完整训练状态
│   ├── params/              ← 仅参数（用于推理）
│   └── assets/              ← 归一化统计量
├── 2000/
├── 3000/
└── ...
```

---

## 4. 主训练循环

```python
# scripts/train.py main() 函数的核心循环（简化注释版）
def main(config: TrainConfig):
    # ──── 初始化阶段 ────

    init_logging()

    # 创建检查点目录
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 创建设备 mesh（多 GPU 分片）
    mesh = sharding.create_mesh(config.fsdp_devices)
    # fsdp_devices=1: 单 GPU
    # fsdp_devices=4: 4 张 GPU 分片

    # 初始化 wandb（实验追踪工具）
    init_wandb(config, resuming=config.resume)

    # 创建数据加载器
    data_loader = _data_loader.create_data_loader(config)
    data_iter = iter(data_loader)

    # 初始化训练状态
    train_state, frozen_params, ... = init_train_state(config, mesh, data_loader)

    # JIT 编译训练步（第一次调用前编译）
    jit_train_step = jax.jit(train_step, ...)

    # 如果是恢复训练，从检查点恢复
    if config.resume:
        train_state = restore_state(checkpoint_manager, train_state)
        start_step = train_state.step
    else:
        start_step = 0

    # ──── 主循环 ────

    for step in range(start_step, config.num_train_steps):
        # 1. 取一个 batch
        batch = next(data_iter)

        # 2. 执行一步训练
        train_state, info = jit_train_step(train_state, frozen_params, batch, rng)

        # 3. 定期打印日志
        if step % config.log_interval == 0:
            logging.info(f"step={step}, loss={info['loss']:.4f}")
            wandb.log(info, step=step)

        # 4. 定期保存检查点
        if step % config.save_interval == 0:
            save_state(checkpoint_manager, train_state, step)

    # ──── 结束 ────
    save_state(checkpoint_manager, train_state, config.num_train_steps)
```

---

## 5. 训练时序图

```
时间 →

[init]
  │
  ├── create_mesh(fsdp_devices)
  ├── init_wandb()
  ├── create_data_loader()
  ├── init_train_state()
  │     ├── create model
  │     ├── load pretrained weights
  │     ├── split frozen/trainable
  │     └── create optimizer
  ├── jit(train_step)          ← 预编译
  │
  ▼
[main loop]  step=0, 1, 2, ..., num_train_steps
  │
  │  每一步:
  │  ┌─────────────────────────────────────────┐
  │  │  batch = next(data_iter)                │
  │  │  state, info = train_step(state, batch) │
  │  │     ├── forward: loss = model(obs, act) │
  │  │     ├── backward: grads = ∂loss/∂params │
  │  │     └── update: params -= lr * grads    │
  │  └─────────────────────────────────────────┘
  │
  │  每 100 步: wandb.log(loss, grad_norm, lr)
  │  每 1000 步: save_state(train_state, params, assets)
  │
  ▼
[finish]
  └── save final checkpoint
```

---

## 6. 三个关键超参对训练的影响

### 6.1 batch_size（批量大小）

```python
batch_size: int = 32    # 默认值
```

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 小（8-16） | 训练慢但省显存，梯度噪声大 | 显存不足时 |
| 中（32-64） | 平衡 | 大多数场景 |
| 大（128+） | 训练快但费显存，可能需要调大学习率 | 多 GPU |

**注意**：多 GPU 时，每张 GPU 处理 `batch_size / num_gpus` 个样本。

### 6.2 lr_schedule（学习率调度）

```python
lr_schedule: LRScheduleConfig
# 通常包含：
#   init_value: 初始学习率（很小，如 1e-7）
#   peak_value: 峰值学习率（如 2.5e-5）
#   warmup_steps: 预热步数（如 1000）
#   end_value: 最终学习率（如 0）
```

学习率的典型变化曲线：
```
学习率
  ▲
  │     peak_value
  │    ╱────────╲
  │   ╱          ╲
  │  ╱            ╲
  │ ╱              ╲
  │╱    warmup      ╲  cosine decay
  ├───────────────────╲──────────── → 步数
  init_value           end_value
```

- **Warmup 阶段**：学习率从很小线性增大到峰值（防止训练初期不稳定）
- **Cosine Decay 阶段**：学习率按余弦曲线逐渐下降到接近 0

### 6.3 ema_decay（EMA 衰减率）

```python
ema_decay: float | None = 0.99
```

| 值 | 效果 |
|----|------|
| 0.99 | EMA 参数更新较快，更跟踪最近的训练 |
| 0.999 | EMA 参数更新更慢，更平滑 |
| None | 不使用 EMA |

**实际影响**：推理时使用 EMA 参数通常比最后一步的参数效果更好，因为它平滑了训练后期的震荡。

---

## 7. 检查点恢复逻辑

```python
# 恢复训练的关键代码
if config.resume:
    # 1. 检查是否有可恢复的检查点
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError("No checkpoint found to resume from")

    # 2. 恢复训练状态（参数 + 优化器状态 + EMA）
    train_state = restore_state(checkpoint_manager, train_state, latest_step)

    # 3. 恢复数据迭代器状态
    # 快进到上次中断的位置，避免重复训练相同数据
    data_iter = fast_forward(data_iter, latest_step * batch_size)

    logging.info(f"Resumed from step {latest_step}")
```

**为什么恢复数据迭代器很重要？**
- 如果不恢复，恢复训练后会重新从第一个 batch 开始
- 这意味着前面的数据被看了两遍，后面的数据可能没看
- 对于训练效果会有轻微影响

---

## 8. Wandb 日志结构

```python
# init_wandb 设置
wandb.init(
    name=config.exp_name,        # 实验名
    config=dataclasses.asdict(config),  # 把配置记录为 wandb config
    project=config.project_name, # 项目名（默认 "openpi"）
)

# 训练中记录
wandb.log({
    "loss": 0.0234,              # 训练损失
    "grad_norm": 1.23,           # 梯度范数（监控训练稳定性）
    "learning_rate": 2.5e-5,     # 当前学习率
    "throughput": 45.6,          # 每秒处理样本数
}, step=step)
```

**小白提示**：Wandb 是一个在线实验追踪工具，可以在网页上看到训练曲线。如果不想用，设置 `wandb_enabled=False`。

---

## 9. 运行命令指南

### 9.1 JAX 训练（单 GPU）

```bash
# 基本训练命令
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_aloha_sim --exp-name=my_first_run

# 参数解释：
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  ← JAX 使用 90% 的 GPU 显存
# pi0_aloha_sim                        ← 配置名（第2周学的）
# --exp-name=my_first_run              ← 实验名（必须指定）
```

### 9.2 恢复中断的训练

```bash
# 添加 --resume 标志
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_aloha_sim --exp-name=my_first_run --resume
```

### 9.3 调整超参

```bash
# 通过命令行覆盖配置中的默认值
uv run scripts/train.py pi0_aloha_sim \
    --exp-name=my_exp \
    --batch-size=16 \
    --num-train-steps=10000 \
    --wandb-enabled=False
```

### 9.4 多 GPU 训练（FSDP）

```bash
# 使用 4 张 GPU
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_aloha_sim \
    --exp-name=my_fsdp_run \
    --fsdp-devices=4
```

### 9.5 运行训练测试（不需要 GPU）

```bash
# train_test.py 会在 CPU 上用 FakeDataset 跑几步
uv run pytest scripts/train_test.py -v
```

---

## 10. 本周产出

### 产出 1：训练时序图

请自己画出本文第 5 节的训练时序图，确保包含：
- 初始化的各个步骤
- 主循环中每一步的操作
- 日志和检查点保存的时机
- 恢复训练的流程

### 产出 2：关键超参影响说明

| 超参 | 调大 | 调小 | 典型值 |
|------|------|------|-------|
| `batch_size` | 训练更快、更稳定，但需要更多显存 | 省显存，但梯度噪声更大 | 32 |
| `lr_schedule.peak_value` | 收敛更快，但可能不稳定 | 更稳定，但收敛慢 | 2.5e-5 |
| `ema_decay` | EMA 更平滑，但对最近变化反应慢 | 跟踪更灵敏，但可能有噪声 | 0.99 |
| `num_train_steps` | 训练更充分，但可能过拟合 | 训练不充分 | 20k-30k |
| `save_interval` | 节省磁盘，但断点恢复间隔大 | 更频繁保存，恢复更灵活 | 1000 |

---

## 11. 本周自检清单

- [ ] 理解 JAX 函数式训练与 PyTorch 的区别
- [ ] 能描述 `init_train_state` 的 6 个步骤
- [ ] 能解释 `train_step` 中前向、反向、更新的流程
- [ ] 理解检查点保存的 3 部分内容
- [ ] 能画出训练时序图
- [ ] 能解释 warmup + cosine decay 学习率曲线
- [ ] 理解 EMA 的含义和作用
- [ ] 知道如何恢复中断的训练
- [ ] 能用命令行启动训练并调整超参
