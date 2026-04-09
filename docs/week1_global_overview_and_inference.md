# 第 1 周：全局认知 + 最小推理闭环

> **目标**：理解 OpenPI 是什么、能做什么，跑通一条从"输入观测"到"输出动作"的完整推理链路，建立全局心智模型。

---

## 1. 背景知识：什么是 VLA 模型？

### 1.1 机器人控制的基本问题

机器人控制的核心问题可以用一句话概括：**看到什么（观测）→ 做什么（动作）**。

```
摄像头图像 + 关节状态 + 语言指令  →  [模型]  →  关节动作序列
```

传统方法需要人工编写控制规则，而 VLA（Vision-Language-Action）模型则用深度学习直接从数据中学习这个映射关系。

### 1.2 OpenPI 的三个模型

OpenPI 提供了三种预训练的 VLA 模型：

| 模型 | 全称 | 生成方式 | 一句话描述 |
|------|------|---------|-----------|
| **π₀** | Pi-Zero | Flow Matching | 像"去噪"一样，从随机噪声逐步生成动作序列 |
| **π₀-FAST** | Pi-Zero FAST | 自回归 | 像 ChatGPT 生成文字一样，逐个 token 生成动作 |
| **π₀.5** | Pi-Zero-Point-Five | Flow Matching（增强版） | π₀ 的升级版，泛化能力更强 |

**你现在不需要理解这些模型的内部原理**，只需要知道：它们都接收"观测"，输出"动作"。

### 1.3 什么是 Policy（策略）？

在机器人学中，**策略（Policy）** 就是"观测 → 动作"的映射函数。在 OpenPI 中，`Policy` 是一个 Python 类，它把模型和数据预处理/后处理包装在一起，对外提供一个简单的 `infer(obs) → action` 接口。

---

## 2. 项目结构速览

先打开终端，看看项目的目录结构：

```bash
# 列出顶层目录
ls /home/qiuziyu/code/openpi/
```

你会看到这些关键目录：

```
openpi/
├── src/openpi/          ← 核心源代码（你学习的主要对象）
│   ├── models/          ← 模型定义（π₀, π₀-FAST 等）
│   ├── models_pytorch/  ← PyTorch 版模型
│   ├── policies/        ← 策略封装（连接模型和数据变换）
│   ├── training/        ← 训练管线（配置、数据加载、检查点）
│   ├── serving/         ← 推理服务器（WebSocket）
│   ├── shared/          ← 公共工具（下载、归一化、类型）
│   └── transforms.py    ← 数据变换管线
├── scripts/             ← 可执行脚本（训练、部署、统计）
├── packages/            ← 客户端库（openpi-client）
├── examples/            ← 各机器人平台的使用示例
└── docs/                ← 文档
```

**小白提示**：不要试图一次看完所有代码。本周我们只关注 3 个文件。

---

## 3. 核心阅读 1：README.md

```bash
# 用你喜欢的编辑器打开 README.md
# 重点关注以下内容：
# - "What is OpenPI?" 部分
# - "Getting Started" 部分
# - 硬件要求（GPU 内存）
```

**阅读要点**：
- OpenPI 的模型是在 10,000+ 小时的机器人数据上预训练的
- 推理至少需要 8GB 显存的 GPU
- 支持的机器人平台：ALOHA、DROID、LIBERO

---

## 4. 核心阅读 2：理解配置获取

### 4.1 什么是"配置"？

在 OpenPI 中，一个"配置"（config）定义了模型的所有参数：用哪个模型架构、用什么数据、怎么预处理、从哪里加载权重等等。

打开文件 `src/openpi/training/config.py`，找到 `get_config` 函数（约第 982 行）：

```python
# src/openpi/training/config.py 第 982-989 行
def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
```

**逐行解读**：
1. 函数接收一个字符串参数 `config_name`，比如 `"pi05_droid"`
2. 从一个全局字典 `_CONFIGS_DICT` 中查找对应的配置
3. 如果找不到，会智能提示最接近的配置名（类似拼写纠错）
4. 返回一个 `TrainConfig` 对象

### 4.2 动手试试

```bash
# 列出所有可用的配置名称
uv run python -c "
from openpi.training.config import _CONFIGS
for c in _CONFIGS:
    print(f'  {c.name}')
"
```

你会看到类似这样的输出：

```
  pi0_aloha
  pi05_aloha
  pi0_aloha_sim
  pi0_droid
  pi05_droid
  pi0_libero
  pi05_libero
  ...
```

每个名字对应一种"模型 + 机器人平台"的组合。

### 4.3 查看一个具体配置

```bash
# 查看 pi05_droid 配置的基本信息
uv run python -c "
from openpi.training.config import get_config
config = get_config('pi05_droid')
print(f'配置名称: {config.name}')
print(f'模型类型: {type(config.model).__name__}')
print(f'Batch size: {config.batch_size}')
print(f'训练步数: {config.num_train_steps}')
print(f'动作维度: {config.model.action_dim}')
print(f'动作时域: {config.model.action_horizon}')
"
```

---

## 5. 核心阅读 3：推理闭环 —— create_trained_policy

这是本周最重要的函数。打开 `src/openpi/policies/policy_config.py`：

```python
# src/openpi/policies/policy_config.py 第 16-25 行
def create_trained_policy(
    train_config: _config.TrainConfig,       # 训练配置（告诉我用什么模型）
    checkpoint_dir: pathlib.Path | str,      # 检查点目录（模型权重在哪）
    *,
    repack_transforms: transforms.Group | None = None,  # 额外的数据重打包变换
    sample_kwargs: dict[str, Any] | None = None,        # 采样参数
    default_prompt: str | None = None,                  # 默认语言指令
    norm_stats: dict[str, transforms.NormStats] | None = None,  # 归一化统计量
    pytorch_device: str | None = None,                  # PyTorch 设备
) -> _policy.Policy:
```

这个函数做了 5 件事情（对应代码约第 45-94 行）：

```
步骤 1: 下载检查点（如果是 GCS 地址，自动下载到本地缓存）
    ↓
步骤 2: 检测模型格式（目录里有 model.safetensors → PyTorch，否则 → JAX）
    ↓
步骤 3: 加载模型权重到内存
    ↓
步骤 4: 加载归一化统计量（norm_stats）
    ↓
步骤 5: 组装 Policy = 模型 + 输入变换链 + 输出变换链
```

### 5.1 步骤 2 详解：自动检测 JAX vs PyTorch

```python
# 第 48-50 行
weight_path = os.path.join(checkpoint_dir, "model.safetensors")
is_pytorch = os.path.exists(weight_path)
```

**小白翻译**：检查检查点目录下是否有 `model.safetensors` 文件。如果有，说明这是 PyTorch 格式的模型。

### 5.2 步骤 5 详解：变换链的组装

```python
# 第 75-94 行（简化版）
return _policy.Policy(
    model,
    transforms=[
        *repack_transforms.inputs,                    # 1. 重打包（可选）
        transforms.InjectDefaultPrompt(default_prompt), # 2. 注入默认指令
        *data_config.data_transforms.inputs,           # 3. 机器人特定变换
        transforms.Normalize(norm_stats, ...),         # 4. 归一化
        *data_config.model_transforms.inputs,          # 5. 模型特定变换
    ],
    output_transforms=[
        *data_config.model_transforms.outputs,         # 5'. 模型输出变换
        transforms.Unnormalize(norm_stats, ...),       # 4'. 反归一化
        *data_config.data_transforms.outputs,          # 3'. 机器人输出变换
        *repack_transforms.outputs,                    # 1'. 重打包输出
    ],
)
```

**注意对称性**：输入变换的顺序是 1→2→3→4→5，输出变换的顺序是 5'→4'→3'→1'（反过来的）。这保证了数据经过模型处理后能正确还原。

---

## 6. 核心阅读 4：Policy.infer() —— 推理的入口

打开 `src/openpi/policies/policy.py`，找到 `infer` 方法（约第 68 行）：

```python
# src/openpi/policies/policy.py 第 68-106 行（简化注释版）
def infer(self, obs: dict, *, noise=None) -> dict:
    # 第一步：复制输入（避免修改原始数据）
    inputs = jax.tree.map(lambda x: x, obs)

    # 第二步：依次执行所有输入变换
    inputs = self._input_transform(inputs)

    # 第三步：添加 batch 维度，转换为张量
    if not self._is_pytorch_model:
        # JAX: numpy → jax.Array, 在前面加一个维度 [1, ...]
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    else:
        # PyTorch: numpy → torch.Tensor, 移到 GPU
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
            inputs
        )

    # 第四步：将字典转换为 Observation 对象
    observation = _model.Observation.from_dict(inputs)

    # 第五步：调用模型生成动作！
    outputs = {
        "state": inputs["state"],
        "actions": self._sample_actions(rng_or_device, observation, **sample_kwargs),
    }

    # 第六步：移除 batch 维度，转回 numpy
    outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

    # 第七步：依次执行所有输出变换
    outputs = self._output_transform(outputs)

    return outputs
```

### 6.1 什么是 `jax.tree.map`？

如果你是 Python 小白，这个函数可能让你困惑。简单解释：

```python
# jax.tree.map 对嵌套字典中的每个"叶子"值执行同一个操作
# 例如：
data = {"image": {"cam1": array1, "cam2": array2}, "state": array3}

# jax.tree.map(lambda x: x + 1, data) 会变成：
# {"image": {"cam1": array1+1, "cam2": array2+1}, "state": array3+1}
```

它就像一个"递归版的 map"，能处理任意嵌套的字典/列表结构。

---

## 7. 核心阅读 5：WebSocket 推理服务器

### 7.1 serve_policy.py —— 启动入口

打开 `scripts/serve_policy.py`，这是启动推理服务器的脚本：

```python
# scripts/serve_policy.py 第 99-117 行
def main(args: Args) -> None:
    policy = create_policy(args)    # 根据参数创建 Policy 对象

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",            # 监听所有网络接口
        port=args.port,             # 默认端口 8000
        metadata=policy_metadata,
    )
    server.serve_forever()          # 开始监听连接
```

### 7.2 WebSocket 协议流程

打开 `src/openpi/serving/websocket_policy_server.py`，核心处理函数（约第 48 行）：

```python
# 第 48-83 行（简化注释版）
async def _handler(self, websocket):
    # 1. 客户端连接后，先发送元数据（模型信息等）
    await websocket.send(packer.pack(self._metadata))

    # 2. 进入主循环
    while True:
        # 3. 接收客户端发来的观测数据（msgpack 格式）
        obs = msgpack_numpy.unpackb(await websocket.recv())

        # 4. 调用 Policy 进行推理
        action = self._policy.infer(obs)

        # 5. 附加计时信息
        action["server_timing"] = {"infer_ms": infer_time * 1000}

        # 6. 将动作发回客户端
        await websocket.send(packer.pack(action))
```

**整体架构图**：

```
┌─────────────┐         WebSocket          ┌─────────────────┐
│   机器人     │ ◄─── action (msgpack) ───── │                 │
│   或仿真器   │ ──── obs (msgpack) ───────► │  Policy Server  │
│             │                             │  (端口 8000)     │
│  运行在机器人 │                             │  运行在 GPU 机器 │
│  上的客户端   │                             │                 │
└─────────────┘                             └─────────────────┘
```

### 7.3 运行命令

```bash
# 启动推理服务器的完整命令（需要 GPU）
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_aloha_sim \
    --policy.dir=gs://openpi-assets/checkpoints/pi0_aloha_sim \
    --port=8000

# 也可以使用默认配置（更简单）
uv run scripts/serve_policy.py --env aloha_sim
```

**注意**：实际运行需要 GPU 和下载模型权重（约几 GB），如果你没有 GPU，本周先理解代码逻辑即可。

---

## 8. 核心阅读 6：JAX vs PyTorch 权重加载

### 8.1 两种格式的区别

OpenPI 同时支持 JAX 和 PyTorch 两种模型格式，它们的检查点目录结构不同：

```
JAX 检查点目录结构：          PyTorch 检查点目录结构：
checkpoint/                  checkpoint/
├── params/                  ├── model.safetensors     ← 关键区别
│   ├── manifest.ocdbt       ├── assets/
│   └── ...                  │   └── norm_stats.json
└── assets/                  └── metadata.pt
    └── norm_stats.json
```

### 8.2 加载路径对比

回到 `policy_config.py` 第 48-57 行：

```python
# 检测格式
weight_path = os.path.join(checkpoint_dir, "model.safetensors")
is_pytorch = os.path.exists(weight_path)

# 根据格式选择不同的加载方式
if is_pytorch:
    # PyTorch 路径：用 safetensors 库加载
    model = train_config.model.load_pytorch(train_config, weight_path)
else:
    # JAX 路径：用 Orbax 库加载
    model = train_config.model.load(
        _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    )
```

**对比表**：

| 方面 | JAX | PyTorch |
|------|-----|---------|
| 权重文件 | `params/` 目录（Orbax 格式） | `model.safetensors`（单文件） |
| 加载库 | `orbax.checkpoint` | `safetensors.torch` |
| 数据精度 | bfloat16 | bfloat16（GPU）/ float32（CPU） |
| 编译优化 | `jax.jit`（自动） | `torch.compile`（可选） |
| 检测方式 | 默认（无 safetensors 文件） | 存在 `model.safetensors` |

---

## 9. 本周产出

### 产出 1：推理流程图

根据本周学习，画出以下流程图（可以手绘或用工具）：

```
输入: obs dict
│  包含: images (摄像头图像)
│        state (关节角度)
│        prompt (语言指令，如 "拿起杯子")
│
├─── [输入变换链] ─────────────────────────────
│    1. RepackTransform: 重命名字段
│    2. InjectDefaultPrompt: 注入默认指令
│    3. AlohaInputs/DroidInputs: 机器人特定适配
│    4. Normalize: (x - mean) / std
│    5. TokenizePrompt: 文本 → token ID
│    6. ResizeImages: → 224x224
│
├─── [添加 batch 维度 + 转张量] ───────────────
│
├─── [Observation.from_dict()] ─────────────────
│    将字典转为结构化对象
│
├─── [model.sample_actions()] ──────────────────
│    π₀: 迭代去噪（10-20步）
│    π₀-FAST: 逐 token 生成
│
├─── [移除 batch 维度 + 转 numpy] ─────────────
│
├─── [输出变换链] ─────────────────────────────
│    1. Unnormalize: x * std + mean
│    2. AlohaOutputs/DroidOutputs: 截取有效维度
│
└─── 输出: action dict
     包含: actions [action_horizon, action_dim]
           例如 [50, 32] 表示未来50步、每步32维动作
```

### 产出 2：JAX vs PyTorch 权重加载笔记

请自己整理一页笔记，包含以下要点：

1. **如何判断检查点格式**：看目录下是否有 `model.safetensors`
2. **JAX 加载流程**：`restore_params()` → `model.load()` → JIT 编译
3. **PyTorch 加载流程**：`model.load_pytorch()` → `.to(device)` → `.eval()`
4. **Policy 类的统一封装**：上层调用者无需关心底层是 JAX 还是 PyTorch
5. **关键代码位置**：`src/openpi/policies/policy_config.py` 第 45-94 行

---

## 10. 练习命令汇总

```bash
# 1. 查看项目结构
ls src/openpi/

# 2. 列出所有可用配置
uv run python -c "from openpi.training.config import _CONFIGS; [print(c.name) for c in _CONFIGS]"

# 3. 查看某个配置的详情
uv run python -c "
from openpi.training.config import get_config
c = get_config('pi05_droid')
print(f'Model: {type(c.model).__name__}')
print(f'Action dim: {c.model.action_dim}')
print(f'Action horizon: {c.model.action_horizon}')
"

# 4. 运行测试（验证环境正常）
uv run pytest --strict-markers -m "not manual" -x -q 2>&1 | head -20

# 5. 查看 serve_policy.py 的帮助信息（不需要 GPU）
uv run python scripts/serve_policy.py --help
```

---

## 11. 本周自检清单

- [ ] 能说出 OpenPI 的三种模型及其区别
- [ ] 能解释 Policy 是什么
- [ ] 能找到 `get_config()` 函数并理解它的作用
- [ ] 能描述 `create_trained_policy()` 的 5 个步骤
- [ ] 能画出完整的推理流程图
- [ ] 知道 JAX 和 PyTorch 检查点的目录结构差异
- [ ] 理解 WebSocket 服务器的请求-响应循环
