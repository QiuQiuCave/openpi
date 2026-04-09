# OpenPI 代码库系统学习指南

本指南为希望深入理解 OpenPI 代码库的开发者和研究者提供一条循序渐进的学习路径。建议按阶段顺序推进，每个阶段都建立在前一阶段的基础上。

---

## 目录

- [前置知识](#前置知识)
- [项目全局架构](#项目全局架构)
- [阶段一：项目概览与环境搭建](#阶段一项目概览与环境搭建)
- [阶段二：核心数据结构与类型系统](#阶段二核心数据结构与类型系统)
- [阶段三：模型架构（JAX）](#阶段三模型架构jax)
- [阶段四：数据变换管线](#阶段四数据变换管线)
- [阶段五：训练管线](#阶段五训练管线)
- [阶段六：策略封装与推理服务](#阶段六策略封装与推理服务)
- [阶段七：PyTorch 实现与高级主题](#阶段七pytorch-实现与高级主题)
- [核心数据流详解](#核心数据流详解)
- [关键设计模式](#关键设计模式)
- [常用命令速查](#常用命令速查)

---

## 前置知识

开始学习前，建议对以下领域有基本了解：

| 领域 | 需要掌握的概念 | 推荐资源 |
|------|---------------|---------|
| **Transformer** | Self-attention、KV cache、Prefix-LM attention | Vaswani et al. "Attention Is All You Need" |
| **扩散模型 / Flow Matching** | 前向扩散、去噪过程、速度场预测（velocity prediction） | Lipman et al. "Flow Matching for Generative Modeling" |
| **视觉-语言模型** | PaliGemma、SigLIP 图像编码、多模态 token 拼接 | Google PaliGemma 技术报告 |
| **JAX / Flax** | `jax.jit`、`jax.vmap`、PyTree、Flax NNX 模块系统 | JAX 官方教程、Flax NNX 文档 |
| **PyTorch** | `nn.Module`、DDP、`torch.compile` | PyTorch 官方文档 |
| **机器人控制基础** | 关节空间、动作序列（action chunk）、末端执行器 | 根据目标平台选读 |

---

## 项目全局架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenPI 架构全景                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌────────────┐    ┌──────────────┐            │
│  │ 数据源    │───▶│ 数据变换    │───▶│  训练管线     │            │
│  │ LeRobot  │    │ transforms │    │ train.py     │            │
│  │ RLDS     │    │            │    │ train_pt.py  │            │
│  └──────────┘    └────────────┘    └──────┬───────┘            │
│                                           │                     │
│                                     ┌─────▼─────┐              │
│                                     │  检查点     │              │
│                                     │ checkpoint │              │
│                                     └─────┬─────┘              │
│                                           │                     │
│  ┌──────────┐    ┌────────────┐    ┌──────▼───────┐            │
│  │ 客户端    │◀──▶│ WebSocket  │◀──▶│  策略 Policy  │            │
│  │ Client   │    │  Server    │    │ = Model      │            │
│  │ Library  │    │            │    │ + Transforms │            │
│  └──────────┘    └────────────┘    └──────────────┘            │
│                                                                 │
│  ┌─────────────────────────────────────────────┐               │
│  │              模型层 Models                    │               │
│  │  ┌────────┐  ┌──────────┐  ┌─────────────┐ │               │
│  │  │ SigLIP │  │ PaliGemma│  │Action Expert│ │               │
│  │  │ 视觉   │  │ 语言     │  │ 动作生成    │ │               │
│  │  └────────┘  └──────────┘  └─────────────┘ │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

**核心模块关系：**

- `models/` — 定义模型架构（π₀ 流匹配、π₀-FAST 自回归）
- `transforms.py` — 可组合的数据变换管线，连接原始数据与模型输入
- `training/` — 训练配置、数据加载、检查点管理
- `policies/` — 将模型 + 变换封装为统一的推理接口
- `serving/` — WebSocket 推理服务器
- `packages/openpi-client/` — 独立的客户端库，用于远程推理

---

## 阶段一：项目概览与环境搭建

### 学习目标
- 理解 OpenPI 的定位：为机器人提供预训练的视觉-语言-动作（VLA）模型
- 理解三种模型变体的区别：π₀、π₀-FAST、π₀.5
- 搭建本地开发环境

### 核心阅读
| 文件 | 重点关注 |
|------|---------|
| `README.md` | 项目介绍、模型能力、硬件要求、安装步骤 |
| `pyproject.toml` | 依赖关系、workspace 结构、ruff 配置 |
| `CLAUDE.md` | 命令速查和架构速览 |

### 关键概念

**三种模型变体：**

| 模型 | 动作生成方式 | 特点 |
|------|------------|------|
| **π₀** | Flow Matching（连续扩散） | 迭代去噪生成动作序列，通用性强 |
| **π₀-FAST** | 自回归（离散 token） | 将动作量化为离散 token 后逐个生成，速度更快 |
| **π₀.5** | Flow Matching + 知识隔离 | π₀ 的升级版，状态输入离散化，泛化能力更强 |

### 动手练习
```bash
# 1. 克隆并安装
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 2. 运行测试验证环境
uv run pytest --strict-markers -m "not manual" -x -q 2>&1 | head -20

# 3. 浏览项目目录结构
find src/openpi -type f -name "*.py" | head -30
```

---

## 阶段二：核心数据结构与类型系统

### 学习目标
- 理解模型输入输出的数据结构：`Observation`、`Actions`
- 理解 `jaxtyping` + `beartype` 的数组形状标注系统
- 理解 `NormStats` 归一化统计量

### 核心阅读
| 文件 | 重点关注 |
|------|---------|
| `src/openpi/models/model.py` | `Observation` dataclass（第 83-137 行）、`Actions` 类型别名、`BaseModel` 抽象基类、`BaseModelConfig` |
| `src/openpi/shared/array_typing.py` | `Array` 类型别名（兼容 JAX/PyTorch）、`@typecheck` 装饰器 |
| `src/openpi/shared/normalize.py` | `NormStats` 数据类、Z-score 与分位数归一化 |

### 关键概念

**Observation 结构** — 模型接收的所有输入打包在一个对象中：
```python
Observation(
    images: Dict[str, Array[*b, h, w, c]],        # 多视角 RGB 图像，值域 [-1, 1]
    image_masks: Dict[str, Array[*b]],             # 图像是否有效
    state: Array[*b, s],                           # 机器人关节状态
    tokenized_prompt: Array[*b, l],                # 分词后的语言指令
    tokenized_prompt_mask: Array[*b, l],           # 指令的 padding mask
    token_ar_mask: Array[*b, l],                   # 自回归 mask（仅 FAST）
    token_loss_mask: Array[*b, l],                 # 损失 mask（仅 FAST）
)
```

**Actions** — 模型输出的动作序列：`Array[*b, action_horizon, action_dim]`
- `action_horizon`：预测的未来时间步数（π₀ 默认 50，FAST 默认 32）
- `action_dim`：动作空间维度（默认 32）

**数组形状标注** — 使用 `jaxtyping` 在运行时检查张量维度：
```python
@typecheck
def process(x: Float[Array, "batch seq dim"]) -> Float[Array, "batch seq"]:
    ...
```

### 动手练习
```bash
# 阅读 model.py 中的数据结构定义
# 重点理解 Observation.from_dict() 如何将嵌套字典转换为结构化对象

# 运行模型相关测试，观察数据如何流转
uv run pytest src/openpi/models/model_test.py -v
```

---

## 阶段三：模型架构（JAX）

### 学习目标
- 理解 π₀ 的 Flow Matching 训练与采样过程
- 理解 π₀-FAST 的自回归训练与采样过程
- 理解 SigLIP 视觉编码和 PaliGemma 语言模型的角色
- 理解 Prefix-LM 注意力掩码机制

### 核心阅读
| 文件 | 重点关注 |
|------|---------|
| `src/openpi/models/pi0_config.py` | `Pi0Config`：模型维度、Gemma 变体选择、LoRA 配置、`get_freeze_filter()` |
| `src/openpi/models/pi0.py` | `Pi0` 类：`embed_prefix`（图像+文本编码）、`embed_suffix`（状态+动作+时间编码）、`compute_loss`（Flow Matching 损失）、`sample_actions`（迭代去噪） |
| `src/openpi/models/pi0_fast.py` | `Pi0FAST` 类：`embed_inputs`（统一 token 化）、`compute_loss`（交叉熵）、`sample_actions`（自回归生成） |
| `src/openpi/models/siglip.py` | SigLIP 视觉编码器，将图像转换为 patch token |
| `src/openpi/models/gemma.py` | Gemma 变体定义（gemma_2b、gemma_300m）、NNX Bridge 用法 |
| `src/openpi/models/tokenizer.py` | `PaligemmaTokenizer`（文本分词）、`FASTTokenizer`（动作量化 + 文本分词） |
| `src/openpi/models/lora.py` | LoRA 的 `Einsum` 和 `FeedForward` 替换实现 |

### 关键概念

**π₀ Flow Matching 训练过程：**
```
1. 采样噪声 ε ~ N(0, I)，时间步 t ~ Beta(1.5, 1)
2. 前向插值：x_t = t * ε + (1 - t) * actions
3. 目标速度：u_t = ε - actions
4. 模型预测速度 v_t = model(observation, x_t, t)
5. 损失：MSE(v_t, u_t)
```

**π₀ 采样（推理）过程：**
```
1. 初始化 x_1 ~ N(0, I)（纯噪声）
2. 用 KV cache 缓存 prefix（图像 + 文本）的注意力
3. 循环 t 从 1.0 到 0.0（步长 dt = -1/num_steps）：
   a. 编码 suffix（状态 + x_t + t）
   b. 模型预测 v_t
   c. 去噪：x_t = x_t + dt * v_t
4. 返回 x_0（去噪后的动作序列）
```

**π₀-FAST 自回归过程：**
```
训练：对动作 token 序列做 next-token prediction（交叉熵损失）
推理：逐 token 生成，支持 temperature 采样和 EOS 提前停止
```

**Prefix-LM 注意力掩码：**
```
             prefix tokens    suffix tokens (actions)
prefix:    [  双向注意力  ] [         不可见          ]
suffix:    [  可见 prefix ] [  因果注意力（仅看左侧） ]
```
这使得图像和文本 token 之间相互可见，而动作 token 只能看到之前的 token。实现在 `pi0.py` 的 `make_attn_mask()` 函数中。

**模型组合方式（π₀）：**
```
Pi0
├── PaliGemma（视觉-语言模型，2B 参数）
│   ├── SigLIP：图像 → patch token 嵌入
│   └── Gemma-2B：语言理解 + 多模态融合
├── Action Expert（Gemma-300M）：专门的动作预测头
├── action_in_proj / action_out_proj：动作空间的线性投影
├── state_proj：状态嵌入投影
└── action_time_mlp：时间步嵌入 MLP
```

### 动手练习
```bash
# 运行模型配置测试，理解 LoRA 冻结过滤器的行为
uv run pytest src/openpi/models/pi0_test.py -v

# 阅读 tokenizer 测试，理解文本和动作的 token 化过程
uv run pytest src/openpi/models/tokenizer_test.py -v
```

**阅读建议：** 先读 `pi0_config.py` 理解配置，再读 `pi0.py` 的 `compute_loss` 理解训练，最后读 `sample_actions` 理解推理。`pi0_fast.py` 用同样的顺序阅读。

---

## 阶段四：数据变换管线

### 学习目标
- 理解 `DataTransformFn` 协议和 `Group` 组合模式
- 理解数据如何从原始格式变换为模型输入
- 理解归一化（Normalize）和反归一化（Unnormalize）的流程
- 理解机器人特定的变换（Aloha、DROID、LIBERO）

### 核心阅读
| 文件 | 重点关注 |
|------|---------|
| `src/openpi/transforms.py` | `DataTransformFn` 协议、`Group` 容器、`RepackTransform`（键映射）、`Normalize/Unnormalize`、`ResizeImages`、`TokenizePrompt`、`DeltaActions/AbsoluteActions` |
| `src/openpi/policies/aloha_policy.py` | `AlohaInputs/Outputs`：摄像头命名映射、夹爪角度转换 |
| `src/openpi/policies/droid_policy.py` | `DroidInputs/Outputs`：DROID 特定的图像和状态映射 |
| `src/openpi/policies/libero_policy.py` | `LiberoInputs/Outputs`：LIBERO 仿真器的数据格式适配 |

### 关键概念

**变换管线的执行顺序：**
```
原始数据 dict
  ↓ RepackTransform     — 将数据集的键名映射到统一格式
  ↓ 机器人特定变换       — AlohaInputs / DroidInputs 等
  ↓ DeltaActions        — 绝对动作 → 相对动作（如需要）
  ↓ Normalize           — Z-score 或分位数归一化
  ↓ TokenizePrompt      — 语言指令 → token ID 序列
  ↓ ResizeImages        — 调整图像为 224×224 并 padding
  ↓ PadStatesAndActions — 填充到固定维度
  → 模型可接受的输入
```

**反向变换（推理输出后处理）：**
```
模型输出 actions
  ↓ Unnormalize         — 还原到原始尺度
  ↓ AbsoluteActions     — 相对动作 → 绝对动作
  ↓ 机器人特定输出变换   — AlohaOutputs 等
  → 可发送给机器人的动作
```

**Group 容器：** 将输入变换和输出变换打包在一起，确保前后处理对称。

```python
data_transforms = Group(
    inputs=[AlohaInputs(adapt_to_pi=True)],
    outputs=[AlohaOutputs(adapt_to_pi=True)],
)
```

### 动手练习
```bash
# 运行变换相关测试
uv run pytest src/openpi/transforms_test.py -v

# 阅读 Aloha 策略测试，理解机器人特定的数据映射
uv run pytest src/openpi/policies/ -v -k "not manual"
```

---

## 阶段五：训练管线

### 学习目标
- 理解 `TrainConfig` / `DataConfig` 配置系统及其预置配置
- 理解 JAX 训练循环：数据加载 → 前向/反向 → 优化器更新 → 检查点保存
- 理解权重加载策略（从零训练、加载预训练、LoRA 微调）
- 理解 FSDP 分片和多 GPU 训练

### 核心阅读
| 文件 | 重点关注 |
|------|---------|
| `src/openpi/training/config.py` | `TrainConfig` 所有字段含义、`DataConfig` 工厂模式、预置配置列表（`_CONFIGS`）、`get_config()` 函数 |
| `src/openpi/training/data_loader.py` | `Dataset/IterableDataset` 协议、`TorchDataLoader`、`TransformedDataset`、`FakeDataset` |
| `scripts/train.py` | JAX 训练主循环：初始化 → mesh 创建 → 数据加载 → JIT 编译训练步 → 循环训练 → 检查点保存 |
| `src/openpi/training/weight_loaders.py` | `WeightLoader` 协议、`CheckpointWeightLoader`（加载已有检查点）、`PaliGemmaWeightLoader`（从 PaliGemma 初始化） |
| `src/openpi/training/checkpoints.py` | Orbax `CheckpointManager` 使用、异步保存、自动清理 |
| `src/openpi/training/sharding.py` | FSDP 分片策略 |

### 关键概念

**配置系统层次：**
```
TrainConfig
├── name: str                    — 配置标识符（如 "pi0_aloha_sim"）
├── model: BaseModelConfig       — 模型架构配置
├── data: DataConfigFactory      — 数据加载配置工厂
│   ├── repo_id: str             — LeRobot 数据集 ID
│   ├── repack_transforms        — 键名映射
│   ├── data_transforms          — 机器人特定变换
│   └── model_transforms         — 模型特定变换（分词、缩放）
├── weight_loader: WeightLoader  — 预训练权重来源
├── batch_size: int              — 全局 batch size
├── num_train_steps: int         — 总训练步数
├── freeze_filter: Filter        — LoRA 时冻结哪些参数
└── fsdp_devices: int            — FSDP 分片设备数
```

**训练循环（JAX）关键步骤：**
```python
# 1. 创建 mesh（多设备分片）
mesh = create_mesh(config.fsdp_devices)

# 2. 加载预训练权重
params = weight_loader.load(model_params)

# 3. 初始化优化器和训练状态
train_state = init_train_state(params, optimizer)

# 4. JIT 编译训练步
@jax.jit
def train_step(state, batch):
    loss, grads = jax.value_and_grad(model.compute_loss)(...)
    state = state.apply_gradients(grads)
    return state, loss

# 5. 主循环
for step in range(num_train_steps):
    batch = next(data_iter)
    train_state, info = train_step(train_state, batch)
    # 定期保存检查点和日志
```

**权重加载策略：**
| 场景 | 使用的 WeightLoader | 说明 |
|------|---------------------|------|
| 从零训练 | `NoOpWeightLoader` | 随机初始化 |
| 加载预训练 | `CheckpointWeightLoader` | 从本地/GCS 加载 |
| PaliGemma 初始化 | `PaliGemmaWeightLoader` | 保留视觉-语言权重，随机初始化动作头 |

**检查点保存内容：**
- `train_state/`：完整训练状态（参数 + 优化器状态 + EMA）
- `params/`：仅推理参数（部署用）
- `assets/`：归一化统计量

### 动手练习
```bash
# 阅读训练配置，列出所有预置配置名
uv run python -c "from openpi.training.config import _CONFIGS; print([c.name for c in _CONFIGS])"

# 运行训练相关测试（CPU 模式）
uv run pytest scripts/train_test.py -v
```

---

## 阶段六：策略封装与推理服务

### 学习目标
- 理解 `Policy` 类如何将模型和变换统一封装
- 理解 `create_trained_policy()` 工厂函数的完整流程
- 理解 WebSocket 推理服务器的工作方式
- 理解客户端库的使用方式和 Runtime 抽象

### 核心阅读
| 文件 | 重点关注 |
|------|---------|
| `src/openpi/policies/policy.py` | `Policy.infer()` 方法：变换 → 批处理 → 转 Observation → 采样 → 反变换 |
| `src/openpi/policies/policy_config.py` | `create_trained_policy()`：加载检查点 → 检测 JAX/PyTorch → 组装变换 → 创建 Policy |
| `scripts/serve_policy.py` | 服务启动入口：解析参数 → 创建策略 → 启动 WebSocket 服务器 |
| `src/openpi/serving/websocket_policy_server.py` | WebSocket 协议：连接握手 → 接收观测 → 推理 → 返回动作 |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | 客户端：WebSocket 连接 → msgpack 序列化 → 发送/接收 |
| `packages/openpi-client/src/openpi_client/runtime/runtime.py` | `Runtime` 循环：`get_observation() → infer() → apply_action()` |

### 关键概念

**Policy.infer() 完整数据流：**
```
输入: obs dict（原始观测）
  ↓ input_transforms（变换链）
  ↓ 添加 batch 维度
  ↓ 转换为 JAX/PyTorch 张量
  ↓ Observation.from_dict()
  ↓ model.sample_actions()
  ↓ 转回 NumPy
  ↓ 移除 batch 维度
  ↓ output_transforms（反变换链）
输出: actions dict（可执行的动作）
```

**create_trained_policy() 做了什么：**
```
1. 从检查点目录检测模型格式（JAX: params/ 目录，PyTorch: model.safetensors 文件）
2. 加载模型权重
3. 加载归一化统计量
4. 组装变换链：
   - 输入：Repack → DataTransforms.inputs → Normalize → ModelTransforms.inputs
   - 输出：ModelTransforms.outputs → Unnormalize → DataTransforms.outputs
5. 返回 Policy 对象
```

**远程推理架构：**
```
机器人/仿真器
    ↕ (observation / action)
openpi-client (WebSocketClientPolicy)
    ↕ (msgpack over WebSocket)
serve_policy.py (WebSocketPolicyServer)
    ↕
Policy (model + transforms)
```

**Runtime 抽象（客户端库）：**
```python
runtime = Runtime(
    environment=MyRobotEnv(),         # 实现 get_observation(), apply_action()
    agent=PolicyAgent(policy=policy), # 封装推理逻辑
    subscribers=[VideoSaver()],       # 观察者：录像、日志等
    max_hz=50,                        # 控制频率
)
runtime.run()
```

### 动手练习
```bash
# 阅读一个完整的端到端示例
cat examples/aloha_sim/main.py

# 阅读客户端库测试
uv run pytest packages/openpi-client/ -v -k "not manual"
```

---

## 阶段七：PyTorch 实现与高级主题

### 学习目标
- 理解 PyTorch 实现如何镜像 JAX 版本
- 理解 DDP 多 GPU 训练的设置
- 理解 `torch.compile` 优化
- 理解 LoRA 微调的完整流程

### 核心阅读
| 文件 | 重点关注 |
|------|---------|
| `src/openpi/models_pytorch/pi0_pytorch.py` | `PI0Pytorch` 类：与 JAX Pi0 的对应关系、`torch.compile` 使用、梯度检查点 |
| `src/openpi/models_pytorch/gemma_pytorch.py` | `PaliGemmaWithExpertModel`：HuggingFace Transformers 集成 |
| `scripts/train_pytorch.py` | DDP 初始化 → DataLoader → 训练循环 → safetensors 保存 |
| `src/openpi/models/lora.py` | LoRA Einsum 替换、rank-stabilized LoRA |
| `examples/libero/README.md` | 完整的 LoRA 微调流程示例 |

### 关键概念

**JAX vs PyTorch 对照：**

| 方面 | JAX 实现 | PyTorch 实现 |
|------|---------|-------------|
| 模型基类 | Flax NNX Module | `nn.Module` |
| 参数管理 | PyTree + `nnx.split/merge` | `state_dict` |
| 编译优化 | `jax.jit` | `torch.compile` |
| 多 GPU | FSDP via mesh | DDP via `torchrun` |
| 检查点格式 | Orbax | safetensors |
| 精度 | bfloat16（默认） | bfloat16（GPU）/ float32（CPU） |

**LoRA 微调关键点：**
- `Pi0Config.get_freeze_filter()` 返回 NNX 过滤器，标记哪些参数可训练
- LoRA 模块替换了 Gemma 的 `Einsum` 和 `FeedForward`
- 缩放因子：标准模式 `alpha/rank`，RS-LoRA 模式 `alpha/sqrt(rank)`
- 支持对 PaliGemma 和 Action Expert 分别配置 LoRA

### 动手练习
```bash
# 对比 JAX 和 PyTorch 的模型定义
# 重点关注 compute_loss 和 sample_actions 的实现差异

# 运行 PyTorch 训练测试
uv run pytest scripts/train_test.py -v -k "pytorch"
```

---

## 核心数据流详解

### 完整链路：从原始数据到机器人执行

```
═══════════════════════════════════════════════════════════
 阶段 1: 数据准备
═══════════════════════════════════════════════════════════

LeRobot 数据集（HuggingFace 格式）
│  包含: images, state, action, language_instruction
│
├─ RepackTransform ──────────────────────────────────────
│  将数据集字段映射到统一格式:
│  "observation.images.top" → "image/base_0_rgb"
│  "observation.state"      → "state"
│  "action"                 → "actions"
│
├─ 机器人特定变换 ────────────────────────────────────────
│  AlohaInputs: 摄像头名映射, 夹爪角度 → 线性距离
│  DroidInputs: 外部/腕部摄像头映射
│
├─ DeltaActions（如需要）──────────────────────────────────
│  将绝对关节角度转换为相邻帧差值
│
├─ Normalize ─────────────────────────────────────────────
│  使用预计算的 NormStats 进行归一化:
│  Z-score: (x - mean) / (std + ε)
│  或分位数: (x - q01) / (q99 - q01 + ε) * 2 - 1
│
├─ TokenizePrompt ────────────────────────────────────────
│  "pick up the cube" → [token_ids], padding to max_len
│
├─ ResizeImages ──────────────────────────────────────────
│  任意尺寸 → 224×224, 保持比例 + padding
│
└─ PadStatesAndActions ───────────────────────────────────
   填充到固定的 action_dim (32)

═══════════════════════════════════════════════════════════
 阶段 2: 训练
═══════════════════════════════════════════════════════════

批量数据 (Observation, Actions) [batch_size, ...]
│
├─ π₀ Flow Matching:
│  t ~ Beta(1.5, 1)
│  x_t = t * noise + (1-t) * actions
│  loss = MSE(model(obs, x_t, t), noise - actions)
│
└─ π₀-FAST 自回归:
   loss = CrossEntropy(model(obs, tokens[:-1]), tokens[1:])

检查点保存: params/ + assets/(norm_stats)

═══════════════════════════════════════════════════════════
 阶段 3: 部署推理
═══════════════════════════════════════════════════════════

加载检查点 → create_trained_policy() → Policy 对象
│
├─ 本地推理: policy.infer(obs_dict) → action_dict
│
└─ 远程推理:
   serve_policy.py → WebSocket Server (:8000)
   客户端 → WebSocketClientPolicy → msgpack 通信
      → policy.infer(obs) → actions
      → 发送回客户端

═══════════════════════════════════════════════════════════
 阶段 4: 机器人执行
═══════════════════════════════════════════════════════════

Runtime 循环 (max_hz=50):
  obs = env.get_observation()
  action = policy.infer(obs)    # 包含 action_horizon 步
  env.apply_action(action[0])   # 执行第一步（或使用 ActionChunkBroker）
```

---

## 关键设计模式

### 1. 冻结 Dataclass 配置
所有配置使用 `@dataclasses.dataclass(frozen=True)`，不可变，预置配置实例直接定义在模块级别。
```python
# src/openpi/training/config.py
_CONFIGS = [
    TrainConfig(name="pi0_aloha", model=Pi0Config(...), ...),
    TrainConfig(name="pi0_aloha_sim", ...),
    ...
]
```
通过 `get_config(name)` 获取，支持 `tyro` 命令行覆盖。

### 2. Protocol 接口
核心抽象使用 `typing.Protocol` 而非继承，实现鸭子类型：
- `DataTransformFn`：`(DataDict) → DataDict`
- `Dataset`：`__getitem__` + `__len__`
- `IterableDataset`：`__iter__`
- `WeightLoader`：`load(params) → params`

### 3. 输入/输出变换对称
每个机器人平台的变换成对出现（Group），确保推理时的反变换正确还原：
```python
Group(
    inputs=[AlohaInputs(adapt_to_pi=True)],    # 预处理
    outputs=[AlohaOutputs(adapt_to_pi=True)],   # 后处理（逆操作）
)
```

### 4. JAX/PyTorch 双框架透明切换
`Policy` 类内部根据 `is_pytorch` 标志自动处理张量转换，上层调用者无需关心底层框架。检查点格式通过目录内容自动检测。

### 5. GCS 资产缓存
预训练权重和归一化统计量存放在 `gs://openpi-assets`，通过 `shared/download.py` 自动下载并缓存到 `~/.cache/openpi`（可通过 `OPENPI_DATA_HOME` 覆盖）。

### 6. KV Cache 加速推理
π₀ 的 `sample_actions` 将图像和文本的 prefix 注意力缓存起来，去噪循环中只重新计算 suffix（动作 token），大幅减少计算量。

---

## 常用命令速查

```bash
# ──── 环境 ────
GIT_LFS_SKIP_SMUDGE=1 uv sync          # 安装依赖
pre-commit install                       # 安装 pre-commit 钩子

# ──── 代码质量 ────
ruff check .                             # 检查代码风格
ruff format .                            # 格式化代码
ruff check . && ruff format .            # 提交前必做

# ──── 测试 ────
uv run pytest --strict-markers -m "not manual"     # CI 测试（排除需 GPU 的测试）
uv run pytest src/openpi/models/model_test.py      # 单文件测试
uv run pytest -v -k "test_pi0"                     # 按名称过滤测试

# ──── 训练 ────
# JAX 训练（需设置 XLA 内存比例）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<name>

# PyTorch 单 GPU
uv run scripts/train_pytorch.py <config> --exp_name <name>

# PyTorch 多 GPU
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<n> scripts/train_pytorch.py <config>

# ──── 推理 ────
# 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config> --policy.dir=<checkpoint_path>

# ──── 数据 ────
# 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name <config>
```
