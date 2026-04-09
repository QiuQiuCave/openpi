# 第 5 周：模型结构层

> **目标**：深入理解 π₀ 和 π₀-FAST 的模型架构、训练损失和推理采样过程，建立对模型输入输出契约的精确认知。

---

## 1. 背景知识

### 1.1 Flow Matching（流匹配）

Flow Matching 是一种生成模型方法，核心思想是学习一个"速度场"，把简单分布（噪声）平滑地变换为复杂分布（真实动作）。

```
时间 t=1           时间 t=0
纯噪声 ─────────→ 真实动作
      沿速度场流动
```

**训练时**：
1. 取真实动作 x₀ 和随机噪声 ε
2. 线性插值：x_t = t × ε + (1-t) × x₀
3. 模型预测速度：v_t = model(x_t, t)
4. 真实速度：u_t = ε - x₀（从动作指向噪声的方向）
5. 损失：MSE(v_t, u_t)

**推理时**：
1. 从纯噪声 x₁ 开始
2. 循环：x_t → x_{t-dt}，每步走 dt
3. 更新：x_{t-dt} = x_t + dt × model(x_t, t)
4. 最终得到 x₀（去噪后的动作）

### 1.2 自回归生成（Autoregressive）

自回归是 ChatGPT 等语言模型使用的方法：逐个预测序列中的下一个元素。

```
输入:  [图像, 文本, 状态, token_1, token_2, ...]
输出:  [                  token_2, token_3, ..., EOS]
```

**π₀-FAST** 用 FAST tokenizer 把连续动作量化为离散 token，然后用自回归方式生成。

### 1.3 Prefix-LM 注意力

OpenPI 使用一种混合注意力模式：

```
Token 类型:   [图像] [文本]  |  [动作1] [动作2] [动作3]
              ← 前缀(prefix) →  ← 后缀(suffix) →

注意力规则:
  前缀 token: 可以看到所有其他前缀 token（双向）
  后缀 token: 可以看到所有前缀 + 左侧的后缀（因果）
```

这比纯因果注意力更有效，因为图像和文本之间的信息可以自由流动。

---

## 2. 核心数据结构

### 2.1 ModelType 枚举

打开 `src/openpi/models/model.py`：

```python
# src/openpi/models/model.py 第 30-35 行
class ModelType(enum.Enum):
    PI0 = "pi0"           # Flow Matching
    PI0_FAST = "pi0_fast" # 自回归
    PI05 = "pi05"         # Flow Matching（增强版）
```

### 2.2 Observation —— 模型输入

```python
# src/openpi/models/model.py 约第 83-137 行
@struct.dataclass
class Observation(Generic[ArrayT]):
    # 多视角 RGB 图像，值域 [-1, 1]
    images: dict[str, ArrayT]          # {"base_0_rgb": [b, 224, 224, 3], ...}
    image_masks: dict[str, ArrayT]     # {"base_0_rgb": [b], ...} 布尔掩码

    # 机器人状态（关节角度等）
    state: ArrayT                       # [b, state_dim]

    # 语言指令（已分词）
    tokenized_prompt: ArrayT | None     # [b, max_token_len] 整数
    tokenized_prompt_mask: ArrayT | None # [b, max_token_len] 布尔

    # 仅 FAST 模型使用
    token_ar_mask: ArrayT | None        # [b, max_token_len] 自回归掩码
    token_loss_mask: ArrayT | None      # [b, max_token_len] 损失掩码
```

**关键约定**：
- 图像必须有 3 个：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`
- 图像分辨率：224×224
- 图像值域：[-1, 1]（uint8 图像在 `from_dict` 中自动转换）

### 2.3 Actions —— 模型输出

```python
# Actions 就是一个数组类型别名
Actions = ArrayT   # 形状 [batch, action_horizon, action_dim]
```

- `action_horizon`：预测未来多少步（π₀ 默认 50，FAST 默认 32）
- `action_dim`：每步动作的维度（默认 32）

### 2.4 BaseModel —— 抽象基类

```python
# src/openpi/models/model.py 约第 263-284 行
class BaseModel(nnx.Module, abc.ABC):
    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(self, rng, observation, actions, *, train=True):
        """计算训练损失"""
        ...

    @abc.abstractmethod
    def sample_actions(self, rng, observation, **kwargs):
        """推理时生成动作"""
        ...
```

每种模型都必须实现这两个方法。这就是模型的"契约"。

---

## 3. Pi0Config —— π₀ 模型配置

打开 `src/openpi/models/pi0_config.py`：

```python
# src/openpi/models/pi0_config.py 约第 19-75 行
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    # 动作空间
    action_dim: int = 32              # 动作维度
    action_horizon: int = 50          # 预测时域（未来多少步）

    # 语言
    max_token_len: int = 48           # π₀ 最大 token 长度
    # pi05=True 时变为 200（π₀.5 支持更长的指令）

    # Gemma 模型变体
    paligemma_variant: str = "gemma_2b"     # 视觉-语言模型（2B 参数）
    action_expert_variant: str = "gemma_300m" # 动作专家（300M 参数）

    # LoRA（低秩适应，用于微调）
    paligemma_lora: str | None = None       # 如 "gemma_2b_lora"
    action_expert_lora: str | None = None   # 如 "gemma_300m_lora"

    # π₀.5 开关
    pi05: bool = False
    discrete_state_input: bool = False  # pi05=True 时自动开启

    # PyTorch 编译模式
    pytorch_compile_mode: str | None = None  # "default", "reduce-overhead", "max-autotune"
```

**`get_freeze_filter()` 方法**：决定 LoRA 微调时哪些参数可训练。

```python
    def get_freeze_filter(self):
        """返回 LoRA 时的冻结过滤器"""
        # 如果没有启用 LoRA，所有参数都可训练
        if self.paligemma_lora is None and self.action_expert_lora is None:
            return nnx.Everything()

        # 启用 LoRA 后，只有 LoRA 参数可训练
        return nnx.All(nnx.Param, LoraParam)
```

---

## 4. π₀ 模型详解

打开 `src/openpi/models/pi0.py`：

### 4.1 模型组成

```python
# src/openpi/models/pi0.py 约第 66-100 行
class Pi0(BaseModel):
    def __init__(self, config: Pi0Config, *, rngs):
        # 组件 1: PaliGemma（视觉+语言）
        self.PaliGemma = {
            "llm": ...,    # Gemma-2B 语言模型
            "img": ...,    # SigLIP 视觉编码器
        }

        # 组件 2: Action Expert（动作生成专家）
        self.action_expert = Gemma(variant="gemma_300m")

        # 组件 3: 投影层
        self.action_in_proj = Linear(action_dim → model_width)   # 动作输入投影
        self.action_out_proj = Linear(model_width → action_dim)  # 动作输出投影
        self.state_proj = Linear(action_dim → model_width)       # 状态投影

        # 组件 4: 时间嵌入
        self.action_time_mlp_in = Linear(...)   # 时间步 → 嵌入
        self.action_time_mlp_out = Linear(...)
```

**架构图**：

```
                    PaliGemma (2B)
                 ┌─────────────────┐
  图像 ──→ SigLIP ──→ patch tokens │
                 │                 │──→ prefix tokens
  文本 ──→ Gemma embedding ──────→ │
                 └─────────────────┘
                                          ↓
                                   ┌──────────────┐
  状态 ──→ state_proj ──→          │              │
                                   │  Transformer  │──→ action tokens
  动作 ──→ action_in_proj ──→      │   (Gemma)    │
                                   │              │
  时间 t ──→ time_mlp ──→          │              │
                                   └──────────────┘
                                          ↓
                                   action_out_proj
                                          ↓
                                    预测速度 v_t
```

### 4.2 embed_prefix —— 编码图像和文本

```python
# src/openpi/models/pi0.py 约第 106-137 行
def embed_prefix(self, observation):
    # 1. 用 SigLIP 编码图像 → patch tokens
    #    每张 224x224 图像 → 256 个 token
    img_tokens = self.PaliGemma["img"](observation.images)

    # 2. 用 Gemma embedding 编码文本 token
    txt_tokens = self.PaliGemma["llm"].embed(observation.tokenized_prompt)

    # 3. 拼接：[img_tokens, txt_tokens]
    prefix_tokens = concat(img_tokens, txt_tokens)

    # 4. 创建注意力掩码（prefix 内部双向）
    prefix_ar_mask = zeros(...)  # 全 0 = 双向注意力

    return prefix_tokens, prefix_mask, prefix_ar_mask
```

### 4.3 embed_suffix —— 编码状态和动作

```python
# src/openpi/models/pi0.py 约第 140-186 行
def embed_suffix(self, observation, noisy_actions, timestep):
    # 1. 编码状态
    state_token = self.state_proj(observation.state)  # [b, 1, width]

    # 2. 编码时间步（正弦-余弦位置编码）
    time_emb = sinusoidal_embedding(timestep)     # [b, width]
    time_emb = self.action_time_mlp_in(time_emb)
    time_emb = silu(time_emb)
    time_emb = self.action_time_mlp_out(time_emb) # [b, width]

    # 3. 编码带噪动作 + 加上时间嵌入
    action_tokens = self.action_in_proj(noisy_actions)  # [b, horizon, width]
    action_tokens = action_tokens + time_emb[:, None, :]  # 广播加

    # 4. 拼接：[state_token, action_tokens]
    suffix_tokens = concat(state_token, action_tokens)

    # 5. 创建注意力掩码（suffix 内部因果）
    suffix_ar_mask = ones(...)  # 全 1 = 因果注意力

    return suffix_tokens, suffix_mask, suffix_ar_mask
```

### 4.4 compute_loss —— 训练损失

```python
# src/openpi/models/pi0.py 约第 189-214 行
def compute_loss(self, rng, observation, actions, *, train=True):
    # 1. 采样噪声和时间步
    noise = jax.random.normal(rng1, actions.shape)
    timestep = jax.random.beta(rng2, 1.5, 1.0)   # Beta(1.5, 1) 分布
    # 为什么用 Beta 分布？让模型更多关注接近噪声端（t≈1）的去噪

    # 2. 前向扩散：在真实动作和噪声之间插值
    x_t = timestep * noise + (1 - timestep) * actions

    # 3. 目标速度（从动作到噪声的方向）
    u_t = noise - actions

    # 4. 模型预测速度
    prefix = self.embed_prefix(observation)
    suffix = self.embed_suffix(observation, x_t, timestep)
    v_t = self.forward(prefix, suffix)   # 通过 Transformer
    v_t = self.action_out_proj(v_t)      # 投影到动作空间

    # 5. MSE 损失
    loss = mean_squared_error(v_t, u_t)  # [batch, action_horizon]
    return loss
```

### 4.5 sample_actions —— 推理采样

```python
# src/openpi/models/pi0.py 约第 217-279 行
def sample_actions(self, rng, observation, *, num_steps=10, **kwargs):
    # 1. 编码 prefix 并缓存 KV（只算一次！）
    prefix = self.embed_prefix(observation)
    kv_cache = self.fill_kv_cache(prefix)

    # 2. 从纯噪声开始
    x_t = jax.random.normal(rng, [batch, action_horizon, action_dim])

    # 3. 迭代去噪
    dt = -1.0 / num_steps   # 时间步长（从 1 走向 0）
    t = 1.0

    for _ in range(num_steps):
        # 3a. 编码当前噪声动作
        suffix = self.embed_suffix(observation, x_t, t)

        # 3b. 用缓存的 KV 高效计算（不需要重新编码 prefix）
        v_t = self.forward_with_cache(kv_cache, suffix)
        v_t = self.action_out_proj(v_t)

        # 3c. 沿速度场前进一步
        x_t = x_t + dt * v_t
        t = t + dt

    return x_t   # 去噪后的动作序列
```

**KV Cache 的作用**：prefix（图像+文本）在去噪过程中不变，只需要计算一次。后续每步只需要重新计算 suffix（动作）部分，节省大量计算。

---

## 5. π₀-FAST 模型详解

打开 `src/openpi/models/pi0_fast.py`：

### 5.1 与 π₀ 的关键差异

| 方面 | π₀ | π₀-FAST |
|------|-----|---------|
| 动作表示 | 连续浮点数 | 离散 token |
| 生成方式 | 迭代去噪（10-20步） | 逐 token 自回归 |
| Gemma | 双模型（2B + 300M） | 单模型（2B） |
| 训练损失 | MSE | 交叉熵 |
| 默认 action_horizon | 50 | 32 |
| 速度 | 较慢（多步去噪） | 较快（单次前向） |

### 5.2 Pi0FASTConfig

```python
# src/openpi/models/pi0_fast.py 约第 77-131 行
@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(BaseModelConfig):
    action_dim: int = 32
    action_horizon: int = 32          # 比 π₀ 短
    max_token_len: int = 250          # 比 π₀ 长（需要放动作 token）
    paligemma_variant: str = "gemma_2b"
    # 没有 action_expert！FAST 只用一个 Gemma
```

### 5.3 FAST Tokenizer

```python
# src/openpi/models/tokenizer.py 约第 51-140 行
class FASTTokenizer:
    """将连续动作量化为离散 token"""

    def tokenize(self, prompt, state):
        # 1. 文本 prefix: "Task: pick up the cube, State: [0.1, 0.2, ...];\n"
        prefix_tokens = self.paligemma_tokenizer.tokenize(prompt, state)

        # 2. 动作 token: 用 FAST 编码器量化连续动作
        action_tokens = self.fast_encoder.encode(actions)
        # 连续动作 [50, 32] → 离散 token [128]（大幅压缩！）

        # 3. 映射到 PaliGemma 词表的最后 128 个位置
        action_tokens = action_tokens + (vocab_size - 128)

        # 4. 拼接: [prefix_tokens, "Action: ", action_tokens, "|"]
        return full_tokens, masks
```

### 5.4 compute_loss —— 交叉熵损失

```python
# src/openpi/models/pi0_fast.py 约第 198-233 行
def compute_loss(self, rng, observation, actions, *, train=True):
    # 1. 编码所有输入
    tokens = self.embed_inputs(observation)

    # 2. 前向传播得到 logits
    logits = self.forward(tokens)   # [batch, seq_len, vocab_size]

    # 3. 移位：用位置 i 的输出预测位置 i+1 的 token
    predictions = logits[:, :-1, :]           # [batch, seq_len-1, vocab_size]
    targets = observation.tokenized_prompt[:, 1:]  # [batch, seq_len-1]

    # 4. 只对动作 token 计算损失（忽略文本部分）
    loss_mask = observation.token_loss_mask[:, 1:]

    # 5. 交叉熵损失
    loss = cross_entropy(predictions, targets) * loss_mask
    return loss
```

### 5.5 sample_actions —— 自回归采样

```python
# src/openpi/models/pi0_fast.py 约第 236-313 行
def sample_actions(self, rng, observation, *, temperature=0.0, **kwargs):
    # 1. 左对齐输入（batch 中不同样本长度可能不同）
    tokens = left_to_right_align(observation.tokenized_prompt, ...)

    # 2. 填充 KV cache
    kv_cache = self.fill_kv_cache(tokens)

    # 3. 逐 token 生成
    generated_tokens = []
    for i in range(max_new_tokens):
        # 预测下一个 token
        logits = self.forward_one_step(kv_cache, last_token)

        if temperature == 0:
            next_token = argmax(logits)   # 贪心
        else:
            next_token = sample(logits / temperature)  # 温度采样

        generated_tokens.append(next_token)

        # 如果生成了 EOS token，提前停止
        if next_token == eos_token:
            break

    # 4. 用 FAST decoder 将离散 token 解码为连续动作
    actions = self.fast_decoder.decode(generated_tokens)
    return actions   # [batch, action_horizon, action_dim]
```

---

## 6. π₀ vs π₀-FAST vs π₀.5 对比表

| 特性 | π₀ | π₀-FAST | π₀.5 |
|------|-----|---------|------|
| **配置类** | `Pi0Config()` | `Pi0FASTConfig()` | `Pi0Config(pi05=True)` |
| **模型类** | `Pi0` | `Pi0FAST` | `Pi0`（同一个类） |
| **动作生成** | Flow Matching（连续） | 自回归（离散 token） | Flow Matching（连续） |
| **训练损失** | MSE（速度预测） | 交叉熵（下一 token） | MSE（速度预测） |
| **Gemma 模型** | 2B + 300M（双模型） | 2B（单模型） | 2B + 300M（双模型） |
| **状态输入** | 连续（`state_proj`） | 离散（tokenized） | 离散（tokenized） |
| **max_token_len** | 48 | 250 | 200 |
| **action_horizon** | 50 | 32 | 50 |
| **action_dim** | 32 | 32 | 32 |
| **推理步数** | 10-20步（迭代去噪） | 1步（逐 token） | 10-20步（迭代去噪） |
| **KV Cache** | prefix 缓存 | prefix 缓存 | prefix 缓存 |
| **LoRA 支持** | PaliGemma + Expert | PaliGemma | PaliGemma + Expert |
| **时间嵌入** | MLP | 无 | adaRMSNorm |
| **图像命名** | base_0, left_wrist, right_wrist | base_0, base_1, wrist | base_0, left_wrist, right_wrist |
| **适用场景** | 通用、高精度 | 快速推理 | 开放世界泛化 |

**π₀.5 与 π₀ 的关键区别**：
1. 状态输入是离散的（tokenized），而不是连续向量
2. 使用 adaRMSNorm 代替 MLP 进行时间条件化
3. max_token_len 更长（200 vs 48）
4. 知识隔离设计，泛化能力更强

---

## 7. Tokenizer 详解

打开 `src/openpi/models/tokenizer.py`：

### 7.1 PaligemmaTokenizer（文本分词）

```python
# src/openpi/models/tokenizer.py 约第 14-48 行
class PaligemmaTokenizer:
    def tokenize(self, prompt: str, state=None):
        if state is not None:
            # π₀.5 格式（状态作为离散 token）
            text = f"Task: {prompt}, State: {state_str};\nAction: "
        else:
            # π₀ 格式
            text = f"Task: {prompt}\nAction: "

        tokens = self.sp_model.encode(text)  # SentencePiece 分词
        tokens = pad_to(tokens, max_token_len)  # 填充到固定长度
        return tokens, mask
```

### 7.2 FASTTokenizer（动作分词）

```python
# 约第 51-140 行
class FASTTokenizer:
    def tokenize(self, prompt, state):
        # Prefix: "Task: pick up cube, State: [0.1, 0.2];\n"
        # Action tokens: [fast_encoded_actions]
        # Postfix: "|"

        # AR mask: [0,0,0,..., 1,1,1,...]
        # prefix 部分=0（双向），action 部分=1（因果）

        # Loss mask: [0,0,0,..., 1,1,1,...]
        # 只对 action token 计算损失
```

---

## 8. 运行命令指南

```bash
# 运行模型相关测试
uv run pytest src/openpi/models/model_test.py -v

# 运行 π₀ 配置测试（LoRA 冻结等）
uv run pytest src/openpi/models/pi0_test.py -v

# 运行 tokenizer 测试
uv run pytest src/openpi/models/tokenizer_test.py -v

# 查看模型参数量（粗略估计）
uv run python -c "
from openpi.models.pi0_config import Pi0Config
config = Pi0Config()
print(f'action_dim: {config.action_dim}')
print(f'action_horizon: {config.action_horizon}')
print(f'PaliGemma variant: {config.paligemma_variant}')
print(f'Action expert variant: {config.action_expert_variant}')
print(f'max_token_len: {config.max_token_len}')
"
```

---

## 9. 本周产出

### 产出：π₀ vs π₀-FAST vs π₀.5 对比分析

请整理第 6 节的对比表，并补充以下思考：

1. **什么时候选 π₀？** 需要高精度动作（如精细操作），不在意推理速度
2. **什么时候选 π₀-FAST？** 需要快速推理（实时控制），可以接受稍低精度
3. **什么时候选 π₀.5？** 需要在新环境/新物体上泛化，对 prompt 理解要求高

---

## 10. 本周自检清单

- [ ] 能解释 Flow Matching 的训练和推理过程
- [ ] 能解释自回归生成的训练和推理过程
- [ ] 理解 Prefix-LM 注意力掩码的含义
- [ ] 能说出 Observation 和 Actions 的具体字段和形状
- [ ] 理解 π₀ 的 `compute_loss` 中 5 个步骤
- [ ] 理解 π₀ 的 `sample_actions` 中 KV Cache 的作用
- [ ] 理解 FAST Tokenizer 如何将连续动作转为离散 token
- [ ] 能完成三种模型的对比分析
