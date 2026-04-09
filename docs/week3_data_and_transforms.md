# 第 3 周：数据与 Transforms

> **目标**：理解数据如何从原始格式一步步变换为模型可接受的输入，掌握归一化统计量的计算和使用。

---

## 1. 背景知识：数据预处理为什么复杂？

### 1.1 问题的根源

不同来源的数据格式不同，但模型需要统一的输入格式。比如：

```
ALOHA 数据集:                        模型期望的格式:
{                                    {
  "observation": {                     "image": {
    "images": {                          "base_0_rgb": [224,224,3],
      "cam_high": [480,640,3],           "left_wrist_0_rgb": [224,224,3],
      "cam_left_wrist": [480,640,3]      "right_wrist_0_rgb": [224,224,3]
    },                                 },
    "state": [14]                      "state": [32],   ← 填充到32维
  },                                   "tokenized_prompt": [200],
  "action": [14]                       "actions": [50, 32]  ← 填充
}                                    }
```

这个转换涉及：字段重命名、图像缩放、状态填充、文本分词、归一化......

### 1.2 变换管线（Transform Pipeline）的设计思想

OpenPI 用"管线"模式处理这个问题：每个变换做一件小事，然后串联起来。

```
原始数据 → [变换1] → [变换2] → [变换3] → ... → 模型输入
```

就像工厂的流水线，每一站做一个加工步骤。

---

## 2. transforms.py 核心概念

打开 `src/openpi/transforms.py`，这是整个变换系统的核心文件。

### 2.1 DataTransformFn 协议

```python
# src/openpi/transforms.py 约第 23-36 行
class DataTransformFn(Protocol):
    """数据变换函数的协议（接口）"""
    def __call__(self, data: DataDict) -> DataDict:
        ...
```

**小白翻译**：任何实现了 `__call__(data) -> data` 的类都是一个"变换"。`DataDict` 就是一个可能嵌套的字典，叶子节点是 numpy 数组。

### 2.2 Group —— 变换容器

```python
# src/openpi/transforms.py 约第 39-59 行
@dataclasses.dataclass(frozen=True)
class Group:
    """一组变换，包含输入变换和输出变换"""
    inputs: Sequence[DataTransformFn] = ()    # 输入时执行的变换
    outputs: Sequence[DataTransformFn] = ()   # 输出时执行的变换（反向操作）

    def push(self, *, inputs=(), outputs=()):
        """添加新变换，返回新的 Group"""
        return Group(
            inputs=(*self.inputs, *inputs),    # 新输入变换加到末尾
            outputs=(*outputs, *self.outputs), # 新输出变换加到开头（注意顺序！）
        )
```

**为什么输出变换加到开头？** 因为输出变换需要反向执行。如果输入是 A→B→C，输出就应该是 C'→B'→A'。`push` 方法自动保证这个顺序。

### 2.3 CompositeTransform —— 串联执行

```python
# src/openpi/transforms.py 约第 62-71 行
class CompositeTransform(DataTransformFn):
    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)     # 依次执行每个变换
        return data

def compose(transforms):
    """将多个变换组合成一个"""
    return CompositeTransform(transforms)
```

---

## 3. 核心变换类详解

### 3.1 RepackTransform —— 字段重命名

```python
# src/openpi/transforms.py 约第 79-101 行
@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """将数据集的字段名映射到标准名"""
    structure: dict  # 映射表

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)   # 先把嵌套字典展平
        return jax.tree.map(lambda k: flat_item[k], self.structure)
```

**实际使用例子**：

```python
# ALOHA 数据的字段重命名
RepackTransform({
    "images": "observation/images",           # observation.images → images
    "state": "observation/state",             # observation.state → state
    "actions": "action",                      # action → actions
    "prompt": "task/language_instruction",    # task.language_instruction → prompt
})
```

变换前后对比：
```
变换前: {"observation": {"images": {...}, "state": [14]}, "action": [14]}
变换后: {"images": {...}, "state": [14], "actions": [14]}
```

### 3.2 Normalize / Unnormalize —— 归一化

```python
# src/openpi/transforms.py 约第 126-139 行
class Normalize(DataTransformFn):
    def _normalize(self, x, stats: NormStats):
        mean = stats.mean[..., :x.shape[-1]]
        std = stats.std[..., :x.shape[-1]]
        return (x - mean) / (std + 1e-6)      # Z-score 归一化

    def _normalize_quantile(self, x, stats: NormStats):
        q01 = stats.q01[..., :x.shape[-1]]
        q99 = stats.q99[..., :x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2 - 1  # 映射到 [-1, 1]

# 反归一化（推理输出后使用）
class Unnormalize(DataTransformFn):
    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean         # 还原到原始尺度
```

**为什么归一化很重要？**
- 模型训练时，如果不同维度的数值范围差很多（比如角度 0-3 rad vs 位置 0-1000 mm），梯度会不稳定
- 归一化后所有维度的值都在差不多的范围内

**两种归一化方式**：
| 方式 | 公式 | 适用场景 |
|------|------|---------|
| Z-score | (x - mean) / std | 数据近似正态分布 |
| 分位数 | (x - q01) / (q99 - q01) * 2 - 1 | 数据有异常值 |

### 3.3 TokenizePrompt —— 语言指令分词

```python
# src/openpi/transforms.py 约第 252-266 行
class TokenizePrompt(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        prompt = data.pop("prompt", None)   # 取出语言指令字符串
        if prompt is None:
            raise ValueError("Prompt is required")

        # 使用 PaliGemma 分词器将文本转为 token ID 序列
        tokens, token_masks = self.tokenizer.tokenize(prompt, state)

        return {
            **data,
            "tokenized_prompt": tokens,           # [max_token_len] 的整数数组
            "tokenized_prompt_mask": token_masks,  # [max_token_len] 的布尔数组
        }
```

**小白翻译**：把人类可读的文字（如 "pick up the red cube"）转换成模型可理解的数字序列。

### 3.4 ResizeImages —— 图像缩放

```python
# src/openpi/transforms.py 约第 184-191 行
class ResizeImages(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        # 将所有图像缩放到 224x224，保持比例，不足部分用黑色填充
        data["image"] = jax.tree.map(
            lambda img: image_tools.resize_with_pad(img, height, width),
            data["image"],
        )
        return data
```

---

## 4. 机器人特定变换

### 4.1 DroidInputs —— DROID 机器人适配

打开 `src/openpi/policies/droid_policy.py`：

```python
# src/openpi/policies/droid_policy.py 约第 30-74 行
class DroidInputs(DataTransformFn):
    model_type: ModelType    # 需要知道模型类型来决定图像命名

    def __call__(self, data: dict) -> dict:
        # 1. 拼接关节位置和夹爪位置为 state
        state = np.concatenate([
            data["observation/joint_position"],
            data["observation/gripper_position"],
        ])

        # 2. 解析图像
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        # 3. 根据模型类型选择不同的图像命名方案
        match self.model_type:
            case ModelType.PI0 | ModelType.PI05:
                # π₀ 用 base + left_wrist + right_wrist（缺少的用零填充）
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case ModelType.PI0_FAST:
                # FAST 用 base_0 + base_1 + wrist（不掩码）
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)

        return {
            "state": state,
            "image": dict(zip(names, images)),
            "image_mask": dict(zip(names, image_masks)),
        }
```

### 4.2 DroidOutputs

```python
# 约第 77-81 行
class DroidOutputs(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 模型输出 32 维，DROID 只需要前 8 维（7关节+1夹爪）
        return {"actions": np.asarray(data["actions"][:, :8])}
```

### 4.3 AlohaInputs

打开 `src/openpi/policies/aloha_policy.py`：

```python
# src/openpi/policies/aloha_policy.py 约第 24-87 行
class AlohaInputs(DataTransformFn):
    adapt_to_pi: bool = True   # 是否将 Aloha 的值域转换为 Pi 内部格式

    EXPECTED_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        # 1. 解码 Aloha 特定格式（夹爪角度→线性距离等）
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        # 2. 映射摄像头名称
        images = {
            "base_0_rgb": data["images"]["cam_high"],
            "left_wrist_0_rgb": data["images"].get("cam_left_wrist", zeros),
            "right_wrist_0_rgb": data["images"].get("cam_right_wrist", zeros),
        }
        # 缺少的摄像头用零图像替代，并设置 mask=False

        return {"image": images, "image_mask": masks, "state": data["state"]}
```

### 4.4 LiberoInputs

打开 `src/openpi/policies/libero_policy.py`：

```python
# src/openpi/policies/libero_policy.py 约第 29-83 行
class LiberoInputs(DataTransformFn):
    """最好的学习模板！注释非常详细，适合作为自定义适配器的起点"""
    model_type: ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),  # 没有右腕摄像头
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == ModelType.PI0_FAST else np.False_,
            },
        }
        return inputs
```

**学习要点**：`libero_policy.py` 的注释写得最详细，专门指导你如何为自己的数据集写适配器。

---

## 5. 归一化统计量（Norm Stats）

### 5.1 NormStats 数据结构

打开 `src/openpi/shared/normalize.py`：

```python
# src/openpi/shared/normalize.py 约第 9-14 行
@pydantic.dataclasses.dataclass
class NormStats:
    mean: NDArray       # 均值，形状 [dim]
    std: NDArray        # 标准差，形状 [dim]
    q01: NDArray | None = None  # 第 1 百分位数（可选）
    q99: NDArray | None = None  # 第 99 百分位数（可选）
```

每个需要归一化的字段（如 `state`、`actions`）都有一组统计量。

### 5.2 RunningStats —— 流式计算统计量

```python
# 约第 17-86 行
class RunningStats:
    """流式计算大数据集的统计量（不需要一次全部加载到内存）"""

    def update(self, batch: np.ndarray) -> None:
        """用一个 batch 更新统计量"""
        # 增量更新均值：new_mean = old_mean + (batch_mean - old_mean) * (n/N)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        # 同时更新方差和直方图（用于计算分位数）

    def get_statistics(self) -> NormStats:
        """返回当前的统计量"""
        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)
```

### 5.3 保存和加载

```python
# 约第 134-146 行
def save(directory, norm_stats: dict[str, NormStats]) -> None:
    """保存归一化统计量到 JSON 文件"""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))

def load(directory) -> dict[str, NormStats]:
    """从 JSON 文件加载归一化统计量"""
    path = pathlib.Path(directory) / "norm_stats.json"
    return deserialize_json(path.read_text())
```

### 5.4 完整的使用链路

```
1. 计算阶段（训练前）:
   compute_norm_stats.py → RunningStats.update() → save()
   保存到: assets/{config_name}/{asset_id}/norm_stats.json

2. 训练阶段:
   load() → Normalize(norm_stats) → 作为变换插入管线

3. 推理阶段:
   create_trained_policy() → load_norm_stats() → Normalize + Unnormalize
```

---

## 6. compute_norm_stats.py 详解

打开 `scripts/compute_norm_stats.py`：

```python
# scripts/compute_norm_stats.py 约第 89-113 行
def main(config_name: str, max_frames: int | None = None):
    # 1. 获取配置
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 2. 创建数据加载器（只应用 repack + data transforms，不做归一化）
    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, ...
    )

    # 3. 遍历所有数据，计算 state 和 actions 的统计量
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    # 4. 保存统计量
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    normalize.save(output_path, norm_stats)
```

**注意**：计算归一化统计量时，只应用了 `repack_transforms` 和 `data_transforms`（字段重命名 + 机器人适配），**不**应用归一化本身（否则就是"用归一化的数据计算归一化参数"，逻辑错误）。

---

## 7. 完整的变换管线图

把所有变换串联起来，数据从原始格式到模型输入的完整路径：

```
原始数据（LeRobot 格式）
│
│  字段: observation/images/cam_high, observation/state, action, ...
│
├── [RepackTransform] ─────────────────────────────────
│   observation/images → images
│   observation/state  → state
│   action             → actions
│
├── [InjectDefaultPrompt] ─────────────────────────────
│   如果数据中没有 prompt，注入默认指令
│   "prompt": "pick up the cube"
│
├── [机器人特定变换 Inputs] ────────────────────────────
│   AlohaInputs:  cam_high → base_0_rgb, 夹爪角度转换
│   DroidInputs:  exterior_image → base_0_rgb, 拼接 state
│   LiberoInputs: observation/image → base_0_rgb
│
├── [Normalize] ────────────────────────────────────────
│   state:   (x - mean) / (std + 1e-6)
│   actions: (x - mean) / (std + 1e-6)
│   (图像不做归一化，在 Observation.from_dict 中处理)
│
├── [TokenizePrompt] ───────────────────────────────────
│   "pick up the cube" → [12043, 892, 3341, ...]
│   生成 tokenized_prompt 和 tokenized_prompt_mask
│
├── [ResizeImages] ─────────────────────────────────────
│   任意尺寸 → 224×224，保持比例 + 黑色填充
│
├── [PadStatesAndActions] ──────────────────────────────
│   state [14] → state [32]  （后面补零）
│   actions [horizon, 14] → actions [horizon, 32]
│
└── 模型输入就绪！
    {
        "image": {"base_0_rgb": [224,224,3], ...},
        "image_mask": {"base_0_rgb": True, ...},
        "state": [32],
        "tokenized_prompt": [200],
        "tokenized_prompt_mask": [200],
        "actions": [50, 32]    ← 仅训练时有
    }
```

---

## 8. 运行命令指南

### 8.1 计算归一化统计量

```bash
# 为 Aloha 仿真数据集计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim

# 限制最大帧数（加速，用于测试）
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim --max-frames 1000
```

### 8.2 运行变换相关测试

```bash
# 运行 transforms 的单元测试
uv run pytest src/openpi/transforms_test.py -v

# 运行策略适配器的测试
uv run pytest src/openpi/policies/ -v -k "not manual"

# 运行归一化工具的测试
uv run pytest src/openpi/shared/normalize_test.py -v
```

### 8.3 调试变换管线

```bash
# 查看变换前后的数据变化
uv run python -c "
from openpi.training.config import get_config
import numpy as np

config = get_config('pi0_aloha_sim')
data_config = config.data.create(config.assets_dirs, config.model)

print('Repack transforms (输入):')
for t in data_config.repack_transforms.inputs:
    print(f'  {type(t).__name__}')

print('Data transforms (输入):')
for t in data_config.data_transforms.inputs:
    print(f'  {type(t).__name__}')

print('Model transforms (输入):')
for t in data_config.model_transforms.inputs:
    print(f'  {type(t).__name__}')

print('Data transforms (输出):')
for t in data_config.data_transforms.outputs:
    print(f'  {type(t).__name__}')
"
```

---

## 9. 本周产出

### 产出 1：数据字段映射表

| 阶段 | ALOHA 字段 | DROID 字段 | LIBERO 字段 | 标准字段 |
|------|-----------|-----------|------------|---------|
| 原始 | observation/images/cam_high | observation/exterior_image_1_left | observation/image | - |
| RepackTransform 后 | images/cam_high | 同上（无 repack） | 同上（无 repack） | - |
| 机器人适配后 | image/base_0_rgb | image/base_0_rgb | image/base_0_rgb | `image/base_0_rgb` |
| 原始 | observation/state [14] | observation/joint_position [7] + gripper [1] | observation/state [7] | - |
| 机器人适配后 | state [14] | state [8] | state [7] | `state` |
| Pad 后 | state [32] | state [32] | state [32] | `state [action_dim]` |

### 产出 2：Norm Stats 流转图

```
计算: compute_norm_stats.py
  ↓ 遍历数据集，增量计算 mean/std/q01/q99
  ↓ 保存到 assets/{config}/{asset_id}/norm_stats.json

训练时加载:
  config.py → DataConfig → asset_id
  ↓ normalize.load(assets_dir / asset_id)
  ↓ Normalize(norm_stats) 插入输入变换链

推理时加载:
  policy_config.py → create_trained_policy()
  ↓ load_norm_stats(checkpoint_dir / "assets", asset_id)
  ↓ Normalize(norm_stats) + Unnormalize(norm_stats) 分别插入输入/输出变换链
```

---

## 10. 本周自检清单

- [ ] 能解释 DataTransformFn 协议的含义
- [ ] 能说出 Group 类的 push 方法为什么输出变换加到开头
- [ ] 能描述 RepackTransform 的作用
- [ ] 理解 Normalize 和 Unnormalize 的数学公式
- [ ] 能说出三种机器人适配器（Droid/Aloha/Libero）的主要区别
- [ ] 理解 compute_norm_stats.py 的工作流程
- [ ] 知道 norm_stats.json 存在哪里、怎么加载
- [ ] 能画出完整的变换管线图
