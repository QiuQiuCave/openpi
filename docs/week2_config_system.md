# 第 2 周：配置系统（最关键的一周）

> **目标**：深入理解 OpenPI 的配置系统 —— 这是整个代码库的"中枢神经"，几乎所有功能都从配置开始。

---

## 1. 背景知识：为什么配置系统这么重要？

### 1.1 机器学习项目的配置问题

一个 ML 项目有大量可调参数：

- **模型参数**：用哪个模型、多少层、多少维度
- **数据参数**：用哪个数据集、怎么预处理
- **训练参数**：学习率、batch size、训练多少步
- **部署参数**：从哪里加载权重、用什么设备

如果这些参数散落在代码各处，维护起来会非常痛苦。OpenPI 用 **frozen dataclass**（不可变数据类）把所有参数集中管理。

### 1.2 什么是 frozen dataclass？

```python
import dataclasses

# 普通 dataclass —— 创建后可以修改
@dataclasses.dataclass
class MutableConfig:
    name: str = "default"

c = MutableConfig()
c.name = "new_name"   # ✅ 可以修改

# frozen dataclass —— 创建后不能修改
@dataclasses.dataclass(frozen=True)
class FrozenConfig:
    name: str = "default"

c = FrozenConfig()
c.name = "new_name"   # ❌ 报错！FrozenInstanceError
```

**为什么用 frozen？** 防止意外修改配置导致难以调试的 bug。配置一旦创建就是"只读"的。

### 1.3 什么是 tyro？

OpenPI 使用 `tyro` 库将 dataclass 自动转换为命令行参数。你定义一个 dataclass，tyro 就自动生成对应的 `--参数名 值` 命令行接口。

```python
# 定义
@dataclasses.dataclass
class Args:
    port: int = 8000
    name: str = "default"

# 使用
args = tyro.cli(Args)
# 命令行：python script.py --port 9000 --name my_exp
```

---

## 2. 配置系统的三层结构

OpenPI 的配置分为三层，从上到下依次是：

```
┌──────────────────────────────────────────┐
│           TrainConfig（顶层）              │
│  "一个完整的实验需要什么"                    │
│  ├── name: 配置名称                       │
│  ├── model: BaseModelConfig              │
│  ├── data: DataConfigFactory             │
│  ├── weight_loader: WeightLoader         │
│  ├── batch_size, num_train_steps, ...    │
│  └── ...                                 │
├──────────────────────────────────────────┤
│           DataConfig（数据层）              │
│  "数据从哪来、怎么处理"                     │
│  ├── repo_id: 数据集 ID                   │
│  ├── repack_transforms: 字段重映射        │
│  ├── data_transforms: 机器人特定变换       │
│  ├── model_transforms: 模型特定变换        │
│  └── norm_stats: 归一化统计量              │
├──────────────────────────────────────────┤
│        AssetsConfig（资产层）               │
│  "预计算的资产（如 norm_stats）在哪"         │
│  ├── assets_dir: 资产目录                  │
│  └── asset_id: 资产子目录 ID               │
└──────────────────────────────────────────┘
```

---

## 3. 深读 TrainConfig

打开 `src/openpi/training/config.py`，找到 `TrainConfig` 类（约第 465 行）。

### 3.1 关键字段逐一解读

```python
# src/openpi/training/config.py 约第 465-535 行
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # ──── 标识 ────
    name: tyro.conf.Suppress[str]         # 配置名称，如 "pi0_aloha_sim"
                                           # Suppress 表示不出现在命令行参数中
    project_name: str = "openpi"           # wandb 项目名
    exp_name: str = tyro.MISSING           # 实验名（必须在命令行指定）

    # ──── 模型 ────
    model: _model.BaseModelConfig          # 模型配置（决定用 π₀ 还是 π₀-FAST）

    # ──── 权重 ────
    weight_loader: WeightLoader            # 预训练权重从哪加载
    pytorch_weight_path: str | None = None # PyTorch 权重路径（可选）

    # ──── 优化器 ────
    lr_schedule: LRScheduleConfig          # 学习率调度策略
    optimizer: OptimizerConfig             # 优化器（Adam 等）
    ema_decay: float | None = 0.99         # 指数移动平均衰减率

    # ──── 冻结（LoRA 微调时用）────
    freeze_filter: Filter                  # 哪些参数冻结不训练

    # ──── 数据 ────
    data: DataConfigFactory                # 数据配置工厂（注意是工厂，不是直接的配置）

    # ──── 路径 ────
    assets_base_dir: str = "./assets"      # 资产根目录
    checkpoint_base_dir: str = "./checkpoints"  # 检查点根目录

    # ──── 训练超参 ────
    seed: int = 42                         # 随机种子
    batch_size: int = 32                   # 全局 batch size
    num_workers: int = 2                   # 数据加载线程数
    num_train_steps: int = 30_000          # 总训练步数

    # ──── 日志与保存 ────
    log_interval: int = 100                # 每 100 步打印日志
    save_interval: int = 1000              # 每 1000 步保存检查点
    keep_period: int | None = 5000         # 每 5000 步永久保留一个检查点

    # ──── 其他 ────
    resume: bool = False                   # 是否从上次中断处恢复
    wandb_enabled: bool = True             # 是否启用 wandb 日志
    fsdp_devices: int = 1                  # FSDP 分布式训练设备数
```

### 3.2 两个重要的计算属性

```python
    @property
    def assets_dirs(self) -> pathlib.Path:
        # 资产目录 = assets_base_dir / config_name
        # 例如：./assets/pi0_aloha_sim/
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        # 检查点目录 = checkpoint_base_dir / config_name / exp_name
        # 例如：./checkpoints/pi0_aloha_sim/my_experiment/
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()
```

**小白提示**：`@property` 让方法可以像属性一样使用，不需要加括号调用。

---

## 4. 深读 DataConfig

### 4.1 DataConfig 类

```python
# src/openpi/training/config.py 约第 64-99 行
@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None              # LeRobot 数据集 ID
                                             # 例如 "lerobot/aloha_sim_transfer_cube_human"
    asset_id: str | None = None             # 归一化统计量子目录
                                             # 例如 "trossen"（Aloha 用的）

    # 三种变换（下周会详细学习）
    repack_transforms: Group                 # 字段重命名
    data_transforms: Group                   # 机器人特定处理
    model_transforms: Group                  # 模型特定处理（分词、缩放）

    use_quantile_norm: bool = False          # 是否用分位数归一化

    action_sequence_keys: Sequence[str] = ("actions",)  # 哪些字段包含动作序列
    prompt_from_task: bool = False           # 是否从数据集任务标签获取指令
```

### 4.2 DataConfigFactory 模式

注意 `TrainConfig.data` 的类型是 `DataConfigFactory`，不是 `DataConfig`。这是一个"工厂模式"：

```python
# 工厂模式的好处：延迟创建，可以在创建时注入额外参数
data_config = train_config.data.create(
    assets_dirs,   # 资产目录
    model_config,  # 模型配置（需要知道模型信息才能构建正确的变换）
)
```

常见的工厂类型：
- `SimpleDataConfig` —— 通用数据配置
- `LeRobotAlohaDataConfig` —— Aloha 机器人专用
- `LeRobotLiberoDataConfig` —— LIBERO 仿真器专用
- `FakeDataConfig` —— 生成假数据（调试用）

### 4.3 AssetsConfig 类

```python
# src/openpi/training/config.py 约第 37-62 行
@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """资产配置：归一化统计量等预计算文件的位置"""
    assets_dir: str | None = None    # 完整路径覆盖
    asset_id: str | None = None      # 资产子目录 ID
```

资产目录结构：
```
assets/
└── pi0_aloha_sim/              ← config.assets_dirs
    └── trossen/                ← asset_id
        └── norm_stats.json     ← 归一化统计量
```

---

## 5. _CONFIGS 注册机制

### 5.1 所有配置都在哪？

在 `config.py` 文件底部（约第 560 行开始），有一个巨大的列表 `_CONFIGS`：

```python
# src/openpi/training/config.py 约第 560 行
_CONFIGS = [
    # ──── Aloha 相关 ────
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),   # pi05=True 开启 π₀.5 模式
        ...
    ),

    # ──── DROID 相关 ────
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        ...
    ),

    # ──── LIBERO 相关 ────
    TrainConfig(
        name="pi0_libero",
        model=pi0_config.Pi0Config(),
        ...
    ),

    # ──── 微调配置 ────
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        weight_loader=CheckpointWeightLoader(...),  # 从预训练加载
        num_train_steps=20_000,
        ...
    ),
    ...
]
```

### 5.2 从列表到字典

```python
# 列表转字典，方便按名称查找
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}
```

### 5.3 调用路径

```
用户输入: "pi05_droid"
    ↓
get_config("pi05_droid")
    ↓
_CONFIGS_DICT["pi05_droid"]
    ↓
返回对应的 TrainConfig 对象
```

---

## 6. 各机器人平台的数据适配

### 6.1 为什么需要适配？

不同机器人平台的数据格式不同：

| 平台 | 摄像头名称 | 状态维度 | 动作维度 |
|------|-----------|---------|---------|
| **ALOHA** | cam_high, cam_left_wrist, cam_right_wrist | 14（7关节×2臂） | 14 |
| **DROID** | exterior_image_1_left, wrist_image_left | 8（7关节+夹爪） | 8 |
| **LIBERO** | observation/image, observation/wrist_image | 7 | 7 |

但模型统一使用：
```
图像名：base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
状态维度：填充到 action_dim（32）
```

所以需要"适配器"来转换格式。

### 6.2 Aloha 适配配置

```python
# 简化的 LeRobotAlohaDataConfig
class LeRobotAlohaDataConfig(DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return DataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            # 字段重映射：LeRobot 格式 → OpenPI 格式
            repack_transforms=Group(inputs=[
                RepackTransform({
                    "images": "observation.images",   # 图像
                    "state": "observation.state",      # 关节状态
                    "actions": "action",               # 动作
                }),
            ]),
            # Aloha 特定变换
            data_transforms=Group(
                inputs=[AlohaInputs(adapt_to_pi=True)],
                outputs=[AlohaOutputs(adapt_to_pi=True)],
            ),
        )
```

### 6.3 DROID 适配配置

```python
# 简化的 DROID 数据配置
TrainConfig(
    name="pi0_droid",
    model=pi0_config.Pi0Config(action_horizon=10),   # DROID 用更短的动作时域
    data=SimpleDataConfig(
        data_transforms=lambda model: Group(
            inputs=[DroidInputs(model_type=model.model_type)],
            outputs=[DroidOutputs()],
        ),
    ),
)
```

### 6.4 LIBERO 适配配置

```python
# LIBERO 配置特点：语言条件任务
TrainConfig(
    name="pi0_libero",
    data=LeRobotLiberoDataConfig(
        # LIBERO 的每个任务有不同的语言指令
        # prompt_from_task=True 表示从数据集元数据中获取指令
    ),
)
```

---

## 7. 推理配置 vs 微调配置 vs 调试配置

### 7.1 配置分类表

```bash
# 运行这个命令查看所有配置
uv run python -c "
from openpi.training.config import _CONFIGS
for c in _CONFIGS:
    has_wl = 'pretrained' if not isinstance(c.weight_loader, type(c.weight_loader)) else 'scratch'
    print(f'{c.name:30s}  steps={c.num_train_steps:6d}  batch={c.batch_size:3d}')
"
```

根据用途，配置可以分为三类：

| 类型 | 典型配置名 | 特征 | 用途 |
|------|-----------|------|------|
| **推理配置** | `pi0_aloha`, `pi05_droid` | 无 weight_loader | 仅用于加载预训练模型进行推理 |
| **微调配置** | `pi0_aloha_sim`, `pi0_libero` | 有 weight_loader + 较少 steps | 在预训练模型上微调 |
| **全量训练** | 自定义 | 无 weight_loader + 大量 steps | 从零训练（很少用到） |

### 7.2 调试用的 FakeDataConfig

```python
# 不需要真实数据，生成随机数据用于调试
TrainConfig(
    name="pi0_debug",
    data=FakeDataConfig(),    # 生成假数据
    num_train_steps=10,       # 只跑 10 步
    wandb_enabled=False,      # 关闭日志
)
```

---

## 8. 动手实验：探索配置系统

### 8.1 查看所有配置

```bash
uv run python -c "
from openpi.training.config import _CONFIGS
print(f'共有 {len(_CONFIGS)} 个预置配置\n')
for c in _CONFIGS:
    print(f'  {c.name}')
"
```

### 8.2 对比两个配置

```bash
uv run python -c "
from openpi.training.config import get_config

# 对比 pi0 和 pi05 的区别
c1 = get_config('pi0_aloha')
c2 = get_config('pi05_aloha')

print('=== pi0_aloha ===')
print(f'  model type: {c1.model.model_type}')
print(f'  action_dim: {c1.model.action_dim}')
print(f'  action_horizon: {c1.model.action_horizon}')
print(f'  max_token_len: {c1.model.max_token_len}')

print('\n=== pi05_aloha ===')
print(f'  model type: {c2.model.model_type}')
print(f'  action_dim: {c2.model.action_dim}')
print(f'  action_horizon: {c2.model.action_horizon}')
print(f'  max_token_len: {c2.model.max_token_len}')
"
```

### 8.3 查看数据配置

```bash
uv run python -c "
from openpi.training.config import get_config

c = get_config('pi0_aloha_sim')
data_config = c.data.create(c.assets_dirs, c.model)

print(f'数据集 repo_id: {data_config.repo_id}')
print(f'资产 ID: {data_config.asset_id}')
print(f'使用分位数归一化: {data_config.use_quantile_norm}')
print(f'从任务获取指令: {data_config.prompt_from_task}')
print(f'输入变换数量: {len(data_config.repack_transforms.inputs)}')
print(f'数据变换数量: {len(data_config.data_transforms.inputs)}')
print(f'模型变换数量: {len(data_config.model_transforms.inputs)}')
"
```

### 8.4 测试拼写纠错

```bash
# 故意输入错误的配置名
uv run python -c "
from openpi.training.config import get_config
try:
    get_config('pi0_aloh')   # 拼错了
except ValueError as e:
    print(f'错误信息: {e}')
"
# 输出类似: Config 'pi0_aloh' not found. Did you mean 'pi0_aloha'?
```

---

## 9. 本周产出

### 产出 1：新增 config 的 checklist

当你需要为一个新的机器人平台或数据集新增配置时，需要修改/新增以下内容：

**必须修改的字段：**

- [ ] `name` —— 给配置起个唯一的名字，格式建议 `{model}_{platform}`
- [ ] `model` —— 选择 `Pi0Config()` 或 `Pi0Config(pi05=True)` 等
- [ ] `data` —— 创建一个 `DataConfigFactory`，指定：
  - [ ] `repo_id` —— LeRobot 数据集的 Hugging Face 路径
  - [ ] `asset_id` —— 归一化统计量的目录名
  - [ ] `repack_transforms` —— 你的数据集字段名到标准名的映射
  - [ ] `data_transforms` —— 你的机器人特定的输入/输出变换
  - [ ] `model_transforms` —— 通常不需要自定义（使用默认即可）

**按需修改的字段：**

- [ ] `weight_loader` —— 如果从预训练模型微调，指定检查点路径
- [ ] `batch_size` —— 根据 GPU 显存调整
- [ ] `num_train_steps` —— 微调一般 10k-30k 步
- [ ] `action_horizon` —— 不同机器人可能需要不同的时域
- [ ] `freeze_filter` —— LoRA 微调时指定冻结哪些参数

**新增后别忘了：**

- [ ] 把新配置添加到 `_CONFIGS` 列表中
- [ ] 编写对应的 `DataTransformFn`（如 `MyRobotInputs`/`MyRobotOutputs`）
- [ ] 计算归一化统计量：`uv run scripts/compute_norm_stats.py --config-name your_config`

### 产出 2：常用 config 列表

| 配置名 | 模型 | 平台 | 用途 | 关键特征 |
|--------|------|------|------|---------|
| `pi0_aloha` | π₀ | ALOHA | 推理 | 基础 Aloha 配置 |
| `pi05_aloha` | π₀.5 | ALOHA | 推理 | π₀.5 升级版 |
| `pi0_aloha_sim` | π₀ | ALOHA仿真 | 微调 | 带预训练权重加载 |
| `pi0_droid` | π₀ | DROID | 推理 | action_horizon=10 |
| `pi05_droid` | π₀.5 | DROID | 推理 | π₀.5 + DROID |
| `pi0_libero` | π₀ | LIBERO | 微调 | 语言条件任务 |
| `pi05_libero` | π₀.5 | LIBERO | 微调 | π₀.5 + LIBERO |

---

## 10. 本周自检清单

- [ ] 能解释 `frozen=True` 的含义和好处
- [ ] 能说出 TrainConfig 的 5 个最重要字段
- [ ] 理解 DataConfig 和 DataConfigFactory 的区别
- [ ] 能描述 `get_config()` 的调用路径
- [ ] 知道 `_CONFIGS` 列表在哪里、怎么扩展
- [ ] 能区分推理配置和微调配置
- [ ] 能整理出"新增一个 config"需要哪些步骤
