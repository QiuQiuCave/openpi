# 第 8 周：综合实战（读码层面）

> **目标**：综合运用前 7 周的知识，完成两个设计任务——"新数据集微调方案"和"新机器人适配方案"，并查缺补漏。

---

## 1. 回顾：你已经掌握的知识体系

```
第 1 周 ─ 全局认知        你知道了 OpenPI 是什么、推理链路怎么走
第 2 周 ─ 配置系统        你知道了配置怎么组织、怎么扩展
第 3 周 ─ 数据变换        你知道了数据怎么从原始格式变成模型输入
第 4 周 ─ 训练循环        你知道了训练脚本的完整流程
第 5 周 ─ 模型架构        你知道了 π₀ 和 π₀-FAST 的内部结构
第 6 周 ─ 服务部署        你知道了推理服务器和客户端怎么协作
第 7 周 ─ PyTorch         你知道了双框架实现的异同
```

本周将把这些知识串联起来，完成两个综合设计任务。

---

## 2. 实战任务 A：新数据集微调方案

### 2.1 场景设定

假设你有一个自己采集的机器人数据集，已经上传到 HuggingFace 的 LeRobot 格式：

```
数据集信息：
- HuggingFace repo: "my_lab/my_robot_data"
- 机器人: 6 自由度机械臂 + 1 夹爪 = 7 维动作
- 摄像头: 1 个外部摄像头（top_cam）、1 个腕部摄像头（wrist_cam）
- 状态: 7 维（6 关节角 + 1 夹爪开合度）
- 任务: 语言条件的物体操作（带 language_instruction 字段）
- 数据量: 100 个 episode，每个约 500 步
```

### 2.2 设计步骤

#### 步骤 1: 选择基础模型

**决策依据**（回顾第 5 周）：

| 选项 | 适合场景 | 本例选择 |
|------|---------|---------|
| π₀ | 高精度操作 | ✅ 通用推荐 |
| π₀-FAST | 快速推理 | 可选 |
| π₀.5 | 开放世界泛化 | ✅ 如果需要泛化到新物体 |

**本例选择 π₀.5**（`Pi0Config(pi05=True)`），因为语言条件任务需要好的指令理解能力。

#### 步骤 2: 编写数据变换适配器

**需要新建文件**：`src/openpi/policies/my_robot_policy.py`

```python
# src/openpi/policies/my_robot_policy.py
import dataclasses
import numpy as np
from openpi import transforms
from openpi.models import model as _model

def _parse_image(image):
    """确保图像是 uint8 [H, W, C] 格式"""
    if image.dtype == np.float32 and image.shape[0] == 3:
        # LeRobot 格式: float32 [C, H, W] → uint8 [H, W, C]
        image = (image * 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class MyRobotInputs(transforms.DataTransformFn):
    """将自定义数据集格式转换为模型输入格式
    
    参考: src/openpi/policies/libero_policy.py（模板文件）
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 1. 解析图像（从数据集字段名映射到标准名）
        base_image = _parse_image(data["observation/top_cam"])
        wrist_image = _parse_image(data["observation/wrist_cam"])

        # 2. 构建标准输入格式
        inputs = {
            "state": np.asarray(data["observation/state"]),  # [7]
            "image": {
                "base_0_rgb": base_image,              # 外部摄像头 → base_0
                "left_wrist_0_rgb": wrist_image,       # 腕部摄像头 → left_wrist_0
                "right_wrist_0_rgb": np.zeros_like(base_image),  # 无右腕 → 零填充
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_     # π₀/π₀.5 模式下掩码填充图像
                ),
            },
        }

        # 3. 动作（仅训练时有）
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # 4. 语言指令
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MyRobotOutputs(transforms.DataTransformFn):
    """将模型输出转换回机器人可执行的动作格式"""

    def __call__(self, data: dict) -> dict:
        # 模型输出 32 维，我们的机器人只需要前 7 维
        return {"actions": np.asarray(data["actions"][:, :7])}
```

#### 步骤 3: 创建训练配置

**需要修改文件**：`src/openpi/training/config.py`

在 `_CONFIGS` 列表中添加新配置：

```python
# 在 _CONFIGS 列表中添加

# ──── 我的机器人配置 ────
TrainConfig(
    name="pi05_my_robot",
    model=pi0_config.Pi0Config(
        pi05=True,                    # 使用 π₀.5
        action_dim=32,                # 保持默认（内部会填充）
        action_horizon=50,            # 预测未来 50 步
    ),
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="my_robot"),
        base_config=DataConfig(
            repo_id="my_lab/my_robot_data",  # HuggingFace 数据集 ID
            prompt_from_task=True,            # 从数据集获取语言指令
        ),
        repack_transforms=lambda model: _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    # 数据集字段 → 标准字段
                    "observation/top_cam": "observation/top_cam",
                    "observation/wrist_cam": "observation/wrist_cam",
                    "observation/state": "observation/state",
                    "actions": "actions",
                    "prompt": "task/language_instruction",
                }),
            ],
        ),
        data_transforms=lambda model: _transforms.Group(
            inputs=[MyRobotInputs(model_type=model.model_type)],
            outputs=[MyRobotOutputs()],
        ),
    ),
    # 从预训练 π₀.5 加载权重
    weight_loader=CheckpointWeightLoader(
        params_path="gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    # LoRA 微调（推荐，省显存）
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_lora="gemma_2b_lora",
        action_expert_lora="gemma_300m_lora",
    ).get_freeze_filter(),
    # 训练超参
    batch_size=16,                    # 根据显存调整
    num_train_steps=20_000,           # 100 episode 数据，20k 步足够
    lr_schedule=LRScheduleConfig(
        warmup_steps=500,
        peak_value=2.5e-5,
    ),
    wandb_enabled=True,
),
```

#### 步骤 4: 计算归一化统计量

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_my_robot
```

这会：
1. 加载 `my_lab/my_robot_data` 数据集
2. 应用 RepackTransform + MyRobotInputs
3. 遍历所有数据计算 state 和 actions 的 mean/std/q01/q99
4. 保存到 `assets/pi05_my_robot/my_robot/norm_stats.json`

#### 步骤 5: 开始训练

```bash
# LoRA 微调（推荐，~22.5GB 显存）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_my_robot --exp-name=lora_v1

# 或 PyTorch 训练
uv run scripts/train_pytorch.py pi05_my_robot --exp_name lora_v1_pt
```

#### 步骤 6: 部署推理

```bash
# 启动推理服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_my_robot \
    --policy.dir=./checkpoints/pi05_my_robot/lora_v1/20000 \
    --default-prompt "pick up the red cube" \
    --port 8000
```

### 2.3 完整文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `src/openpi/policies/my_robot_policy.py` | 数据适配器 |
| 修改 | `src/openpi/training/config.py` | 添加配置到 `_CONFIGS` |
| 生成 | `assets/pi05_my_robot/my_robot/norm_stats.json` | 归一化统计量 |
| 生成 | `checkpoints/pi05_my_robot/lora_v1/` | 训练检查点 |

---

## 3. 实战任务 B：新机器人输入输出适配方案

### 3.1 场景设定

假设你要为一个新的机器人平台"DualArm"编写适配器：

```
机器人信息：
- 双臂机器人，每臂 7 自由度 + 夹爪 = 16 维状态/动作
- 3 个摄像头：head_cam（头部）、left_hand_cam（左手）、right_hand_cam（右手）
- 特殊需求：夹爪值需要从 [0, 1] 映射到 [-1, 1]
- 坐标系：使用弧度制，范围 [-π, π]
```

### 3.2 设计文档

#### 3.2.1 摄像头映射

```
DualArm 摄像头         →    模型标准名
head_cam               →    base_0_rgb        (主视角)
left_hand_cam          →    left_wrist_0_rgb  (左腕)
right_hand_cam         →    right_wrist_0_rgb (右腕)
```

所有 3 个摄像头都有实际图像，不需要零填充，`image_mask` 全部为 `True`。

#### 3.2.2 状态映射

```
DualArm 状态 [16]:
  [0:7]   = 左臂 7 关节角度（弧度）
  [7]     = 左夹爪 [0, 1]
  [8:15]  = 右臂 7 关节角度（弧度）
  [15]    = 右夹爪 [0, 1]

模型状态 [32]:
  [0:16]  = DualArm 状态（夹爪转换后）
  [16:32] = 零填充
```

夹爪转换：`model_gripper = dual_arm_gripper * 2 - 1`（[0,1] → [-1,1]）

#### 3.2.3 动作映射

```
模型输出 [action_horizon, 32]:
  取前 16 维 → DualArm 动作 [action_horizon, 16]
  夹爪逆转换: dual_arm_gripper = (model_gripper + 1) / 2
```

#### 3.2.4 实现代码

```python
# src/openpi/policies/dualarm_policy.py
import dataclasses
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image):
    """解析图像格式"""
    if image.dtype == np.float32 and image.ndim == 3 and image.shape[0] == 3:
        image = (image * 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
    return image


def _gripper_to_model(gripper_val):
    """夹爪值: [0, 1] → [-1, 1]"""
    return gripper_val * 2.0 - 1.0


def _gripper_from_model(model_val):
    """夹爪值: [-1, 1] → [0, 1]"""
    return np.clip((model_val + 1.0) / 2.0, 0.0, 1.0)


@dataclasses.dataclass(frozen=True)
class DualArmInputs(transforms.DataTransformFn):
    """DualArm 机器人的输入适配器"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 1. 解析图像
        head_image = _parse_image(data["observation/head_cam"])
        left_hand_image = _parse_image(data["observation/left_hand_cam"])
        right_hand_image = _parse_image(data["observation/right_hand_cam"])

        # 2. 处理状态：转换夹爪值
        state = np.asarray(data["observation/state"]).copy()  # [16]
        state[7] = _gripper_to_model(state[7])    # 左夹爪
        state[15] = _gripper_to_model(state[15])   # 右夹爪

        # 3. 构建标准输入
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": left_hand_image,
                "right_wrist_0_rgb": right_hand_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,  # 都有真实图像！
            },
        }

        # 4. 处理动作（训练时）
        if "actions" in data:
            actions = np.asarray(data["actions"]).copy()  # [horizon, 16]
            actions[:, 7] = _gripper_to_model(actions[:, 7])
            actions[:, 15] = _gripper_to_model(actions[:, 15])
            inputs["actions"] = actions

        # 5. 语言指令
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DualArmOutputs(transforms.DataTransformFn):
    """DualArm 机器人的输出适配器"""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :16]).copy()  # 取前 16 维
        # 逆转换夹爪值
        actions[:, 7] = _gripper_from_model(actions[:, 7])
        actions[:, 15] = _gripper_from_model(actions[:, 15])
        return {"actions": actions}
```

#### 3.2.5 配置注册

```python
# 在 config.py 的 _CONFIGS 中添加
TrainConfig(
    name="pi05_dualarm",
    model=pi0_config.Pi0Config(pi05=True),
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="dualarm"),
        base_config=DataConfig(
            repo_id="my_lab/dualarm_data",
            prompt_from_task=True,
        ),
        data_transforms=lambda model: _transforms.Group(
            inputs=[DualArmInputs(model_type=model.model_type)],
            outputs=[DualArmOutputs()],
        ),
    ),
    weight_loader=CheckpointWeightLoader(
        params_path="gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=16,
    num_train_steps=20_000,
),
```

#### 3.2.6 验证清单

```bash
# 1. 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi05_dualarm

# 2. 验证配置可以正常加载
uv run python -c "
from openpi.training.config import get_config
c = get_config('pi05_dualarm')
print(f'Config loaded: {c.name}')
data_config = c.data.create(c.assets_dirs, c.model)
print(f'Repo: {data_config.repo_id}')
print(f'Asset ID: {data_config.asset_id}')
"

# 3. 运行单元测试
uv run pytest src/openpi/policies/ -v -k "not manual"

# 4. 训练几步验证流程
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_dualarm \
    --exp-name=test_run \
    --num-train-steps=10 \
    --wandb-enabled=False
```

---

## 4. 设计模式总结

### 4.1 新增适配的最小改动集

```
新数据集/新机器人适配的标准流程:

1. 新建 policy 文件:
   src/openpi/policies/my_robot_policy.py
   ├── MyRobotInputs(DataTransformFn)    # 输入适配
   └── MyRobotOutputs(DataTransformFn)   # 输出适配

2. 修改 config 文件:
   src/openpi/training/config.py
   └── _CONFIGS 列表中添加新条目

3. 计算 norm stats:
   uv run scripts/compute_norm_stats.py --config-name <新配置名>

4. 训练:
   uv run scripts/train.py <新配置名> --exp-name <实验名>

5. 部署:
   uv run scripts/serve_policy.py policy:checkpoint \
       --policy.config=<新配置名> --policy.dir=<检查点路径>
```

### 4.2 编写适配器的通用模板

参考 `src/openpi/policies/libero_policy.py`，这个文件的注释最详细，专门作为自定义适配器的模板。

关键要点：
1. **图像映射**：你的摄像头名 → `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`
2. **缺失图像**：用 `np.zeros_like(base_image)` 填充 + `image_mask=False`
3. **状态处理**：值域转换（如夹爪 [0,1]→[-1,1]）
4. **动作截取**：模型输出 32 维，只取你需要的前 N 维
5. **动作逆转换**：与输入转换对称

---

## 5. 查缺补漏：你可能遗漏的知识点

### 5.1 LoRA 微调 vs 全量微调

```
LoRA 微调:
  - 只训练少量新增的低秩参数
  - 显存需求: ~22.5 GB
  - 推荐：数据量少时（< 1000 episode）

全量微调:
  - 训练所有参数
  - 显存需求: ~70 GB
  - 推荐：数据量大时（> 1000 episode）
```

### 5.2 Action Chunk 策略

模型一次预测 `action_horizon` 步，但机器人通常只执行前几步：

```python
# ActionChunkBroker 策略
class ActionChunkBroker:
    def __init__(self, policy, action_horizon=10):
        self._policy = policy
        self._horizon = action_horizon  # 每次推理后执行的步数
        self._buffer = []

    def infer(self, obs):
        if len(self._buffer) == 0:
            result = self._policy.infer(obs)
            self._buffer = list(result["actions"][:self._horizon])
        return {"actions": self._buffer.pop(0)}
```

### 5.3 多任务训练

一个配置可以包含多个数据集的混合：

```python
DataConfig(
    repo_id=None,  # 多数据集时不使用单一 repo
    datasets=[
        {"repo_id": "dataset_A", "weight": 0.7},
        {"repo_id": "dataset_B", "weight": 0.3},
    ],
)
```

---

## 6. 运行命令汇总（完整参考）

```bash
# ──── 环境搭建 ────
GIT_LFS_SKIP_SMUDGE=1 uv sync
pre-commit install

# ──── 代码质量 ────
ruff check . && ruff format .
uv run pytest --strict-markers -m "not manual"

# ──── 数据准备 ────
uv run scripts/compute_norm_stats.py --config-name <config>

# ──── JAX 训练 ────
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py <config> --exp-name=<name>

# ──── PyTorch 训练 ────
uv run scripts/train_pytorch.py <config> --exp_name <name>

# ──── PyTorch 多 GPU ────
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<n> \
    scripts/train_pytorch.py <config> --exp_name <name>

# ──── 推理服务 ────
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config> --policy.dir=<ckpt_dir> --port=8000

# ──── 恢复训练 ────
uv run scripts/train.py <config> --exp-name=<name> --resume
```

---

## 7. 本周产出

### 产出 1：新数据 config 设计说明

请基于第 2 节的内容，完成一份完整的设计文档，包含：
- 数据集描述（字段、维度、任务类型）
- 模型选择及理由
- 适配器代码（`MyRobotInputs` / `MyRobotOutputs`）
- 配置注册（完整的 `TrainConfig` 条目）
- 归一化统计量计算命令
- 训练和部署命令

### 产出 2：新 policy adapter 设计说明

请基于第 3 节的内容，完成一份完整的设计文档，包含：
- 机器人规格（自由度、摄像头、值域）
- 字段映射表（摄像头、状态、动作）
- 值域转换逻辑（如夹爪 [0,1] ↔ [-1,1]）
- 完整的适配器代码
- 验证命令和测试步骤

---

## 8. 8 周学习总结

```
┌────────────────────────────────────────────────────┐
│           OpenPI 知识体系全景                        │
│                                                    │
│  配置系统 (Week 2)                                  │
│    TrainConfig → DataConfig → AssetsConfig          │
│         ↓                                          │
│  数据管线 (Week 3)                                  │
│    Raw Data → Repack → RobotAdapt → Norm → Model   │
│         ↓                                          │
│  模型层 (Week 5)                                    │
│    π₀(Flow) / π₀-FAST(AR) / π₀.5(Flow+)           │
│    SigLIP + PaliGemma + ActionExpert                │
│         ↓                                          │
│  训练 (Week 4)                                     │
│    JAX: mesh + jit + orbax                         │
│    PyTorch: DDP + compile + safetensors (Week 7)   │
│         ↓                                          │
│  部署 (Week 1, 6)                                   │
│    Policy = Model + Transforms                     │
│    Server (WebSocket) ↔ Client (Runtime)           │
│         ↓                                          │
│  扩展 (Week 8)                                     │
│    新数据集 → 新 config + 新 adapter                 │
│    新机器人 → 新 policy + 新 transform               │
└────────────────────────────────────────────────────┘
```

---

## 9. 最终自检清单

- [ ] 能独立为一个新数据集编写完整的微调配置
- [ ] 能独立为一个新机器人编写输入/输出适配器
- [ ] 能解释从原始数据到训练到推理到部署的完整链路
- [ ] 能根据需求选择 π₀ / π₀-FAST / π₀.5
- [ ] 能根据显存选择 LoRA 微调还是全量微调
- [ ] 能启动推理服务器并用客户端连接
- [ ] 知道 JAX 和 PyTorch 训练的命令和差异
- [ ] 遇到问题时知道去哪个文件找答案
