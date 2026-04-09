# 第 6 周：服务化与客户端运行时

> **目标**：理解 WebSocket 推理服务器的实现、客户端如何与服务器通信、以及 Runtime 如何编排机器人控制循环。

---

## 1. 背景知识

### 1.1 为什么需要服务化？

训练好的模型通常在高性能 GPU 服务器上运行，而机器人的控制程序运行在机器人本体上（可能没有 GPU）。需要一种方式让两者通信：

```
机器人（边缘设备）          GPU 服务器
┌────────────────┐          ┌────────────────┐
│  摄像头 → 观测  │ ──网络──→ │  模型推理       │
│  控制器 ← 动作  │ ←──网络── │  返回动作       │
└────────────────┘          └────────────────┘
```

### 1.2 什么是 WebSocket？

WebSocket 是一种网络通信协议，与 HTTP 不同的是它保持**长连接**：

```
HTTP:  每次请求都要建立和断开连接（开销大）
  客户端 ──→ 服务器  ──→ 客户端  ✕断开
  客户端 ──→ 服务器  ──→ 客户端  ✕断开

WebSocket:  建立一次连接，持续双向通信（低延迟）
  客户端 ←──→ 服务器  ←──→ 客户端 ←──→ ...
  │            一直保持连接              │
```

对于机器人控制来说，每秒可能需要 50 次推理，WebSocket 的低延迟很重要。

### 1.3 什么是 msgpack？

msgpack 是一种高效的二进制序列化格式，比 JSON 更小更快。OpenPI 用它来传输 numpy 数组。

```python
import msgpack_numpy

# 序列化（Python 对象 → 二进制字节）
data = {"state": np.array([1.0, 2.0]), "image": np.zeros((224, 224, 3))}
binary = msgpack_numpy.packb(data)   # 高效的二进制格式

# 反序列化（二进制字节 → Python 对象）
data = msgpack_numpy.unpackb(binary)  # numpy 数组自动还原
```

---

## 2. 服务器端：WebsocketPolicyServer

### 2.1 整体架构

打开 `src/openpi/serving/websocket_policy_server.py`：

```python
# 约第 21-32 行
class WebsocketPolicyServer:
    def __init__(
        self,
        policy: BasePolicy,          # 推理策略（模型+变换）
        host: str = "0.0.0.0",       # 监听地址（0.0.0.0 = 所有网卡）
        port: int | None = None,     # 端口号
        metadata: dict | None = None, # 服务器元数据（发给客户端）
    ):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
```

### 2.2 启动服务器

```python
# 约第 34-46 行
def serve_forever(self) -> None:
    asyncio.run(self.run())      # 启动异步事件循环

async def run(self):
    async with websockets.serve(
        self._handler,             # 每个连接的处理函数
        self._host,
        self._port,
        compression=None,          # 不压缩（已经是二进制）
        max_size=None,             # 不限制消息大小
        process_request=_health_check,  # HTTP 健康检查
    ) as server:
        await server.serve_forever()
```

### 2.3 连接处理流程

```python
# 约第 48-83 行
async def _handler(self, websocket):
    logger.info(f"Connection from {websocket.remote_address} opened")
    packer = msgpack_numpy.Packer()

    # 步骤 1: 发送元数据（一次性）
    await websocket.send(packer.pack(self._metadata))
    # metadata 包含模型信息、reset_pose 等

    # 步骤 2: 请求-响应循环
    prev_total_time = None
    while True:
        try:
            start_time = time.monotonic()

            # 2a. 接收观测（客户端发来的 msgpack 数据）
            obs = msgpack_numpy.unpackb(await websocket.recv())

            # 2b. 调用 Policy 推理
            infer_time = time.monotonic()
            action = self._policy.infer(obs)
            infer_time = time.monotonic() - infer_time

            # 2c. 附加计时信息
            action["server_timing"] = {
                "infer_ms": infer_time * 1000,
            }
            if prev_total_time is not None:
                action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

            # 2d. 发送动作给客户端
            await websocket.send(packer.pack(action))
            prev_total_time = time.monotonic() - start_time

        except websockets.ConnectionClosed:
            logger.info("Connection closed")
            break
```

### 2.4 健康检查

```python
# 约第 86-90 行
def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(200, "OK\n")
    return None   # 非 /healthz 路径正常处理 WebSocket
```

**用途**：负载均衡器或 Kubernetes 可以通过 `GET /healthz` 检查服务是否存活。

```bash
# 测试健康检查
curl http://localhost:8000/healthz
# 返回: OK
```

---

## 3. 启动脚本：serve_policy.py

打开 `scripts/serve_policy.py`：

### 3.1 命令行参数

```python
# 约第 39-56 行
@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM     # 目标环境
    default_prompt: str | None = None     # 默认语言指令
    port: int = 8000                       # 服务端口
    record: bool = False                   # 是否记录推理行为
    policy: Checkpoint | Default = Default()  # 策略来源
```

### 3.2 默认检查点

```python
# 约第 59-76 行
DEFAULT_CHECKPOINT = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
}
```

### 3.3 使用方式

```bash
# 方式 1: 使用默认检查点（最简单）
uv run scripts/serve_policy.py --env aloha_sim

# 方式 2: 指定自定义检查点
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_aloha_sim \
    --policy.dir=./checkpoints/pi0_aloha_sim/my_exp/10000

# 方式 3: 指定端口和默认指令
uv run scripts/serve_policy.py --env droid \
    --port 9000 \
    --default-prompt "pick up the red cube"

# 方式 4: 记录推理行为（调试用）
uv run scripts/serve_policy.py --env aloha_sim --record
# 会保存每一步的输入输出到 policy_records/ 目录
```

---

## 4. 客户端：WebSocketClientPolicy

打开 `packages/openpi-client/src/openpi_client/websocket_client_policy.py`：

### 4.1 类定义

```python
class WebSocketClientPolicy(BasePolicy):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        api_key: str | None = None,    # 可选的 API 密钥
    ):
        self._url = f"ws://{host}:{port}"
        self._api_key = api_key
        self._ws = None
        self._metadata = None
```

### 4.2 连接和元数据

```python
    def _ensure_connected(self):
        """确保 WebSocket 连接已建立"""
        if self._ws is None:
            # 建立连接
            self._ws = websockets.sync.client.connect(self._url)

            # 接收服务器元数据（第一条消息）
            self._metadata = msgpack_numpy.unpackb(self._ws.recv())
            # metadata 可能包含 {"reset_pose": [0, -1.5, ...], ...}

    def get_server_metadata(self) -> dict:
        """获取服务器元数据"""
        self._ensure_connected()
        return self._metadata
```

### 4.3 推理调用

```python
    def infer(self, obs: dict) -> dict:
        """发送观测，接收动作"""
        self._ensure_connected()

        # 1. 序列化观测并发送
        self._ws.send(msgpack_numpy.packb(obs))

        # 2. 接收动作
        response = msgpack_numpy.unpackb(self._ws.recv())

        return response
        # response 包含:
        # {
        #   "actions": np.array([action_horizon, action_dim]),
        #   "state": np.array([state_dim]),
        #   "server_timing": {"infer_ms": 45.2, "prev_total_ms": 52.1},
        #   "policy_timing": {"infer_ms": 40.1},
        # }
```

### 4.4 使用示例

```python
from openpi_client import WebSocketClientPolicy

# 连接到推理服务器
policy = WebSocketClientPolicy(host="192.168.1.100", port=8000)

# 获取服务器信息
metadata = policy.get_server_metadata()
print(f"Reset pose: {metadata.get('reset_pose')}")

# 推理
observation = {
    "image": camera.read(),              # numpy 数组
    "state": robot.get_joint_positions(), # numpy 数组
    "prompt": "pick up the red cube",     # 字符串
}
result = policy.infer(observation)
actions = result["actions"]   # [action_horizon, action_dim]

# 执行第一步动作
robot.move_to(actions[0])
```

---

## 5. Runtime —— 控制循环编排

打开 `packages/openpi-client/src/openpi_client/runtime/runtime.py`：

### 5.1 核心抽象

```python
# Environment 接口（你需要实现的）
class Environment(ABC):
    def reset(self) -> dict:
        """重置环境到初始状态，返回第一个观测"""
        ...

    def get_observation(self) -> dict:
        """获取当前观测"""
        ...

    def apply_action(self, action: np.ndarray) -> None:
        """执行一个动作"""
        ...

    def is_episode_complete(self) -> bool:
        """当前 episode 是否结束"""
        ...

# Agent 接口
class Agent(ABC):
    def act(self, obs: dict) -> np.ndarray:
        """根据观测返回动作"""
        ...

# PolicyAgent 是最常用的 Agent 实现
class PolicyAgent(Agent):
    def __init__(self, policy: BasePolicy, action_horizon: int = 1):
        self._policy = policy

    def act(self, obs: dict) -> np.ndarray:
        result = self._policy.infer(obs)
        return result["actions"][0]  # 只执行第一步
```

### 5.2 Runtime 主循环

```python
class Runtime:
    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        subscribers: list[Subscriber] = [],  # 观察者（录像、日志等）
        max_hz: float = 50,                   # 最大控制频率
        num_episodes: int = 1,                # 运行几个 episode
    ):
        ...

    def run(self):
        for episode in range(self.num_episodes):
            # 1. 重置环境
            obs = self.environment.reset()

            # 通知观察者
            for sub in self.subscribers:
                sub.on_episode_start(episode)

            # 2. 控制循环
            while not self.environment.is_episode_complete():
                loop_start = time.monotonic()

                # 2a. 获取观测
                obs = self.environment.get_observation()

                # 2b. Agent 决策
                action = self.agent.act(obs)

                # 2c. 执行动作
                self.environment.apply_action(action)

                # 2d. 通知观察者
                for sub in self.subscribers:
                    sub.on_step(obs, action)

                # 2e. 限频（保证不超过 max_hz）
                elapsed = time.monotonic() - loop_start
                if elapsed < 1.0 / self.max_hz:
                    time.sleep(1.0 / self.max_hz - elapsed)

            # 3. 通知 episode 结束
            for sub in self.subscribers:
                sub.on_episode_end(episode)
```

### 5.3 Subscriber（观察者模式）

```python
class Subscriber(ABC):
    def on_episode_start(self, episode: int) -> None: ...
    def on_step(self, obs: dict, action: np.ndarray) -> None: ...
    def on_episode_end(self, episode: int) -> None: ...

# 内置实现
class VideoSaver(Subscriber):
    """录制 episode 视频"""
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def on_step(self, obs, action):
        # 保存每一帧图像

    def on_episode_end(self, episode):
        # 将帧合成视频文件
```

---

## 6. 完整部署调用链

```
┌─────────────────────────────────────────────────────┐
│                    GPU 服务器                         │
│                                                     │
│  serve_policy.py                                    │
│    ├── create_policy(args)                          │
│    │     ├── get_config("pi0_aloha_sim")           │
│    │     └── create_trained_policy(config, ckpt)   │
│    │           ├── 加载模型权重                      │
│    │           ├── 加载 norm_stats                  │
│    │           └── 组装 Policy(model, transforms)   │
│    │                                                │
│    └── WebsocketPolicyServer(policy, port=8000)    │
│          └── serve_forever()                        │
│                ├── 等待连接                          │
│                └── _handler(websocket):             │
│                      send(metadata)                 │
│                      loop:                          │
│                        obs = recv()                 │
│                        action = policy.infer(obs)   │
│                        send(action)                 │
└─────────────────┬───────────────────────────────────┘
                  │ WebSocket (ws://server:8000)
                  │ 数据格式: msgpack
┌─────────────────┴───────────────────────────────────┐
│                    机器人端                           │
│                                                     │
│  WebSocketClientPolicy(host="server", port=8000)   │
│    └── infer(obs) → action                          │
│                                                     │
│  PolicyAgent(policy=client_policy)                  │
│    └── act(obs) → action[0]                         │
│                                                     │
│  Runtime(                                           │
│    environment=MyRobotEnv(),                        │
│    agent=PolicyAgent(policy),                       │
│    subscribers=[VideoSaver("videos/")],             │
│    max_hz=50,                                       │
│  )                                                  │
│    └── run():                                       │
│          env.reset()                                │
│          loop:                                      │
│            obs = env.get_observation()              │
│            action = agent.act(obs)                  │
│            env.apply_action(action)                 │
│            sleep_to_maintain_hz()                   │
└─────────────────────────────────────────────────────┘
```

---

## 7. 运行命令指南

### 7.1 启动服务器

```bash
# 启动 Aloha 仿真推理服务器
uv run scripts/serve_policy.py --env aloha_sim --port 8000
```

### 7.2 运行客户端示例

```bash
# 查看 Aloha 仿真示例
cat examples/aloha_sim/main.py

# 运行示例（需要先启动服务器）
cd examples/aloha_sim
uv run python main.py
```

### 7.3 手动测试连接

```bash
# 测试健康检查
curl http://localhost:8000/healthz

# 用 Python 测试 WebSocket 连接
uv run python -c "
from openpi_client import WebSocketClientPolicy
policy = WebSocketClientPolicy(host='localhost', port=8000)
metadata = policy.get_server_metadata()
print(f'Server metadata: {metadata}')
"
```

### 7.4 运行客户端库测试

```bash
uv run pytest packages/openpi-client/ -v -k "not manual"
```

---

## 8. 本周产出

### 产出：线上部署调用链说明

请用自己的话重新描述第 6 节的完整调用链，包含以下要点：

1. **服务器启动流程**：从命令行参数到 Policy 创建到 WebSocket 监听
2. **通信协议**：连接建立 → 元数据 → 请求/响应循环 → 断开
3. **数据序列化**：msgpack 格式，自动处理 numpy 数组
4. **客户端使用**：WebSocketClientPolicy 的自动重连机制
5. **Runtime 控制循环**：Environment + Agent + Subscriber 的协作
6. **性能考虑**：max_hz 限频、server_timing 计时

---

## 9. 本周自检清单

- [ ] 理解 WebSocket 与 HTTP 的区别
- [ ] 能描述服务器的连接处理流程（metadata → loop → close）
- [ ] 理解 msgpack 序列化的作用
- [ ] 能说出 serve_policy.py 的命令行参数含义
- [ ] 理解 WebSocketClientPolicy 的 infer() 方法
- [ ] 能画出 Runtime 的控制循环流程
- [ ] 理解 Environment / Agent / Subscriber 三者的角色
- [ ] 能描述完整的"服务器 ↔ 客户端"部署链路
