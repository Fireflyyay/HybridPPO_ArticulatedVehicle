# Hybrid PPO for Center-Articulated Vehicles

该仓库实现一版面向中心铰接车辆的研究型算法骨架，核心方法为：

- SMDP-based Hybrid PPO
- dual actor heads
- parameterized motion primitives
- center-articulated vehicle kinematics

当前版本聚焦于核心算法抽象，而不是完整训练流水线。仓库内包含：

- 铰接车状态与运动学模型
- 语义稳定的参数化 primitive 库与执行器
- SMDP macro-transition 记账
- 双 actor 头 PPO agent
- 成功判定与几何评价工具
- 面向核心逻辑的最小单元测试
- baseline 风格的 Debug / Warmup / 搅拌站风格场景生成
- TensorBoard 驱动的最小训练闭环

## 代码怎么用

这个项目的主要入口是 `train.py`。训练时，代码会自动完成下面的流程：

1. 构建 `ExperimentConfig`。
2. 初始化中心铰接车辆环境、参数化 primitive 执行器和 Hybrid PPO agent。
3. 按 episode 收集 macro rollout。
4. 周期性做 PPO 更新、评估和 checkpoint 保存。
5. 把训练过程写入 TensorBoard 日志目录。

如果你只想修改车辆参数，请优先改 [src/common/config.py](src/common/config.py) 里的 `VehicleConfig`。这个配置会被环境、success checker 和 primitive 执行器共同读取，因此改这里会影响整条链路。

如果你想改训练行为，请看 [src/common/runtime_config.py](src/common/runtime_config.py) 和 [src/training/cli.py](src/training/cli.py)。前者定义默认实验配置，后者负责把命令行参数覆盖到配置对象上。

如果你想改 primitive 生成或执行逻辑，请看 [src/primitives/library.py](src/primitives/library.py) 和 [src/primitives/__init__.py](src/primitives/__init__.py)。

## 当前模块

- `src/common/`
	- 配置、状态类型、SMDP transition 数据结构

- `src/env/`
	- 中心铰接车运动学积分器
	- macro action wrapper
	- 几何成功判定与车体多边形计算

- `src/primitives/`
	- 稳定语义 primitive 定义
	- 连续参数边界映射
	- controller-based parameterized primitive executor

- `src/model/`
	- PPO 网络、分布、buffer、SMDP target 计算
	- `model/agent/` 下为双 actor 头 Hybrid PPO agent

- `src/training/`
	- rollout driver
	- checkpoint / evaluation / TensorBoard logger
	- 训练主循环与 CLI

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Shapely
- TensorBoard

推荐在已有的 `HOPE` conda 环境中运行。


## 安装

建议在已有的 `HOPE` conda 环境中直接安装：

```bash
pip install -e .[dev]
```

如果你还没有创建环境，可以先安装依赖，再使用 `PYTHONPATH=src` 运行脚本；项目的测试配置已经把 `src` 加入了 Python path。

## 测试

```bash
pytest
```

当前最小测试覆盖：

- SMDP reward/GAE 计算
- 参数化 primitive rollout
- dual-head PPO action/update smoke test
- macro wrapper 的折扣回报累计
- 成功判定几何检查
- 任务环境 reset/step smoke test
- rollout driver smoke test
- checkpoint round-trip smoke test

## 训练示例

最小训练命令如下：

```bash
PYTHONPATH=src conda run -n HOPE python train.py --episodes 200 --episodes-per-update 8
```

## 生成 Proxy Safety Sidecar

如果你要启用 soft mask，第一步不是直接改训练脚本，而是先离线生成 proxy safety sidecar。sidecar 会为当前 semantic primitive 库预计算：

- articulation angle bins
- 每个 semantic action 对应的 proxy parameter grid
- 基于前后车体 swept volume 的 ray required-clearance

仓库已经提供了离线生成脚本 [example/build_proxy_safety_sidecar.py](example/build_proxy_safety_sidecar.py)。最小用法如下：

```bash
PYTHONPATH=src conda run -n HOPE python example/build_proxy_safety_sidecar.py \
	--output data/proxy_safety_sidecar.npz
```

一个更接近实际训练配置的例子：

```bash
PYTHONPATH=src conda run -n HOPE python example/build_proxy_safety_sidecar.py \
	--output data/proxy_safety_sidecar.npz \
	--lidar-beams 108 \
	--lidar-range 30 \
	--articulation-bins 7 \
	--proxy-resolution 2 \
	--max-macro-steps 32
```

常用参数：

- `--output`：sidecar 输出路径，训练时会通过这个路径加载
- `--lidar-beams`：必须和训练时观测里的 lidar beam 数一致
- `--lidar-range`：必须和训练时观测里的 lidar 最大距离一致
- `--articulation-bins`：离线索引使用的 articulation 离散 bin 数
- `--proxy-resolution`：每个语义动作内部代理参数网格的离散分辨率
- `--max-proxies-per-action`：如果不为空，会对每个语义动作保留固定数量的代理点，降低 sidecar 体积
- `--max-macro-steps`：离线构建时每个 proxy rollout 的最大 macro horizon

要点：

- sidecar 必须和当前 primitive library、车辆参数、lidar 配置保持一致，否则加载后虽然能运行，但安全分数会失真。
- 如果你修改了 [src/common/config.py](src/common/config.py) 里的 `VehicleConfig`、`PrimitiveExecutorConfig` 或 primitive 参数边界，应该重新生成 sidecar。
- sidecar 只作为安全代理索引使用，不会扩展真实动作空间，也不会改写真实 SMDP transition。

常用参数：

- `--episodes`：总训练回合数
- `--episodes-per-update`：每收集多少个 episode 做一次 PPO 更新
- `--max-macro-steps`：每个 episode 允许的最大 macro action 数
- `--max-low-level-steps`：低层控制步数上限
- `--eval-interval`：评估间隔
- `--eval-episodes`：每个 level 的评估回合数
- `--warmup-episodes`：Warmup 阶段持续回合数
- `--debug-phase-episodes`：Debug 阶段持续回合数
- `--train-level`：debug 后默认训练场景 level
- `--seed`：随机种子
- `--device`：`auto`、`cpu` 或 `cuda`
- `--log-root`：TensorBoard 日志根目录
- `--run-name`：运行名称
- `--save-interval`：checkpoint 保存间隔
- `--lidar-beams`：观测里激光束数量
- `--proxy-safety-sidecar`：离线 sidecar 文件路径
- `--enable-soft-mask`：开启语义 soft mask
- `--soft-mask-gamma`：控制 soft mask 对低安全分数的压制强度
- `--soft-mask-logit-scale`：控制 semantic score 映射到 logits bias 的强度，越小越宽松
- `--soft-mask-floor`：soft mask 的概率下限，越大越不容易把动作压得过低
- `--safety-loss-coef`：连续参数 auxiliary safety loss 的权重
- `--resume`：从已有 checkpoint 恢复

训练输出默认包括：

- TensorBoard 日志目录
- latest checkpoint
- periodic checkpoint
- best checkpoint

## 开启 Soft Mask

生成 sidecar 之后，可以在训练命令里显式开启 soft mask：

```bash
PYTHONPATH=src conda run -n HOPE python train.py \
	--episodes 200 \
	--episodes-per-update 8 \
	--lidar-beams 108 \
	--proxy-safety-sidecar data/proxy_safety_sidecar.npz \
	--enable-soft-mask \
	--soft-mask-gamma 1.0 \
	--soft-mask-logit-scale 1.0 \
	--soft-mask-floor 0.2 \
	--safety-loss-coef 0.05
```

当前版本的开启条件很简单：

- `--proxy-safety-sidecar` 指向一个有效的 sidecar 文件
- `--enable-soft-mask` 打开 agent 的 soft mask 分支

其中：

- `--enable-soft-mask` 会把 [src/common/runtime_config.py](src/common/runtime_config.py) 里的 `ExperimentConfig.agent.soft_mask_enabled` 设为 `True`
- `--proxy-safety-sidecar` 会写入 [src/common/runtime_config.py](src/common/runtime_config.py) 里的 `ExperimentConfig.proxy_safety.sidecar_path`
- `--soft-mask-gamma`、`--soft-mask-logit-scale`、`--soft-mask-floor` 和 `--safety-loss-coef` 会覆盖 agent 超参数

如果你不想走 CLI，也可以直接在 [src/common/runtime_config.py](src/common/runtime_config.py) 修改默认值：

- 把 `HybridPPOHyperConfig.soft_mask_enabled` 改成 `True`
- 在 `ProxySafetyConfig.sidecar_path` 里填 sidecar 路径

当前 soft mask 的作用范围是：

- 它会在高层离散语义动作采样前，对 semantic logits 加一个分布感知的 soft bias
- 它会额外引入连续参数的 auxiliary safety loss
- 它不会把 proxy primitive 当作真实动作执行，也不会篡改真实执行得到的 tau、reward、next state 或 value target

当前默认值已经刻意设得偏宽松，适合先在相对狭窄的场景里避免一次性压掉大量动作：

- `soft_mask_gamma = 1.0`
- `soft_mask_logit_scale = 1.0`
- `soft_mask_floor = 0.2`

如果你发现 early training 里语义动作还是收缩得太快，优先继续减小 `--soft-mask-logit-scale`，或者增大 `--soft-mask-floor`，而不是先把 mask 直接关掉。

## 运行逻辑

训练主循环在 [src/training/train_loop.py](src/training/train_loop.py) 中实现，核心顺序是：

1. 用场景工厂生成环境。
2. 用参数化 primitive 执行器把高层 macro action 展开为低层控制。
3. 用 Hybrid PPO agent 收集 rollout 并更新策略。
4. 用 [src/training/evaluator.py](src/training/evaluator.py) 在 Debug / Warmup / Normal 场景上做评估。
5. 按 checkpoint 配置保存模型。

环境中的车辆几何由 [src/common/config.py](src/common/config.py) 的 `VehicleConfig` 控制，成功判定由 [src/env/success.py](src/env/success.py) 计算。

## 代码结构速览

- [src/common/config.py](src/common/config.py)：默认车辆参数、primitive 执行参数、训练超参数
- [src/common/runtime_config.py](src/common/runtime_config.py)：实验默认配置和命令行覆盖入口
- [src/env/dynamics.py](src/env/dynamics.py)：中心铰接车辆运动学积分
- [src/env/task_env.py](src/env/task_env.py)：环境 step/reset、奖励、观测、终止逻辑
- [src/env/success.py](src/env/success.py)：成功判定与车体几何
- [src/primitives/library.py](src/primitives/library.py)：参数化 primitive 定义与执行
- [src/model/](src/model/)：PPO 网络、分布、buffer、SMDP target
- [src/training/](src/training/)：训练循环、评估、日志、checkpoint

## 生成与调试建议

- 先用较小的 `--episodes` 和 `--eval-interval` 跑通流程，再扩大训练规模。
- 如果你只改 vehicle 参数，优先跑 `tests/test_success.py` 和 `tests/test_primitives.py`。
- 如果你改了场景生成或环境逻辑，再跑 `pytest` 全量测试。
- `TensorBoard` 可以直接打开日志目录观察 reward、success rate、loss 和 checkpoint 指标。

## 备注

仓库当前的默认车辆参数已经和 `PPO_articulated_vehicle` 对齐，确保两边在几何尺寸和低层控制范围上使用同一套基础值。

## 训练

建议在 `HOPE` conda 环境中运行：

```bash
PYTHONPATH=src conda run -n HOPE python train.py --episodes 200 --episodes-per-update 8
```
