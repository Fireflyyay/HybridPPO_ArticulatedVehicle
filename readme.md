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

## 当前模块

- `src/hybridppo_articulated_vehicle/common/`
	- 配置、状态类型、SMDP transition 数据结构
	- 对应原先的 `config.py` 与 `types.py`

- `src/hybridppo_articulated_vehicle/env/`
	- 中心铰接车运动学积分器
	- macro action wrapper
	- 几何成功判定与车体多边形计算

- `src/hybridppo_articulated_vehicle/primitives/`
	- 稳定语义 primitive 定义
	- 连续参数边界映射
	- controller-based parameterized primitive executor

- `src/hybridppo_articulated_vehicle/model/`
	- PPO 网络、分布、buffer、SMDP target 计算
	- `model/agent/` 下为双 actor 头 Hybrid PPO agent

- `src/hybridppo_articulated_vehicle/*.py`
	- 保留为兼容导出层
	- 旧导入路径仍可用，但实际实现已按分层结构组织

## 与 baseline 的关系

该仓库没有复制 `/home/cyberbus/Public/baseline/PPO_articulated_vehicle_another` 的完整训练工程，而是抽取并重构了其中最关键的研究抽象：

- 保留中心铰接车状态与动力学建模思路
- 保留 macro-action / primitive 的时间抽象思路
- 保留终端成功应基于几何复合条件这一原则
- 去掉 action mask、teacher、takeover、curriculum 和固定 primitive 查表耦合
- 改为双 actor 头 PPO + 参数化 primitive + SMDP update

因此，这个仓库更适合作为后续科研方法实现和论文方法章节的主干，而不是 baseline 的轻微改版。

## 安装

建议在已有的 `HOPE` conda 环境中直接安装：

```bash
pip install -e .[dev]
```

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
