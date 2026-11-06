# AlloGate

[English](README.md) · **简体中文**

[![测试](https://github.com/ChptrX/AlloGate/actions/workflows/test.yml/badge.svg)](https://github.com/ChptrX/AlloGate/actions/workflows/test.yml)
[![软件包基本功能测试](https://github.com/ChptrX/AlloGate/actions/workflows/package.yml/badge.svg)](https://github.com/ChptrX/AlloGate/actions/workflows/package.yml)

从同一个构象出发，蛋白质会先到达开放态还是闭合态？哪些残基和接触会影响这一几率？AlloGate 用分子动力学轨迹研究这类问题。它先从轨迹中学习能够描述蛋白质缓慢构象变化的集体变量（CV），再在模型中减弱某个残基或接触提供的信息，观察对指定构象转变的预测如何变化。

以开放态和闭合态之间的转变为例。先给定一个动力学参照，用来估计蛋白质先到达闭合态的概率。随后固定训练好的模型和参照，减弱模型中某一接触的贡献，再与干预前的概率比较。概率升高还是降低、变化幅度有多大，可以帮助研究者挑选值得进一步做实验的接触。

**从哪里开始**　本项目目前处于 alpha 阶段，提供模型、训练器和两个使用合成数据的完整示例。可以先运行[慢集体变量训练示例](examples/synthetic_training.py)或[门控扰动评估示例](examples/synthetic_gate_audit.py)。研究自己的蛋白质时，需要将轨迹整理为[按时间滞后配对的构象图](docs/data-contract.md)，并为所研究的转变拟合[动力学参照](docs/kinetic-reference.md)。

## 如何表示不同蛋白质的结构

模型显式指定各层结构之间的对应关系，各层单元的数量可以随蛋白质而变。

```text
残基
 ↓
局部结构单元 ── 接触分组 ──→ 界面
      └─────────┬──────────┘
                ↓
          全局结构单元
                ↓
     两条分支的集体变量（Direct / Route）
```

- **局部结构单元**由一小段残基组成，分组方式可配置。每组残基都属于同一条链、同一个全局结构单元。
- **界面**汇集局部结构单元之间的接触信息，每个界面恰好连接两个全局结构单元。未激活界面的输出严格为零。
- **全局结构单元**由研究者划定，是互不重叠的残基集合，例如结构域、亚结构域、蛋白链或功能区域。

模型从每批输入推断结构单元的数量。残基、局部单元、界面、全局单元、模拟副本、轨迹、GPU 和集体变量的数量均可按研究需要选择。

## 安装

```text
python -m pip install -e ".[training,test]"
python -m pytest
```

单独使用配置、清单、注册表和结果存储模块时，PyTorch 为可选依赖。训练、图神经网络和门控评估所需的依赖可分别通过 `.[training]`、`.[gnn]` 或 `.[evaluation]` 安装。

## 配置与训练

模型、训练目标和优化器的参数都由实验配置指定。可以从 `examples/configs/protein_training.yaml` 入手，调整结构层级、通道维度、滞后时间、损失权重、优化器、批大小、停止条件和数据类型。配置中出现未定义的字段时，程序会报错。

训练接口接受按时间滞后配对的图，每个图的大小可以不同。可用 MDAnalysis、MDTraj、已有的预处理流程或自定义适配器读取轨迹，再把图数据交给训练器。

```python
from allogate.config import load_experiment_config
from allogate.models import AlloGateGeometryModel
from allogate.training import SlowCVTrainer

config = load_experiment_config("experiment.yaml")
model = AlloGateGeometryModel.from_experiment_config(config)
trainer = SlowCVTrainer(model, config, device="cuda")
result = trainer.fit(train_loader, validation_loader, output_directory="run")
```

`train_loader` 和 `validation_loader` 每次返回一个 `LaggedBatch` 对象。使用 PyTorch 的 `DataLoader` 时，将 `collate_lagged_batches` 设为组批函数。完整的输入格式和避免数据泄漏的划分方法见[训练指南](docs/training.md)和[数据接口说明](docs/data-contract.md)。

用下面的命令运行可复现的合成数据训练示例。

```text
python examples/synthetic_training.py --output synthetic-training-run
```

训练结束后，程序会保存最佳模型检查点和逐轮训练记录。

## 门控扰动评估

门控决定模型在预测时使用多少来自某一处的信息。三类门控分别对应三种操作。

- **状态门（State Gate）**调整模型在某个位点保存的特征。
- **传递门（Relay Gate）**调节某个结构单元向下一层传递的信息量。
- **接触门（Contact Gate）**调节一对残基之间某个接触的贡献。

如何改变门控强度并测量响应，见[评估指南](docs/evaluation-contract.md)。可以直接运行示例。

```text
python examples/synthetic_gate_audit.py
```

关闭一个门时，相邻项仍沿用原先的权重。因此，响应反映的是模型的预测在多大程度上依赖这部分信息。实际突变或配体结合能否产生同样的效应，需要另行验证。

## 方法依据

几何编码器遵循几何向量感知机（GVP）对标量和向量特征的变换约束。用图神经网络学习集体变量时无需手工指定描述符，慢模态学习采用 DeepTICA 风格的目标函数。这两部分参考了 Zhang 等人发表的方法（*J. Chem. Theory Comput.* **20**, 10787–10797, 2024）。[原文 DOI](https://doi.org/10.1021/acs.jctc.4c01197)
