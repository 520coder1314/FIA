# FIA 核心代码

该目录是可导入的 Python 包。请在其上一级目录运行 `python -m fia_core`，不要只把本目录内的 `.py` 文件打散上传到仓库根目录。

## 功能

- **Find**：计算干净类别中心距离，按分位数区间筛选候选类对；读取真实评估结果，按平均 ASR−FTR 选方向。
- **Inject**：固定代理参数及 BN 状态，统一预处理，构造软掩码和每分片共享触发器，优化分类、对齐和 TV 项。
- **Attack**：输出修改后的分片与触发器库，供下游微调及按固定顺序复用触发器。
- 外观评分模式必须显式指定：`stop_gradient` 或 `differentiable`。

## 类别数、类对数、方向数

CIFAR-10 只有 **10 个类别**。类对是从其中取两个不同类别组成的组合，不是新增类别。

| 数量 | 计算 | 含义 |
|---|---|---|
| 10 个类别 | 数据集定义 | airplane、deer 等 |
| 45 个无向类对 | 10×9÷2 | `{deer, airplane}` 与 `{airplane, deer}` 是同一对 |
| 90 个有向攻击方向 | 10×9 | deer→airplane 与 airplane→deer 是不同方向 |
| 本地筛选保留 26 个无向类对 | 对45个距离应用30%–90%分位数筛选 | 这是本次数据和代理下的实际筛选结果，不是固定类别数 |
| 对应 52 个候选方向 | 26×2 | 仅列出候选，尚未全部进行注入/微调/攻击评估 |

几何距离对称，但两个攻击方向的成功率不必相同。筛选数量也可能因数据、代理、边界并列值发生变化。
不能把“26个候选类对”写成“26个类别”，也不能把候选清单称为最佳攻击结果。

## 文件

- `core.py`：代理、分类头、几何筛选、掩码、损失、注入及触发器复用。
- `__main__.py`：`find / split / inject / select` 命令及运行记录。
- `resnet.py`：VICReg骨干，保留原始版权和 `LICENSE.vicreg`。
- `__init__.py`：版本号。

## 运行

依赖 Python >=3.10 和 PyTorch >=2.5。进入包含 `fia_core/` 的目录：

```bash
python -m fia_core --help
python -m fia_core find --help
python -m fia_core inject --help
```

完整步骤在本地 `NCFM-Lab/README_FIA_CORE.md`，发布包内则为根目录 `README.md`。

发布仓库应保留：

```text
README.md
requirements-core.txt
fia_core/
config/fia_core/cifar10_classes.json
tests_core/
```

仅发布代码、说明与测试，不包含模型权重、数据、历史实验结果或凭据。
新版代码用于重新实验；已有论文数字不自动成为新版实现的测量结果。
