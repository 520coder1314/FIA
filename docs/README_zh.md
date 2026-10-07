# FIA 使用说明

FIA 表示 **Find—Inject—Attack**。本仓库是精简发布版本，提供几何初筛、ResNet 全候选评分、Top-k 筛选、VLM 代理确认、固定代理注入和触发器复用。

## 从哪里开始

1. 在仓库根目录执行 `python -m pip install -e .`。
2. 执行 `OMP_NUM_THREADS=2 python -m unittest discover -s tests -v`，检查核心测试。
3. 执行 `python examples/smoke_test.py`，无需下载模型即可检查核心接口。
4. 根据 [资产说明](ASSETS.md) 准备真实数据和权重。
5. 按 [完整运行步骤](REPRODUCING.md) 执行 Find、数据划分、候选注入与实测选择。

## 类别不等于类对

CIFAR-10 有10个类别，组合成45个无向类对。此前本地运行保留26对，即52个有向候选。
鹿→飞机与飞机→鹿是不同攻击方向。此数量只是几何筛选输出，不代表已经运行52项攻击实验。
“最佳类对”指小模型 Top-k 候选中，在 VLM 代理上平均 ASR−FTR 最高的方向。

## 两级筛选

几何筛选参数为 q_l、q_u，实验配置为0.30、0.90。全部合格方向都需要候选注入和 ResNet 训练评分。取 Top-k（k≥2，实验配置3），再用 Janus-Pro-7B 代理微调评估全部入围方向。两级都只使用训练选择集，最终测试集保持独立。

`python -m fia.score_resnet` 提供 ResNet-34 候选评分；`fia select --stage small --top-k 3` 导出 shortlist.json；`fia select --stage vlm --shortlist ...` 输出 selected.json。完整命令见 [两级 Find 操作步骤](CASCADED_FIND.md)。Janus 的 SWIFT 训练沿用外部运行程序。

作者确认论文实验使用上述两级流程。ResNet-34 训练3轮，采用42、3407、2026三个种子；两级使用一致的数据、触发器应用、选择集和评分协议。模型专属的优化器和输入处理分别记录。

## 复现范围

当前代码修正了预处理与代理BN状态，不能直接继承旧实现的论文结果。模型权重、蒸馏数据、VLM训练/推理程序不在本精简仓库内。
新实验需要先固定选择集和测试集，候选效果只在选择集比较，最终再评估保留的测试集。
外观项需显式选择 `stop_gradient` 或 `differentiable`，分别记录并报告。

## 框架图

首页框架图（[assets/framework.png](../assets/framework.png)）为作者修订版：攻击者筛选更易受攻击的类对（如 {deer, airplane}），向蒸馏数据注入后门触发器，云端 VLM 经过多轮下游微调后，对带触发器的输入返回目标类别（"Airplane."）。旧版框架图保留在 `assets/framework_archived.png`。
