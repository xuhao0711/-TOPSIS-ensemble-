# 木质纤维素三目标 TOPSIS 与模型集成

新增流程对 Hemicellulose、Cellulose、Lignin 分别排序与集成，不把三个目标不同量纲的误差直接混合。测试集在任何训练前按实验来源分组隔离；候选模型、网格、权重、增强设置必须预先确定。

## 运行

```bash
pip install -r requirements.txt
python -m unittest discover -p 'test_*.py'
python run_topsis.py
# 明确启用真实数据；默认只运行 DT/EN 各一个参数组合
python run_topsis.py --real --data raw.xlsx --group-col source_id --out results_real
# 大网格只在用户确认计算预算后启用 --full-grid；本次未运行
```

在仓库根目录运行。原始 Excel 默认 Sheet1，包含 Notebook 原有 17 个 FEATURES 和三个目标列。`--sheet` 指定 sheet。`source_id` 应代表文献来源/实验批次等独立单元，而非自动生成的逐行编号；可用 `--group-col` 指定实际分组列。保留全局唯一 row_id；缺失整列时在全表切分前生成。存在 ancestry 时连通分量与来源分组合并。拒绝增强行、缺失/无穷数据、跨训练/测试的来源、祖先 ID 或完全相同特征。此检查不能证明不同特征行之间不存在隐性同源性，需核实来源元数据。

## TOPSIS 决策

| 指标 | 方向 | 默认权重 | 理由 |
|---|---|---:|---|
| 训练外层 CV MAE | 越低越好 | 0.50 | 延续原代码以 MAE 为主要任务损失 |
| 训练外层 CV RMSE | 越低越好 | 0.25 | 补充大误差惩罚 |
| 训练外层 CV R² | 越高越好 | 0.25 | 补充相对目标方差的解释能力 |

默认权重是预先声明的工程选择，未经真实数据证明为最优。MAE/RMSE 有相关性，并非独立证据；真实研究应事先声明权重，并对等权、MAE 主导方案做**训练集**敏感性分析。不得观察测试指标后调整。

先求各列 L2 范数，r_ij=x_ij/sqrt(sum_i x_ij²)，再乘归一化权重。成本指标取列最小值为正理想解，R² 取最大值；负 R² 保留符号。计算正/负理想解欧氏距离 D+/D−，贴近度 C=D−/(D++D−)。全相同或单候选时 C=0.5；零范数列贡献零；非有限指标报错。按贴近度降序排名，并列按模型名稳定排列。

集成使用**所有预先指定候选**的训练 CV 贴近度归一化权重 w_i=C_i/sum(C)。不调 top-k、不拟合测试权重、不宣称集成必然更好。保存的 FrozenEnsemble.predict 实际执行加权预测。

## 公平比较与数据边界

每个候选使用相同外层 GroupKFold 划分。每个外层训练折内再用 GroupKFold+GridSearchCV 以 MAE 调参；只预测未参与该网格搜索的外层验证折。TOPSIS 使用各外层折指标的算术平均，CV_MAE 基线也按相同外层平均 MAE 选模型。这个基线保持原来的 MAE 选择规则，但改用嵌套评分以降低调参乐观偏差，因此不是历史 best_score_ 数值的直接复现。

排序、集成权重和全部三个目标的模型先冻结，再对同一个未增强测试集统一评价。各模型最终在全部训练集上用分组网格搜索重拟合。OOF 候选预测可审计；不导出自适应选择后集成的 OOF 指标作为无偏性能，因为权重使用过整份 OOF 结果。独立测试对比才用于评估最终选择策略。本次仅模拟数据验证功能。

## 保留原流程与增强

shuju.ipynb 的 Optuna、SMOTER、Jitter、ResidualAug、模型空间及 GroupKFold 函数保留，清除历史输出，关闭旧自动训练入口。原 Optuna 实际是一个综合标量目标（多目标误差加权），不是 Pareto 多目标 study；固定分出的“测试”数据被多次用于 trial 配置选择，**旧 test_original 不再是独立测试证据**。新增脚本不读取旧 train_augmented_best，不沿用旧 best_cfg。

新增默认模拟增强仅用于功能测试。真实默认不增强；`--augment jitter` 是轻量通用扰动，不代替原工艺边界。

需要沿用原增强函数时，安装 `optuna xgboost`，通过 `--augmentation-config preset.json` 提供预先指定的 JSON，例如：

```json
{"rare_ratio": 0.8, "quantiles": [0.05, 0.95], "n_jitter": 20,
 "n_residual": 20, "beta_range": [0.1, 0.6],
 "base_xgb": {"n_estimators": 20, "max_depth": 3}}
```

LegacyAugment 在每次 GridSearchCV 拟合子集内调用原 SMOTER→Jitter→ResidualAug，从不增强外层验证/测试数据。此适配器一次处理一个目标，尾部定义相较原三目标 OR 规则有变化。原 ResidualAug 内部 KFold 的残差校准仍是旧算法；它仅接触当前拟合折，但其内部同源增强残差相关性仍需真实验证。该可选适配器本次仅静态检查，未运行依赖 XGBoost 的真实增强。不要把旧 test 优化出的配置作为 preset；若重新启用 Optuna，应在每个外层训练折内同时嵌套调增强配置，并按来源划分内层验证，不能复用旧 main。

## 输出与验证范围

每个目标目录输出 ranking.csv（排名、理想距离、贴近度、集成权重）、fold_metrics.csv、oof_predictions.csv、test_comparison.csv、test_predictions.csv、results.xlsx、audit.json、TOPSIS_ensemble.pkl、FINAL_CV_MAE.pkl、FINAL_TOPSIS.pkl，以及 topsis_ranking.png/model_comparison.png。根目录 split_ids.json 保存固定切分。模拟图标明 SIMULATED，不提交模拟性能作为科研结果。生成目录已忽略。

真实数据仍需验证：来源分组合理性、上游 Optuna 与预处理是否污染旧测试集、折数与 R² 稳定性、增强边界与多目标一致性、候选多样性及误差相关性、预先定义的权重敏感性、不同分组种子下稳定性，以及冻结策略在全新外部数据上的表现。不能据模拟结果判断哪种策略真实更优。
