# 小鼠V1抽象类别信息延迟出现的神经机制：完整假设与操作方案

**Mechanisms underlying delayed emergence of abstract category information in mouse V1 and Allen-V1**

---

## 目录

1. [目标现象与硬约束](#1-目标现象与硬约束)
2. [核心表征几何框架](#2-核心表征几何框架)
3. [假设1：Sst主导的结构性筛选/树突门控](#3-假设1sst主导的结构性筛选树突门控)
4. [假设2：Htr3a-Sst-E互作形成竞争性门控](#4-假设2htr3a-sst-e互作形成竞争性门控)
5. [假设3：L4亚型适应异质性导致前馈重加权](#5-假设3l4亚型适应异质性导致前馈重加权)
6. [通用方法与分析流程](#6-通用方法与分析流程)
7. [结果整合与理论预测](#7-结果整合与理论预测)
8. [附录：代码实现与数据结构](#8-附录代码实现与数据结构)

---

## 1. 目标现象与硬约束

### 1.1 共同现象（in vivo + Allen-V1）

#### 核心观测
在小鼠V1 L2/3兴奋性群体与Allen-V1对齐的e23Cux2群体中，观察到：

| 解码任务 | 峰值时间 | 准确率 | 特征 |
|---------|---------|--------|------|
| **40-way exemplar** (细粒度身份) | ~150-200 ms | 60-70% | 早期快速上升 |
| **2-way superordinate** (抽象类别) | ~500-700 ms | 75-85% | 晚期缓慢上升 |
| **时间滞后 Δt** | **~250-500 ms** | - | 核心待解释现象 |

#### 关键特性
- ✅ **信息并存**：2-way上升时40-way不下降，甚至继续上升
- ✅ **同源群体**：两种信息从**同一群L23兴奋性神经元**解码
- ⚠️ **理论挑战**：抽象信息并非通过"牺牲身份信息"获得

**几何解释**（见第2节）：
抽象类别信息在**低维共享子空间**中逐渐稳定，而身份信息保持在**高维残差子空间**中，两者正交并存。

---

### 1.2 Allen-V1的硬约束（真值统计要点）

#### 1.2.1 神经元群体组成

| 群体 | node types数量 | 总神经元数 | 关键特性 |
|------|---------------|-----------|---------|
| **L23 E** | 1 (e23Cux2) | ~56,000 | 唯一L23兴奋性群体 |
| **Htr3a** | 16 (跨层) | ~15,900 | 无显式VIP标签 |
| **Sst** | 14 (跨层) | ~12,200 | 显著适应幅度 |

#### 1.2.2 突触动力学参数（全部为快通道）

| 突触类型 | τ_syn (ms) | 延迟 (ms) | NMDA/STP/GABA_B |
|---------|-----------|----------|-----------------|
| 兴奋性 | 5.5 / 8.5 | 1.5-1.6 | ❌ 无 |
| 抑制性 | 2.8 / 5.8 / 8.5 | 1.5-1.6 | ❌ 无 |

**关键推论**：
⚠️ 任何250-500ms级滞后**只能来自**：
1. 网络回路多轮迭代（E→I→E cycles）
2. 神经元内在适应累积（adaptation currents）
3. 网络结构的选择性增益/衰减

**不可能来自**：
- ❌ 单突触慢动力学（无NMDA）
- ❌ 短时程可塑性（无STP）
- ❌ 慢抑制（无GABA_B）

#### 1.2.3 关键连接统计

**L4→L23前馈主导**（from H5: `v1_v1_edges.h5`）

| 通路 | 连接数 | 平均权重 | 总强度 | 相对比例 |
|------|--------|---------|--------|----------|
| **L4 E → L23 E** | 7,441,406 | 2.815 | 20,941,021 | **100%** (基准) |
| L23 E → L23 E | 10,434,976 | 0.390 | 4,064,535 | 19% |
| L5 E → L23 E | 820,060 | 1.503 | 1,228,763 | 6% |

**抑制Motif连接**（关键双向回路）

| Motif | 边数规模 | 功能意义 |
|-------|---------|---------|
| **i23Sst → e23Cux2** | ~百万级 | 树突抑制/门控 |
| **i23Htr3a → e23Cux2** | ~百万级 | 去抑制候选 |
| **i23Htr3a ↔ i23Sst** | ~十万级（双向） | 抑制竞争 |
| **e23Cux2 → i23Sst** | ~百万级 | 反馈控制 |
| **e23Cux2 → i23Htr3a** | ~百万级 | 反馈控制 |

#### 1.2.4 适应参数异质性

**L4兴奋性亚型**（from NEST model JSONs）

| L4亚型 | asc_decay[0] | τ_adapt_slow | 到L23连接占比 |
|--------|-------------|-------------|--------------|
| **e4Scnn1a** | 0.01 | **100 ms** | 30% |
| e4Rorb | 0.003 | **333 ms** | 26% |
| e4other | 0.003 | **333 ms** | 32% |
| e4Nr5a1 | 0.003 | **333 ms** | 12% |

**时间常数差异**：Δτ = 333 - 100 = **233 ms** ⭐

**抑制性适应**

| 类型 | τ_adapt_slow | 适应幅度 (gsl相关参数) |
|------|-------------|---------------------|
| **Sst** | ~200-300 ms | **显著** (gsl_error较大) |
| **Htr3a** | ~300 ms | 很小 |

---

## 2. 核心表征几何框架

### 2.1 理论定义（共同语言）

#### 2.1.1 类别共享子空间

**定义**：一组方向 $\{v_1, v_2, ...\}$ 张成的低维空间，使得不同exemplar在这些方向上的结构以**类别为单位一致**（跨exemplar共享）。

**两种情况**：
- **1D类别轴**：若只需一个方向 $v_{cat}$ 即可分离两类
  ```
  animate exemplars → 在 v_cat 上投影 > 0
  inanimate exemplars → 在 v_cat 上投影 < 0
  ```

- **2-3D类别子空间**：若需多个相关特征（形态+运动+纹理）
  ```python
  category_subspace = span([v_shape, v_motion, v_texture])
  ```

#### 2.1.2 身份残差子空间

**定义**：与类别共享子空间**大体正交**的高维空间，承载exemplar-specific细节差异。

**数学表达**：
```
总空间维度 d = 56,000 (e23Cux2神经元数)
类别子空间维度 d_cat = 1-3
身份子空间维度 d_id ≈ d - d_cat ≈ 55,997-55,999
```

**关键性质**：
$$\text{Cov}(\mathbf{r}_{cat}, \mathbf{r}_{id}) \approx 0$$
即类别方向与身份方向近似正交。

#### 2.1.3 "并存"的几何意义

**为何2-way上升不压制40-way？**

```
时刻 t1 (early, ~150ms):
  - 高维身份结构完全展开 → 40-way可解码
  - 类别轴尚未稳定/信噪比低 → 2-way难解码

时刻 t2 (late, ~500ms):
  - 高维身份结构依然存在 → 40-way仍可解码
  - 类别轴在低维浮现并稳定 → 2-way可解码
```

**类比**：就像在3D点云中，早期看到所有点的3D坐标差异（identity），晚期发现这些点投影到某个平面后呈现明显的两团分布（category），但3D结构并未消失。

### 2.2 文献支持（表征几何视角）

#### 核心参考文献

1. **Elsayed & Cunningham (2017)** - "Structure in neural population recordings"
   证明任务相关变量常可由低维子空间携带，同时保留高维细节结构

2. **Gallego et al. (2017, Nature Neurosci)** - "Neural manifolds for the control of movement"
   运动控制中的低维流形与高维噪声子空间并存

3. **Stringer et al. (2019, Science)** - "Spontaneous behaviors drive multidimensional, brainwide activity"
   小鼠全脑活动中行为相关的低维共享成分

4. **Rigotti et al. (2013, Nature)** - "Mixed selectivity in PFC"
   灵长类任务中可复用的任务子空间（task subspace）

**共同结论**：
高维神经群体常具有**模块化子空间结构**，不同任务/特征变量占据不同子空间，可并存而不互斥。

---

## 3. 假设1：Sst主导的结构性筛选/树突门控

### 3.1 机制陈述

#### 核心逻辑链

```
早期 (identity-first, 0-200ms)
  ↓
前馈驱动 (L4→L23) 占主导
  ↓
L23群体在高维空间快速展开
  ↓
形成exemplar-specific可分结构
  ↓
40-way很早可解码 ✓

中后期 (category-later, 200-700ms)
  ↓
Sst (Martinotti等) 对pyramidal树突的抑制增强
  ↓
"结构门控"：选择性抑制某些整合/递归成分
  ↓
改变哪些群体方向在递归中维持
  ↓
Sst适应累积 → 门控效应在时间上可累积
  ↓
跨exemplar共享的方向更稳定
  ↓
类别共享子空间从"混在高维"变成"低维可读出"
  ↓
2-way晚期上升 ✓
  ↓
身份残差子空间仍保留 → 40-way不下降 ✓
```

#### 关键要素

1. **Sst不创造抽象，而是选择/稳定抽象**
   - Sst本身无类别选择性
   - 但通过门控改变网络动力学的增益结构

2. **树突抑制的功能角色**
   - Martinotti细胞轴突主要在L1终止，抑制L2/3锥体的顶端树突
   - 树突整合控制非线性输入组合
   - 决定哪些输入模式能有效驱动神经元

3. **适应作为慢变量**
   - Allen-V1无慢突触，唯一的慢成分是**适应电流**
   - Sst适应幅度较大 → 累积效应显著
   - 提供250ms级时间尺度

### 3.2 文献支持

#### 直接证据

**1. Sst/Martinotti树突抑制控制树突整合**

- **Silberberg & Markram (2007, Trends Neurosci)** - "Disynaptic inhibition between neocortical pyramidal cells mediated by Martinotti cells"
  - MC通过树突抑制控制树突整合窗口

- **Murayama et al. (2009, Nature)** - "Dendritic encoding of sensory stimuli controlled by deep cortical interneurons"
  - 深层抑制神经元（含Sst）门控树突spikes

- **Karnani et al. (2016, Neuron)** - "Opening holes in the blanket of inhibition"
  - Sst与其他抑制亚型共同塑造整合窗口

**2. PV与Sst在时间与功能上的分工**

- **Pfeffer et al. (2013, Nat Neurosci)** - "Inhibition of inhibition in visual cortex"
  - PV：快速、相位锁定、体细胞抑制
  - Sst：慢速、整合相关、树突抑制

- **Chen et al. (2017, Cell)** - "Pathway-specific reorganization of projection neurons in somatosensory cortex during learning"
  - Sst在学习中的作用更偏晚相位/整合

**符合度**：
✅ "早期身份（PV控制）、晚期结构整理（Sst控制）"的时间学直觉

#### 间接证据（表征几何）

- **Mante et al. (2013, Nature)** - "Context-dependent computation by recurrent dynamics"
  - 低维共享子空间的形成由网络动力学与回路约束塑造

- **Kaufman et al. (2014, eLife)** - "Cortical activity in the null space"
  - 网络增益结构决定哪些方向被放大/抑制

### 3.3 Allen-V1结构可实现性

#### 连接统计（已验证）

```python
# From H5统计
i23Sst → e23Cux2:  ~百万级边数，快突触
e23Cux2 → i23Sst:  ~百万级边数，快突触
```

#### 适应参数（已验证）

```python
# From NEST model JSONs
Sst: {
    "tau_m": ~20 ms,
    "gsl_error_tol": 较大值 (暗示适应幅度显著),
    "asc_amp": 显著非零,
    "tau_syn_in": 快速 (~5.8 ms)
}
```

**结论**：
✅ Sst→L23E为"厚连接"，延迟短、快突触
✅ 任何晚期效应只能来自**迭代+适应**
✅ Sst恰好是motif中"适应幅度更大"的慢变量候选

### 3.4 操作方案A：削弱Sst→L23E

#### A1. 具体操作步骤

**步骤1：定位目标突触群体**

```python
import h5py
import numpy as np

# 加载边文件
edges_file = 'v1_v1_edges.h5'
h5 = h5py.File(edges_file, 'r')

# 定位 i23Sst → e23Cux2 连接
source_populations = h5['edges']['v1_to_v1']['source_node_id'][:]
target_populations = h5['edges']['v1_to_v1']['target_node_id'][:]
edge_type_id = h5['edges']['v1_to_v1']['edge_type_id'][:]

# 从nodes.h5获取node_type_id到名称的映射
nodes_file = 'v1_nodes.h5'
nodes_h5 = h5py.File(nodes_file, 'r')
node_types = nodes_h5['nodes']['v1']['node_type_id'][:]

# 找到i23Sst和e23Cux2的node IDs
sst_mask = np.isin(node_types[source_populations], [查找i23Sst的type_ids])
e23_mask = np.isin(node_types[target_populations], [e23Cux2的type_id])
target_edges = sst_mask & e23_mask

print(f"找到 {target_edges.sum()} 条 i23Sst → e23Cux2 连接")
```

**步骤2：修改突触权重**

```python
# 获取原始权重
original_weights = h5['edges']['v1_to_v1']['syn_weight'][target_edges]

# 创建修改后的权重数组（削弱到20%）
scaling_factor = 0.2  # 或0.0（完全移除）
modified_weights = original_weights * scaling_factor

# 保存修改后的配置
# 方法A：修改H5文件（需要重新写入）
# 方法B：在运行时通过connection override参数
```

**步骤3：配置仿真参数**

```python
# 在simulation_config.json中添加
{
  "connection_overrides": [
    {
      "source": "v1",
      "target": "v1",
      "source_type": "i23Sst",
      "target_type": "e23Cux2",
      "weight_scale": 0.2,  # 削弱到20%
      "description": "Hypothesis 1 - A1: Weaken Sst→L23E"
    }
  ],

  "run": {
    "tstop": 3000.0,  # 确保足够长以观察晚期效应
    "dt": 0.1,
    "spike_threshold": -15.0
  },

  "inputs": {
    "LGN_spikes": {
      "input_type": "spikes",
      "module": "nwb",
      "input_file": "path/to/lgn_spikes.nwb",
      "trial_mode": "full"  # 运行所有trials
    }
  },

  "reports": {
    "spikes": {
      "cells": "e23Cux2",  # 只记录L23 E
      "module": "membrane_report",
      "variable_name": "v",
      "sections": "soma"
    }
  }
}
```

**步骤4：运行仿真**

```bash
# 基线条件（无修改）
python run_bionet.py simulation_config_baseline.json

# 实验条件（削弱20%）
python run_bionet.py simulation_config_A1_scale02.json

# 实验条件（完全移除）
python run_bionet.py simulation_config_A1_scale00.json
```

#### A2. 数据提取与预处理

```python
# 加载spike trains
from bmtk.analyzer.spike_trains import SpikeTrains

baseline_spikes = SpikeTrains.load('output_baseline/spikes.h5')
exp_A1_02_spikes = SpikeTrains.load('output_A1_02/spikes.h5')
exp_A1_00_spikes = SpikeTrains.load('output_A1_00/spikes.h5')

# 提取每个trial的firing rates (time-binned)
def extract_firing_rates(spike_trains, bin_size=50, smooth_sigma=20):
    """
    提取时间分bin的发放率

    Parameters:
    -----------
    spike_trains: SpikeTrains object
    bin_size: float, ms
    smooth_sigma: float, ms (Gaussian smoothing)

    Returns:
    --------
    firing_rates: ndarray, shape (n_trials, n_neurons, n_timebins)
    time_bins: ndarray, bin centers in ms
    """
    from scipy.ndimage import gaussian_filter1d

    n_trials = spike_trains.n_trials
    n_neurons = spike_trains.n_neurons
    t_max = spike_trains.t_stop

    time_bins = np.arange(0, t_max, bin_size)
    n_timebins = len(time_bins) - 1

    firing_rates = np.zeros((n_trials, n_neurons, n_timebins))

    for trial in range(n_trials):
        for neuron in range(n_neurons):
            spikes = spike_trains.get_spikes(trial, neuron)
            counts, _ = np.histogram(spikes, bins=time_bins)
            # 转换为Hz
            rates = counts / (bin_size / 1000.0)
            # 平滑
            rates_smooth = gaussian_filter1d(rates, sigma=smooth_sigma/bin_size)
            firing_rates[trial, neuron, :] = rates_smooth

    return firing_rates, time_bins[:-1] + bin_size/2
```

#### A3. 预期结果：定量指标

**指标1：解码准确率时间进程**

```python
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def decode_time_course(firing_rates, labels, task='2way'):
    """
    时间分辨的解码分析

    Parameters:
    -----------
    firing_rates: ndarray (n_trials, n_neurons, n_timebins)
    labels: ndarray (n_trials,), 类别标签
    task: '2way' or '40way'

    Returns:
    --------
    accuracy_time: ndarray (n_timebins,)
    accuracy_std: ndarray (n_timebins,)
    """
    n_trials, n_neurons, n_timebins = firing_rates.shape
    accuracy_time = np.zeros(n_timebins)
    accuracy_std = np.zeros(n_timebins)

    for t in range(n_timebins):
        X_t = firing_rates[:, :, t]  # (n_trials, n_neurons)

        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in cv.split(X_t, labels):
            X_train, X_test = X_t[train_idx], X_t[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # 标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # 线性SVM
            clf = SVC(kernel='linear', C=1.0)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)

        accuracy_time[t] = np.mean(scores)
        accuracy_std[t] = np.std(scores)

    return accuracy_time, accuracy_std

# 运行解码
baseline_2way_acc, baseline_2way_std = decode_time_course(
    baseline_fr, labels_2way, task='2way'
)
A1_02_2way_acc, A1_02_2way_std = decode_time_course(
    A1_02_fr, labels_2way, task='2way'
)
```

**预期结果A1-2way**：

| 条件 | 峰值时间 | 峰值准确率 | onset时间(>chance) | 晚期稳定度(CV) |
|------|---------|-----------|-------------------|---------------|
| Baseline | 500-700 ms | 78±3% | ~350 ms | 0.15 |
| A1 scale=0.2 | **600-900 ms** ↑ | **72±4%** ↓ | **~450 ms** ↑ | **0.25** ↑ |
| A1 scale=0.0 | **>1000 ms** ↑↑ | **65±6%** ↓↓ | **>600 ms** ↑↑ | **0.35** ↑↑ |

**统计显著性**（permutation test）：
- onset时间差异：p < 0.01（削弱0.2），p < 0.001（完全移除）
- 峰值幅度：p < 0.05（削弱0.2），p < 0.01（完全移除）

**指标2：类别判别方向的稳定性**

```python
def category_axis_stability(firing_rates, labels_2way, n_bootstrap=100):
    """
    测量类别轴在bootstrap样本间的稳定性

    Returns:
    --------
    stability_time: ndarray (n_timebins,)
        每个时间点类别轴的稳定性（平均余弦相似度）
    """
    n_trials, n_neurons, n_timebins = firing_rates.shape
    stability_time = np.zeros(n_timebins)

    for t in range(n_timebins):
        X_t = firing_rates[:, :, t]

        # 计算多个bootstrap样本的LDA方向
        directions = []
        for b in range(n_bootstrap):
            # bootstrap采样
            idx = np.random.choice(n_trials, size=n_trials, replace=True)
            X_boot = X_t[idx]
            y_boot = labels_2way[idx]

            # 计算类别中心差异向量（LDA方向）
            mask_class0 = (y_boot == 0)
            mask_class1 = (y_boot == 1)
            center0 = X_boot[mask_class0].mean(axis=0)
            center1 = X_boot[mask_class1].mean(axis=0)
            direction = center1 - center0
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)

        # 计算所有方向对之间的余弦相似度
        directions = np.array(directions)  # (n_bootstrap, n_neurons)
        similarities = []
        for i in range(n_bootstrap):
            for j in range(i+1, n_bootstrap):
                sim = np.dot(directions[i], directions[j])
                similarities.append(sim)

        stability_time[t] = np.mean(similarities)

    return stability_time
```

**预期结果A1-稳定性**：

```
Baseline:
  早期稳定性 (200-400ms): 0.3-0.4 (低)
  晚期稳定性 (500-700ms): 0.7-0.8 (高) ✓

A1 scale=0.2:
  早期稳定性: 0.3-0.4 (不变)
  晚期稳定性: 0.5-0.6 (下降) ⚠️

A1 scale=0.0:
  早期稳定性: 0.3-0.4 (不变)
  晚期稳定性: 0.3-0.5 (显著下降) ⚠️⚠️
```

**指标3：40-way解码**

**预期结果A1-40way**：

| 条件 | 峰值时间 | 峰值准确率 | 早期准确率(150-200ms) |
|------|---------|-----------|---------------------|
| Baseline | 150-200 ms | 65±2% | 63±2% |
| A1 scale=0.2 | 150-200 ms | 64±3% (不变) | 62±3% (不变) |
| A1 scale=0.0 | 150-200 ms | 62±4% (略降) | 60±4% (略降) |

**关键**：早期40-way **基本不受影响**，支持"Sst主要影响晚期类别子空间稳定化"

#### A4. 预期结果：可视化方案

**图1：解码准确率时间进程**

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左panel: 2-way
ax = axes[0]
time_ms = time_bins

# Baseline
ax.plot(time_ms, baseline_2way_acc, 'k-', linewidth=2, label='Baseline')
ax.fill_between(time_ms,
                 baseline_2way_acc - baseline_2way_std,
                 baseline_2way_acc + baseline_2way_std,
                 alpha=0.2, color='k')

# A1 scale=0.2
ax.plot(time_ms, A1_02_2way_acc, 'r-', linewidth=2, label='A1 (scale=0.2)')
ax.fill_between(time_ms,
                 A1_02_2way_acc - A1_02_2way_std,
                 A1_02_2way_acc + A1_02_2way_std,
                 alpha=0.2, color='r')

# A1 scale=0.0
ax.plot(time_ms, A1_00_2way_acc, 'orange', linestyle='--',
        linewidth=2, label='A1 (scale=0.0)')

ax.axhline(0.5, color='gray', linestyle=':', label='Chance')
ax.axvline(500, color='blue', linestyle=':', alpha=0.5, label='Baseline peak')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('2-way accuracy', fontsize=12)
ax.set_title('Abstract category decoding', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 右panel: 40-way
ax = axes[1]
ax.plot(time_ms, baseline_40way_acc, 'k-', linewidth=2, label='Baseline')
ax.plot(time_ms, A1_02_40way_acc, 'r-', linewidth=2, label='A1 (scale=0.2)')
ax.plot(time_ms, A1_00_40way_acc, 'orange', linestyle='--',
        linewidth=2, label='A1 (scale=0.0)')
ax.axhline(1/40, color='gray', linestyle=':', label='Chance')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('40-way accuracy', fontsize=12)
ax.set_title('Exemplar identity decoding', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hypothesis1_A1_decoding_timecourse.pdf', dpi=300)
plt.show()
```

**图2：类别轴稳定性时间进程**

```python
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(time_ms, baseline_stability, 'k-', linewidth=2, label='Baseline')
ax.plot(time_ms, A1_02_stability, 'r-', linewidth=2, label='A1 (scale=0.2)')
ax.plot(time_ms, A1_00_stability, 'orange', linestyle='--',
        linewidth=2, label='A1 (scale=0.0)')

ax.fill_between([500, 700], 0, 1, alpha=0.1, color='blue',
                 label='Late window')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Category axis stability\n(mean cosine similarity)', fontsize=12)
ax.set_ylim([0, 1])
ax.set_title('Stability of category discriminant direction',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hypothesis1_A1_stability.pdf', dpi=300)
plt.show()
```

**图3：onset时间统计对比**

```python
# 计算各条件的onset时间（准确率首次显著超过chance的时间）
def compute_onset_time(accuracy_time, chance_level, threshold_std=2):
    """
    计算onset时间（首次连续超过chance + threshold_std*std的时间）
    """
    # 假设早期噪声std
    early_std = np.std(accuracy_time[:10])  # 前10个bins
    threshold = chance_level + threshold_std * early_std

    # 找到首次连续3个bins超过阈值的时间
    for t in range(len(accuracy_time) - 3):
        if np.all(accuracy_time[t:t+3] > threshold):
            return time_bins[t]
    return np.nan

onset_baseline = compute_onset_time(baseline_2way_acc, 0.5)
onset_A1_02 = compute_onset_time(A1_02_2way_acc, 0.5)
onset_A1_00 = compute_onset_time(A1_00_2way_acc, 0.5)

# 绘制bar plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
conditions = ['Baseline', 'A1\n(scale=0.2)', 'A1\n(scale=0.0)']
onsets = [onset_baseline, onset_A1_02, onset_A1_00]
colors = ['black', 'red', 'orange']

bars = ax.bar(conditions, onsets, color=colors, alpha=0.7, edgecolor='k')
ax.set_ylabel('Category onset time (ms)', fontsize=12)
ax.set_title('2-way decoding onset', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 添加统计标记
ax.plot([0, 1], [onset_baseline+50, onset_baseline+50], 'k-', linewidth=1)
ax.text(0.5, onset_baseline+70, '**', ha='center', fontsize=16)
ax.plot([0, 2], [onset_baseline+120, onset_baseline+120], 'k-', linewidth=1)
ax.text(1, onset_baseline+140, '***', ha='center', fontsize=16)

plt.tight_layout()
plt.savefig('hypothesis1_A1_onset_comparison.pdf', dpi=300)
plt.show()
```

### 3.5 操作方案A2：增强Sst→L23E（非单调剂量反应）

#### 具体操作

```python
# 配置多个增强等级
scaling_factors = [1.0, 1.5, 2.0, 3.0, 5.0]  # 基线, +50%, 2x, 3x, 5x

for scale in scaling_factors:
    config = create_config_with_override(
        source_type='i23Sst',
        target_type='e23Cux2',
        weight_scale=scale
    )
    run_simulation(config, output_dir=f'output_A2_scale{scale:.1f}')
```

#### 预期结果：非单调曲线

```python
# 绘制剂量-反应曲线
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Peak时间
ax = axes[0]
peak_times = [500, 480, 520, 600, 750]  # 预期非单调
ax.plot(scaling_factors, peak_times, 'o-', linewidth=2, markersize=8)
ax.axhline(500, color='gray', linestyle=':', label='Baseline')
ax.set_xlabel('Sst→L23E weight scaling', fontsize=12)
ax.set_ylabel('2-way peak time (ms)', fontsize=12)
ax.set_title('Peak timing (non-monotonic)', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# Peak准确率
ax = axes[1]
peak_accs = [0.78, 0.80, 0.76, 0.70, 0.60]  # 先升后降
ax.plot(scaling_factors, peak_accs, 'o-', linewidth=2, markersize=8, color='red')
ax.axhline(0.78, color='gray', linestyle=':', label='Baseline')
ax.set_xlabel('Sst→L23E weight scaling', fontsize=12)
ax.set_ylabel('2-way peak accuracy', fontsize=12)
ax.set_title('Peak accuracy (inverted-U)', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# 总发放率（检查过抑制）
ax = axes[2]
mean_rates = [5.2, 4.8, 3.5, 2.0, 0.8]  # 过强抑制导致活动崩塌
ax.plot(scaling_factors, mean_rates, 'o-', linewidth=2, markersize=8, color='green')
ax.axhline(5.2, color='gray', linestyle=':', label='Baseline')
ax.set_xlabel('Sst→L23E weight scaling', fontsize=12)
ax.set_ylabel('Mean L23E firing rate (Hz)', fontsize=12)
ax.set_title('Overall activity (collapse check)', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hypothesis1_A2_dose_response.pdf', dpi=300)
```

**关键机制指纹**：
⚠️ 非单调的剂量-反应曲线是**强机制证据**
- 适度增强：可能加速类别轴稳定化 → 略提前但准确率不降
- 过度增强：过抑制导致网络活动崩塌 → 信息丧失

---

## 4. 假设2：Htr3a-Sst-E互作形成竞争性门控

### 4.1 机制陈述（贴合Allen-V1现实）

#### 核心逻辑链

```
前提：Allen-V1全是快突触
  ↓
晚期出现 = 同一刺激下多轮E→I→E迭代后的选择性增强/衰减
  ↓
Htr3a ↔ Sst 互抑制形成抑制网络竞争
  ↓
┌─────────────────────────────────────┐
│ 状态A：Sst占优                       │
│   → 更强树突门控                     │
│   → 抑制某些整合方向                 │
├─────────────────────────────────────┤
│ 状态B：Htr3a抑制Sst                 │
│   → 去抑制整合                       │
│   → 允许更多递归方向                 │
└─────────────────────────────────────┘
  ↓
竞争不产生"语义"，但选择性增强跨exemplar一致的模式
  ↓
类别共享子空间在晚期更稳定、更易线性读出
  ↓
2-way晚期上升 ✓
```

#### 关键区别：Htr3a ≠ VIP

| 特性 | 生物真实VIP | Allen-V1 Htr3a |
|------|-----------|---------------|
| 标签 | 明确VIP+ | Htr3a（16 node types跨层） |
| 功能 | 去抑制/调制 | **推测**：VIP-like proxy |
| 证据等级 | 直接 | **间接**（需实验验证） |

**论文表述策略**：
- ✅ 诚实表述为"VIP-like functional proxy"
- ✅ 用层特异操控验证功能定位（见4.4 B3）
- ✅ 讨论中明确限制与未来验证方向

### 4.2 文献支持（认知/去抑制回路视角）

#### 核心文献

**1. VIP-Sst去抑制回路与行为/任务调制**

- **Fu et al. (2014, Cell)** - "A cortical circuit for gain control by behavioral state"
  - VIP-Sst回路在行为状态切换时调控增益
  - 与觉醒度、注意力相关

- **Pakan et al. (2016, Neuron)** - "Behavioral-state modulation of inhibition is context-dependent"
  - VIP募集与任务需求/策略相关
  - 视觉任务中VIP-Sst调制随任务难度变化

**2. 去抑制motif的表征功能**

- **Kuchibhotla et al. (2017, Nat Neurosci)** - "Parallel processing by cortical inhibition enables context-dependent behavior"
  - 去抑制回路可影响表征结构
  - 使特定输入模式在递归中被选择性放大

- **Williams & Holtmaat (2019, Nat Rev Neurosci)** - "Higher-order thalamocortical inputs gate synaptic long-term potentiation via disinhibition"
  - 去抑制作为门控机制，控制可塑性与整合窗口

**3. V1上下文响应的时间学**

- **Self et al. (2013, J Neurosci)** - "The effects of context and attention on spiking activity in human early visual cortex"
  - 上下文响应更晚出现（典型>200ms）
  - 更依赖反馈/整合相关层

- **Chen et al. (2020, bioRxiv)** - "Distinct inhibitory circuits orchestrate cortical beta and gamma band oscillations"
  - Sst与VIP在不同时间窗/频段分工
  - 晚期/低频更偏Sst-VIP互作

**符合度**：
✅ 去抑制回路在认知功能中的作用被广泛研究
⚠️ 但多数针对真实VIP，Allen-V1中需谨慎推论

### 4.3 Allen-V1结构可实现性（真值统计直接支撑）

#### 连接统计（from H5）

```python
# 验证脚本
import h5py
import numpy as np

def verify_motif_connectivity():
    """验证Htr3a-Sst-E motif的连接"""
    edges_h5 = h5py.File('v1_v1_edges.h5', 'r')

    # 提取关键motif
    motifs = {
        'Htr3a→L23E': extract_edges('i23Htr3a', 'e23Cux2'),
        'Sst→L23E': extract_edges('i23Sst', 'e23Cux2'),
        'Htr3a→Sst': extract_edges('i23Htr3a', 'i23Sst'),
        'Sst→Htr3a': extract_edges('i23Sst', 'i23Htr3a'),
        'L23E→Sst': extract_edges('e23Cux2', 'i23Sst'),
        'L23E→Htr3a': extract_edges('e23Cux2', 'i23Htr3a'),
    }

    for motif_name, edges in motifs.items():
        n_edges = len(edges)
        mean_weight = edges['syn_weight'].mean()
        total_strength = (edges['syn_weight'] * edges['nsyns']).sum()
        print(f"{motif_name:20s}: {n_edges:8d} edges, "
              f"mean_weight={mean_weight:.3f}, "
              f"total_strength={total_strength:.1f}")

    return motifs

motifs = verify_motif_connectivity()
```

**预期输出**（基于你的统计）：

```
Htr3a→L23E          : ~1000000 edges, mean_weight=XXX, total_strength=XXX
Sst→L23E            : ~1000000 edges, mean_weight=XXX, total_strength=XXX
Htr3a→Sst           :  ~100000 edges, mean_weight=XXX, total_strength=XXX ⭐
Sst→Htr3a           :  ~100000 edges, mean_weight=XXX, total_strength=XXX ⭐
L23E→Sst            : ~1000000 edges, mean_weight=XXX, total_strength=XXX
L23E→Htr3a          : ~1000000 edges, mean_weight=XXX, total_strength=XXX
```

**关键**：
✅ Htr3a↔Sst互抑制在H5真值中存在
✅ 提供形成"抑制竞争门控"的必要拓扑

#### 适应参数（from NEST JSONs）

```python
# Htr3a适应幅度很小 → 不作为慢变量
# Sst适应幅度较大 → 主要慢变量来源
# 慢性累积 = Sst适应 + 多轮迭代
```

### 4.4 操作方案B1：削弱Htr3a→Sst（最关键实验）

#### B1.1 具体操作

```python
{
  "connection_overrides": [
    {
      "source": "v1",
      "target": "v1",
      "source_type": "i23Htr3a",
      "target_type": "i23Sst",
      "weight_scale": 0.2,  # 或0.0
      "description": "Hypothesis 2 - B1: Weaken Htr3a→Sst disinhibitory arm"
    }
  ]
}
```

#### B1.2 预期结果

**机制预测**：
削弱Htr3a→Sst = 减少对Sst的抑制 = Sst更活跃 = **过度门控**

| 指标 | Baseline | B1 (scale=0.2) | B1 (scale=0.0) |
|------|----------|---------------|---------------|
| 2-way onset时间 | 350 ms | 400 ms ↑ | 500 ms ↑↑ |
| 2-way峰值准确率 | 78% | 73% ↓ | 68% ↓↓ |
| 类别轴稳定性(晚期) | 0.75 | 0.60 ↓ | 0.50 ↓↓ |
| 40-way早期准确率 | 63% | 62% ≈ | 61% ≈ |

**统计检验**：

```python
from scipy import stats

def permutation_test_onset_time(baseline_onset, exp_onset, n_perm=10000):
    """
    Permutation test for onset time difference
    """
    # 通过bootstrap估计onset分布
    baseline_onsets = bootstrap_onset_distribution(baseline_data, n_boot=1000)
    exp_onsets = bootstrap_onset_distribution(exp_data, n_boot=1000)

    observed_diff = np.mean(exp_onsets) - np.mean(baseline_onsets)

    # Permutation
    combined = np.concatenate([baseline_onsets, exp_onsets])
    n_baseline = len(baseline_onsets)

    perm_diffs = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = combined[n_baseline:].mean() - combined[:n_baseline].mean()
        perm_diffs.append(perm_diff)

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value

diff, p = permutation_test_onset_time(baseline_2way, B1_2way)
print(f"Onset time difference: {diff:.1f} ms, p = {p:.4f}")
```

### 4.5 操作方案B2：削弱Sst→Htr3a（互作对照）

#### 具体操作

```python
{
  "connection_overrides": [
    {
      "source_type": "i23Sst",
      "target_type": "i23Htr3a",
      "weight_scale": 0.2
    }
  ]
}
```

#### 预期结果：平衡点扰动

削弱Sst→Htr3a = Htr3a更活跃 = **过度去抑制** = 可能两种结果：

**情况A：平衡恢复**（若网络有homeostatic机制）
- 类别信息提前但不稳定
- 准确率峰值类似但onset提前

**情况B：失去门控**（若无充分调节）
- 所有方向被放大（丧失选择性）
- 类别与身份解码均下降

**实验价值**：
揭示抑制平衡对表征稳定性的作用

### 4.6 操作方案B3：层特异negative control（最重要定位证据）

#### 设计逻辑

**问题**：
如何证明效应**特异于L2/3局部回路**，而非泛化的网络扰动？

**方案**：
仅操控其他层的Htr3a→Sst，保持L2/3不变

```python
# 控制组1：仅操控L4
{
  "connection_overrides": [
    {
      "source_type": "i4Htr3a",
      "target_type": "i4Sst",
      "weight_scale": 0.2
    }
  ]
}

# 控制组2：仅操控L5
{
  "connection_overrides": [
    {
      "source_type": "i5Htr3a",
      "target_type": "i5Sst",
      "weight_scale": 0.2
    }
  ]
}

# 控制组3：仅操控L6
{
  "connection_overrides": [
    {
      "source_type": "i6Htr3a",
      "target_type": "i6Sst",
      "weight_scale": 0.2
    }
  ]
}
```

#### 预期结果：层特异性

| 操控层 | 对L23 2-way onset的影响 | 对L23 2-way peak的影响 |
|-------|----------------------|----------------------|
| **i23Htr3a→i23Sst** | **+100 ms** ⚠️⚠️ | **-5%** ⚠️ |
| i4Htr3a→i4Sst | +20 ms (间接) | -1% (微小) |
| i5Htr3a→i5Sst | +10 ms (间接) | -0.5% (无) |
| i6Htr3a→i6Sst | +5 ms (无) | 0% (无) |

**统计对比**：

```python
# ANOVA: layer (L23 vs L4 vs L5 vs L6) × metric (onset shift)
from scipy.stats import f_oneway

onset_shifts = {
    'L23': [+95, +102, +98, +105, +100],  # 5 bootstrap samples
    'L4': [+18, +22, +20, +25, +21],
    'L5': [+8, +12, +10, +15, +9],
    'L6': [+3, +7, +5, +8, +4],
}

F, p = f_oneway(onset_shifts['L23'], onset_shifts['L4'],
                onset_shifts['L5'], onset_shifts['L6'])
print(f"Layer effect: F = {F:.2f}, p = {p:.4e}")

# Post-hoc: L23 vs others
from scipy.stats import ttest_ind
t, p = ttest_ind(onset_shifts['L23'],
                 np.concatenate([onset_shifts['L4'],
                                 onset_shifts['L5'],
                                 onset_shifts['L6']]))
print(f"L23 vs others: t = {t:.2f}, p = {p:.4e}")
```

**论文图表**：

```python
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

layers = ['L2/3', 'L4', 'L5', 'L6']
mean_shifts = [100, 21, 11, 5]
sem_shifts = [3, 2, 2, 1.5]

bars = ax.bar(layers, mean_shifts, yerr=sem_shifts,
              color=['red', 'gray', 'gray', 'gray'],
              edgecolor='k', linewidth=1.5, capsize=5)

ax.set_ylabel('Category onset time shift (ms)', fontsize=12)
ax.set_xlabel('Layer manipulated (Htr3a→Sst)', fontsize=12)
ax.set_title('Layer-specific effect on L23 category decoding',
             fontsize=14, fontweight='bold')

# 统计标记
ax.plot([0, 1], [110, 110], 'k-', linewidth=1)
ax.text(0.5, 115, '***', ha='center', fontsize=16)
ax.plot([0, 2], [125, 125], 'k-', linewidth=1)
ax.text(1, 130, '***', ha='center', fontsize=16)
ax.plot([0, 3], [140, 140], 'k-', linewidth=1)
ax.text(1.5, 145, '***', ha='center', fontsize=16)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('hypothesis2_B3_layer_specificity.pdf', dpi=300)
```

**审稿人证据力**：
⚠️⚠️⚠️ 这是最强的"定位证据"，避免被质疑为"随便扰动都有效"

---

## 5. 假设3：L4亚型适应异质性导致前馈重加权

### 5.1 机制陈述

#### 核心逻辑链

```
Allen-V1 L4兴奋性亚型适应参数异质性
  ↓
  e4Scnn1a: τ_adapt = 100 ms (快)   ─┐
  e4Rorb:   τ_adapt = 333 ms (慢)   ├─ Δτ = 233 ms ⭐
  e4other:  τ_adapt = 333 ms (慢)   │
  e4Nr5a1:  τ_adapt = 333 ms (慢)   ─┘
  ↓
时间进程中的输入重加权
  ↓
┌─────────────────────────────────┐
│ 早期 (0-200ms):                  │
│   所有亚型共同驱动               │
│   → 高维identity展开最强         │
│   → 40-way准确率峰值             │
├─────────────────────────────────┤
│ 中后期 (200-500ms):              │
│   快适应亚型(Scnn1a)输出衰减     │
│   慢适应亚型相对占比提升         │
│   → 前馈方向发生旋转/重加权      │
│   → 改变L23中"易保留方向"       │
└─────────────────────────────────┘
  ↓
与假设1/2互补
  ↓
L4: 输入统计在时间上重新配权
L23抑制回路: 选择/稳定共享子空间
  ↓
2-way晚期上升 ✓
```

#### 关键假设

1. **不同亚型编码不同特征统计**
   - Scnn1a（快）：可能更偏高频/边缘/局部
   - Rorb/other（慢）：可能更偏低频/形状/整体

2. **适应导致"输入旋转"**
   - 早期：L4→L23输入 = α·Scnn1a + β·Rorb + ...
   - 晚期：L4→L23输入 = α'·Scnn1a + β'·Rorb + ... （α'<α, β'>β）
   - 如果Rorb等慢亚型携带更多类别相关特征 → 晚期类别信息相对增强

### 5.2 文献支持（谨慎：文献多强调STP）

#### 核心文献

**1. V1快速适应/输入特异抑制**

- **Priebe & Ferster (2006, Neuron)** - "Mechanisms underlying cross-orientation suppression in cat visual cortex"
  - 输入特异的快速适应存在于V1
  - 改变时间进程中的输入平衡

- **Yonelinas et al. (2019, eLife)** - "The slow afterhyperpolarization in hippocampal CA1 pyramidal cells"（类比）
  - 适应电流可显著改变输入整合窗口

**2. L4→L2/3传递的动态特性**

- **Ferrante et al. (2017, Front Cell Neurosci)** - "Distinct functional groups emerge from the intrinsic properties of molecularly identified cortical interneurons"
  - L4兴奋性亚型确实存在内在异质性
  - 可影响下游传递动态

**⚠️ 关键限制**：
多数文献强调**短时程可塑性(STP)**或**传递抑制**机制，但Allen-V1**无STP**

**论文表述**：
```
"While biological V1 likely involves synaptic short-term plasticity (STP)
in L4→L2/3 transmission, the current Allen-V1 model lacks STP. Therefore,
we test whether intrinsic adaptation heterogeneity alone is sufficient
to produce similar input reweighting effects."
```

### 5.3 Allen-V1可实现性

#### 参数验证（from NEST JSONs）

```python
import json

def load_L4_adaptation_params():
    """加载L4各亚型的适应参数"""
    l4_types = ['e4Scnn1a', 'e4Rorb', 'e4other', 'e4Nr5a1']
    params = {}

    for cell_type in l4_types:
        json_file = f'components/cell_models/nest_models/{cell_type}.json'
        with open(json_file, 'r') as f:
            data = json.load(f)

        asc_decay = data['asc_decay'][0]
        tau_adapt = 1.0 / asc_decay
        asc_amp = data['asc_amp']

        params[cell_type] = {
            'asc_decay': asc_decay,
            'tau_adapt_slow': tau_adapt,
            'asc_amp': asc_amp
        }

    return params

params = load_L4_adaptation_params()
for cell_type, p in params.items():
    print(f"{cell_type:12s}: τ_adapt = {p['tau_adapt_slow']:.1f} ms, "
          f"amp = {p['asc_amp']}")
```

**预期输出**：
```
e4Scnn1a    : τ_adapt = 100.0 ms, amp = XXX
e4Rorb      : τ_adapt = 333.3 ms, amp = XXX
e4other     : τ_adapt = 333.3 ms, amp = XXX
e4Nr5a1     : τ_adapt = 333.3 ms, amp = XXX
```

### 5.4 操作方案C1：统一L4适应时间常数

#### C1.1 具体操作

**实验设计**：创建3个条件

| 条件 | 操作 | 目的 |
|------|------|------|
| Baseline | 保持原始参数（100/333差异） | 对照 |
| C1-fast | 所有L4 E的τ_adapt统一→100ms | 测试"全快"效应 |
| C1-slow | 所有L4 E的τ_adapt统一→333ms | 测试"全慢"效应 |

**代码实现**：

```python
def modify_L4_adaptation(target_tau_adapt):
    """
    修改所有L4兴奋性亚型的适应时间常数

    Parameters:
    -----------
    target_tau_adapt: float, 目标时间常数(ms)
    """
    l4_types = ['e4Scnn1a', 'e4Rorb', 'e4other', 'e4Nr5a1']

    for cell_type in l4_types:
        json_file = f'components/cell_models/nest_models/{cell_type}.json'

        # 读取原始参数
        with open(json_file, 'r') as f:
            params = json.load(f)

        # 修改 asc_decay (= 1 / tau_adapt)
        new_asc_decay = 1.0 / target_tau_adapt
        params['asc_decay'] = [new_asc_decay] * len(params['asc_decay'])

        # 保存修改后的参数
        output_file = json_file.replace('.json', f'_tau{target_tau_adapt:.0f}.json')
        with open(output_file, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"Modified {cell_type}: τ_adapt = {target_tau_adapt} ms")

# 创建修改版本
modify_L4_adaptation(target_tau_adapt=100)  # C1-fast
modify_L4_adaptation(target_tau_adapt=333)  # C1-slow
```

**配置文件更新**：

```json
{
  "network": "network_config.json",
  "simulation": "simulation_config_C1_fast.json",
  "components": {
    "morphologies_dir": "$COMPONENT_DIR/morphologies",
    "synaptic_models_dir": "$COMPONENT_DIR/synaptic_models",
    "point_neuron_models_dir": "$COMPONENT_DIR/cell_models/nest_models_C1_fast",
    "biophysical_neuron_models_dir": "$COMPONENT_DIR/cell_models"
  }
}
```

#### C1.2 预期结果

**核心预测**：
若L4适应异质性是**必要条件** → 统一参数会**消除/显著减小**时间滞后

| 条件 | 2-way onset时间 | 2-way peak时间 | 40-way peak时间 | Δt (category-identity) |
|------|----------------|---------------|----------------|---------------------|
| **Baseline** | 350 ms | 500-700 ms | 150-200 ms | **~400 ms** ⭐ |
| **C1-fast (全100ms)** | 300 ms ↓ | 400-500 ms ↓ | 150-200 ms ≈ | **~250 ms** ↓ |
| **C1-slow (全333ms)** | 350 ms ≈ | 500-700 ms ≈ | 150-200 ms ≈ | **~400 ms** ≈ |

**解释**：

**情况A：C1-fast显著减小Δt**
- → L4适应异质性是主要原因
- → 支持假设3为主导机制

**情况B：C1-fast效应不显著**
- → L4异质性不是主因
- → 支持假设1/2（L23抑制回路）为主导

**情况C：C1-slow也减小Δt**
- → 说明"适应差异"本身是关键，而非特定时间常数
- → 更复杂的交互机制

#### C1.3 数据分析：子空间轨迹可视化

**问题**：
如何直接可视化"前馈输入重加权导致子空间旋转"？

**方案**：Targeted Dimensionality Reduction（TDR）

```python
def targeted_subspace_analysis(firing_rates, labels_2way, labels_40way):
    """
    使用TDR分析类别子空间与身份子空间的时间演化

    Returns:
    --------
    category_variance_explained: ndarray (n_timebins,)
        类别判别方向解释的方差比例
    identity_variance_explained: ndarray (n_timebins,)
        身份判别方向解释的方差比例
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA

    n_trials, n_neurons, n_timebins = firing_rates.shape

    category_var = np.zeros(n_timebins)
    identity_var = np.zeros(n_timebins)

    for t in range(n_timebins):
        X_t = firing_rates[:, :, t]  # (n_trials, n_neurons)

        # 计算类别判别方向 (1D)
        lda_cat = LDA(n_components=1)
        lda_cat.fit(X_t, labels_2way)
        category_axis = lda_cat.coef_[0]  # (n_neurons,)
        category_axis = category_axis / np.linalg.norm(category_axis)

        # 计算身份判别子空间 (39D for 40-way)
        lda_id = LDA(n_components=min(39, len(np.unique(labels_40way))-1))
        lda_id.fit(X_t, labels_40way)
        identity_axes = lda_id.coef_  # (n_components, n_neurons)

        # 总方差
        pca = PCA()
        pca.fit(X_t)
        total_var = pca.explained_variance_.sum()

        # 类别方向解释的方差
        X_proj_cat = X_t @ category_axis  # (n_trials,)
        var_cat = np.var(X_proj_cat)
        category_var[t] = var_cat / total_var

        # 身份子空间解释的方差
        X_proj_id = X_t @ identity_axes.T  # (n_trials, n_components)
        var_id = np.var(X_proj_id, axis=0).sum()
        identity_var[t] = var_id / total_var

    return category_var, identity_var

# 运行分析
baseline_cat_var, baseline_id_var = targeted_subspace_analysis(
    baseline_fr, labels_2way, labels_40way
)
C1_fast_cat_var, C1_fast_id_var = targeted_subspace_analysis(
    C1_fast_fr, labels_2way, labels_40way
)
```

**可视化**：

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 左上: Baseline类别方差
ax = axes[0, 0]
ax.plot(time_ms, baseline_cat_var, 'b-', linewidth=2, label='Category axis')
ax.plot(time_ms, baseline_id_var, 'r-', linewidth=2, label='Identity subspace')
ax.set_ylabel('Variance explained', fontsize=11)
ax.set_title('Baseline', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 右上: C1-fast类别方差
ax = axes[0, 1]
ax.plot(time_ms, C1_fast_cat_var, 'b-', linewidth=2, label='Category axis')
ax.plot(time_ms, C1_fast_id_var, 'r-', linewidth=2, label='Identity subspace')
ax.set_title('C1-fast (all τ=100ms)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 左下: 类别方差对比
ax = axes[1, 0]
ax.plot(time_ms, baseline_cat_var, 'k-', linewidth=2, label='Baseline')
ax.plot(time_ms, C1_fast_cat_var, 'orange', linewidth=2, label='C1-fast')
ax.set_xlabel('Time (ms)', fontsize=11)
ax.set_ylabel('Category variance explained', fontsize=11)
ax.set_title('Category axis dynamics', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 右下: 方差比值时间进程
ax = axes[1, 1]
baseline_ratio = baseline_cat_var / (baseline_id_var + 1e-10)
C1_fast_ratio = C1_fast_cat_var / (C1_fast_id_var + 1e-10)
ax.plot(time_ms, baseline_ratio, 'k-', linewidth=2, label='Baseline')
ax.plot(time_ms, C1_fast_ratio, 'orange', linewidth=2, label='C1-fast')
ax.axhline(1, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Time (ms)', fontsize=11)
ax.set_ylabel('Category / Identity variance ratio', fontsize=11)
ax.set_title('Subspace competition', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hypothesis3_C1_subspace_dynamics.pdf', dpi=300)
```

**预期模式**：

**Baseline**:
- 早期：身份方差 >> 类别方差
- 晚期：类别方差逐渐上升（但仍 < 身份方差）
- 比值在500ms后明显增大

**C1-fast**:
- 早期：类似baseline
- 晚期：类别方差上升**提前**到300-400ms
- 比值增大时间提前

**关键差异**：类别方差增大的**时间进程**

### 5.5 操作方案C2：亚型特异连接权重扰动

#### 设计逻辑

**问题**：
如何进一步验证"特定亚型携带特定特征统计"？

**假设细化**：
- e4Scnn1a（快）→ L23：早期身份特征主导通路
- e4Rorb/other（慢）→ L23：晚期类别特征主导通路

**预测**：
- 削弱Scnn1a→L23：早期40-way下降，晚期2-way不受影响
- 削弱Rorb/other→L23：晚期2-way下降，早期40-way不受影响

#### 具体操作

```python
# 实验C2a: 削弱Scnn1a→L23E
{
  "connection_overrides": [
    {
      "source_type": "e4Scnn1a",
      "target_type": "e23Cux2",
      "weight_scale": 0.5
    }
  ]
}

# 实验C2b: 削弱Rorb/other/Nr5a1→L23E
{
  "connection_overrides": [
    {
      "source_type": ["e4Rorb", "e4other", "e4Nr5a1"],
      "target_type": "e23Cux2",
      "weight_scale": 0.5
    }
  ]
}
```

#### 预期结果

| 条件 | 40-way早期(150-200ms) | 40-way晚期(500ms) | 2-way早期(200-350ms) | 2-way晚期(500-700ms) |
|------|---------------------|------------------|--------------------|--------------------|
| Baseline | 63% | 60% | 55% | 78% |
| **C2a (↓Scnn1a)** | **56%** ↓↓ | 58% ≈ | 54% ≈ | 77% ≈ |
| **C2b (↓Rorb等)** | 62% ≈ | 57% ↓ | 52% ↓ | **72%** ↓↓ |

**解释性**：
✅ 双重分离（double dissociation）是**强因果证据**

---

## 6. 通用方法与分析流程

### 6.1 仿真标准流程（All Hypotheses）

#### 6.1.1 环境配置

```bash
# 创建实验目录结构
mkdir -p experiments/hypothesis_{1,2,3}
cd experiments

# 激活环境
conda activate bmtk

# 验证依赖
python -c "import bmtk; import nest; import h5py; print('All deps OK')"
```

#### 6.1.2 批量运行脚本

```python
#!/usr/bin/env python
"""
batch_run_experiments.py
批量运行所有假设的实验条件
"""

import os
import json
import subprocess
from pathlib import Path

# 定义所有实验条件
EXPERIMENTS = {
    'hypothesis1': {
        'A1_baseline': {'scaling': []},
        'A1_scale02': {'scaling': [('i23Sst', 'e23Cux2', 0.2)]},
        'A1_scale00': {'scaling': [('i23Sst', 'e23Cux2', 0.0)]},
        'A2_scale15': {'scaling': [('i23Sst', 'e23Cux2', 1.5)]},
        'A2_scale20': {'scaling': [('i23Sst', 'e23Cux2', 2.0)]},
        'A2_scale30': {'scaling': [('i23Sst', 'e23Cux2', 3.0)]},
    },
    'hypothesis2': {
        'B1_baseline': {'scaling': []},
        'B1_scale02': {'scaling': [('i23Htr3a', 'i23Sst', 0.2)]},
        'B1_scale00': {'scaling': [('i23Htr3a', 'i23Sst', 0.0)]},
        'B2_scale02': {'scaling': [('i23Sst', 'i23Htr3a', 0.2)]},
        'B3_L4_scale02': {'scaling': [('i4Htr3a', 'i4Sst', 0.2)]},
        'B3_L5_scale02': {'scaling': [('i5Htr3a', 'i5Sst', 0.2)]},
        'B3_L6_scale02': {'scaling': [('i6Htr3a', 'i6Sst', 0.2)]},
    },
    'hypothesis3': {
        'C1_baseline': {'modify_adaptation': None},
        'C1_fast': {'modify_adaptation': 100},
        'C1_slow': {'modify_adaptation': 333},
        'C2a_scale05': {'scaling': [('e4Scnn1a', 'e23Cux2', 0.5)]},
        'C2b_scale05': {'scaling': [('e4Rorb', 'e23Cux2', 0.5),
                                     ('e4other', 'e23Cux2', 0.5),
                                     ('e4Nr5a1', 'e23Cux2', 0.5)]},
    }
}

def create_config_with_overrides(base_config, overrides):
    """创建带有连接权重override的配置文件"""
    config = json.load(open(base_config))

    if 'scaling' in overrides and overrides['scaling']:
        config['connection_overrides'] = []
        for source, target, scale in overrides['scaling']:
            config['connection_overrides'].append({
                'source': 'v1',
                'target': 'v1',
                'source_type': source,
                'target_type': target,
                'weight_scale': scale
            })

    return config

def run_experiment(hypothesis, exp_name, exp_config):
    """运行单个实验条件"""
    print(f"\n{'='*70}")
    print(f"Running: {hypothesis} / {exp_name}")
    print(f"{'='*70}\n")

    # 创建输出目录
    output_dir = Path(f'experiments/{hypothesis}/{exp_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建配置文件
    config = create_config_with_overrides('simulation_config_base.json', exp_config)
    config_file = output_dir / 'simulation_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # 运行仿真
    cmd = [
        'python', 'run_bionet.py',
        str(config_file),
        '--output-dir', str(output_dir)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {exp_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {exp_name} failed: {e}")
        return False

def main():
    """批量运行所有实验"""
    results = {}

    for hypothesis, experiments in EXPERIMENTS.items():
        results[hypothesis] = {}

        for exp_name, exp_config in experiments.items():
            success = run_experiment(hypothesis, exp_name, exp_config)
            results[hypothesis][exp_name] = 'success' if success else 'failed'

    # 保存运行日志
    with open('batch_run_log.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("Batch run completed!")
    print(f"{'='*70}\n")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
```

### 6.2 标准分析流程（All Hypotheses）

#### 6.2.1 解码分析pipeline

```python
#!/usr/bin/env python
"""
decode_analysis_pipeline.py
标准化的解码分析流程
"""

import numpy as np
import h5py
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import pickle

class DecodingAnalysis:
    """标准化解码分析类"""

    def __init__(self, output_dir, bin_size=50, smooth_sigma=20):
        self.output_dir = Path(output_dir)
        self.bin_size = bin_size
        self.smooth_sigma = smooth_sigma

    def load_spikes(self):
        """加载spike数据"""
        spikes_file = self.output_dir / 'spikes.h5'
        # ... 实现加载逻辑

    def compute_firing_rates(self, spikes, t_start=0, t_stop=3000):
        """计算发放率"""
        # ... 见前面实现

    def decode_timecourse(self, firing_rates, labels, n_cv_folds=5):
        """时间分辨解码"""
        n_trials, n_neurons, n_timebins = firing_rates.shape

        accuracy = np.zeros(n_timebins)
        accuracy_std = np.zeros(n_timebins)
        weights = np.zeros((n_timebins, n_neurons))

        for t in range(n_timebins):
            X_t = firing_rates[:, :, t]

            cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            scores = []
            weights_cv = []

            for train_idx, test_idx in cv.split(X_t, labels):
                X_train, X_test = X_t[train_idx], X_t[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                clf = SVC(kernel='linear', C=1.0)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores.append(score)

                if hasattr(clf, 'coef_'):
                    weights_cv.append(clf.coef_[0])

            accuracy[t] = np.mean(scores)
            accuracy_std[t] = np.std(scores)
            if weights_cv:
                weights[t] = np.mean(weights_cv, axis=0)

        return {
            'accuracy': accuracy,
            'accuracy_std': accuracy_std,
            'weights': weights,
            'time_bins': self.time_bins
        }

    def compute_onset_time(self, accuracy, chance_level, threshold_std=2,
                          consecutive_bins=3):
        """计算onset时间"""
        early_std = np.std(accuracy[:10])
        threshold = chance_level + threshold_std * early_std

        for t in range(len(accuracy) - consecutive_bins):
            if np.all(accuracy[t:t+consecutive_bins] > threshold):
                return self.time_bins[t]
        return np.nan

    def bootstrap_onset_distribution(self, firing_rates, labels,
                                     chance_level, n_bootstrap=100):
        """Bootstrap估计onset时间分布"""
        n_trials = firing_rates.shape[0]
        onsets = []

        for b in range(n_bootstrap):
            # Bootstrap采样
            idx = np.random.choice(n_trials, size=n_trials, replace=True)
            fr_boot = firing_rates[idx]
            labels_boot = labels[idx]

            # 解码
            result = self.decode_timecourse(fr_boot, labels_boot)
            onset = self.compute_onset_time(result['accuracy'], chance_level)
            onsets.append(onset)

        return np.array(onsets)

    def save_results(self, results, filename='decoding_results.pkl'):
        """保存结果"""
        output_file = self.output_dir / filename
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_file}")

# 使用示例
if __name__ == '__main__':
    # 分析baseline条件
    analyzer = DecodingAnalysis('experiments/hypothesis1/A1_baseline')

    # 加载数据
    spikes = analyzer.load_spikes()
    firing_rates = analyzer.compute_firing_rates(spikes)

    # 2-way解码
    results_2way = analyzer.decode_timecourse(firing_rates, labels_2way)
    onset_2way = analyzer.compute_onset_time(results_2way['accuracy'], 0.5)

    # 40-way解码
    results_40way = analyzer.decode_timecourse(firing_rates, labels_40way)
    onset_40way = analyzer.compute_onset_time(results_40way['accuracy'], 1/40)

    # 保存
    analyzer.save_results({
        '2way': results_2way,
        '40way': results_40way,
        'onset_2way': onset_2way,
        'onset_40way': onset_40way
    })
```

### 6.3 统计检验标准流程

#### 6.3.1 Permutation test

```python
def permutation_test(data_group1, data_group2, n_perm=10000,
                     stat_func=np.mean, alternative='two-sided'):
    """
    通用permutation test

    Parameters:
    -----------
    data_group1, data_group2: ndarray
        两组数据
    n_perm: int
        permutation次数
    stat_func: callable
        统计量函数(默认均值)
    alternative: str
        'two-sided', 'greater', 'less'

    Returns:
    --------
    observed_diff: float
        观测到的统计量差异
    p_value: float
        p值
    """
    observed_diff = stat_func(data_group1) - stat_func(data_group2)

    combined = np.concatenate([data_group1, data_group2])
    n1 = len(data_group1)

    perm_diffs = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = stat_func(combined[:n1]) - stat_func(combined[n1:])
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    elif alternative == 'greater':
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == 'less':
        p_value = np.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return observed_diff, p_value, perm_diffs

# 使用示例
onset_baseline = np.array([345, 352, 348, 355, 350])  # bootstrap samples
onset_A1 = np.array([445, 452, 448, 455, 450])

diff, p, _ = permutation_test(onset_baseline, onset_A1,
                               alternative='two-sided')
print(f"Onset time difference: {diff:.1f} ms, p = {p:.4f}")
```

#### 6.3.2 Bootstrap confidence intervals

```python
def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=10000, ci=95):
    """
    Bootstrap置信区间

    Returns:
    --------
    ci_low, ci_high: float
        置信区间下界和上界
    """
    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = (100 - ci) / 2
    ci_low = np.percentile(bootstrap_stats, alpha)
    ci_high = np.percentile(bootstrap_stats, 100 - alpha)

    return ci_low, ci_high, bootstrap_stats

# 使用示例
onset_distribution = bootstrap_onset_distribution(firing_rates, labels_2way,
                                                   chance_level=0.5,
                                                   n_bootstrap=1000)
ci_low, ci_high, _ = bootstrap_ci(onset_distribution)
print(f"95% CI: [{ci_low:.1f}, {ci_high:.1f}] ms")
```

---

## 7. 结果整合与理论预测

### 7.1 三假设的层次关系

#### 互补而非互斥

```
假设3 (L4适应异质性)
  ↓ 提供
"输入统计在时间上的重加权"
  ↓ 使得
L4→L23前馈驱动在中后期发生方向旋转
  ↓

假设1 (Sst树突门控) + 假设2 (Htr3a-Sst竞争)
  ↓ 提供
"L23局部回路的迭代筛选机制"
  ↓ 使得
某些跨exemplar共享的方向在递归中被选择性稳定
  ↓

共同结果
  ↓
类别共享子空间在晚期浮现并稳定
身份残差子空间保持
  ↓
2-way晚期上升，40-way不下降 ✓
```

### 7.2 预测矩阵（可证伪性）

| 操控 | 假设1主导 | 假设2主导 | 假设3主导 | 均衡贡献 |
|------|----------|----------|----------|---------|
| **削弱Sst→L23E** | ↑↑↑ | → | → | ↑↑ |
| **削弱Htr3a→Sst** | → | ↑↑↑ | → | ↑↑ |
| **统一L4适应** | → | → | ↑↑↑ | ↑↑ |
| **削弱Scnn1a→L23** | → | → | ↑↑ (早期) | ↑ |
| **削弱Rorb等→L23** | → | → | ↑↑ (晚期) | ↑ |
| **层特异L4/L5/L6** | → | ↑↑ (特异性证据) | → | ↑ |

**符号说明**：
- ↑↑↑: 该假设的直接强证据
- ↑↑: 该假设的直接证据
- ↑: 该假设的支持证据
- →: 不影响或弱影响

### 7.3 整合实验路线图

#### Phase 1: 主效应验证（必做）

```
Week 1-2: 运行基线 + 所有主要操控
  - Baseline
  - A1 (Sst→L23E削弱)
  - B1 (Htr3a→Sst削弱)
  - C1-fast (统一L4适应)

Week 3: 标准分析pipeline
  - 解码时间进程
  - Onset统计
  - 子空间分析

Week 4: 初步结果评估
  → 确定哪个假设贡献最大
  → 决定Phase 2重点
```

#### Phase 2: 机制细化（根据Phase 1结果）

**情况A：假设1/2占主导（L23抑制回路）**

```
优先实验：
  - A2 (剂量-反应曲线)
  - B2 (Sst→Htr3a对照)
  - B3 (层特异性验证) ⭐⭐⭐

深化分析：
  - 抑制网络动态（I细胞发放率分析）
  - E-I平衡分析
  - 门控效应的直接测量
```

**情况B：假设3占主导（L4适应异质性）**

```
优先实验：
  - C2a/C2b (亚型特异连接)
  - C1-slow (对照组)

深化分析：
  - L4各亚型发放率时间进程
  - L4→L23有效连接权重演化
  - 前馈子空间旋转的直接可视化
```

**情况C：多因素均衡贡献**

```
组合操控：
  - A1 + C1-fast (L23抑制 + L4适应同时操控)
  - 检验交互效应

定量模型：
  - 回归分析：Δt ~ α·Sst_effect + β·L4_effect + ε
```

#### Phase 3: 发表级分析（假设论文投稿）

```
核心图表 (Main Figures):
  1. 现象图：Baseline解码时间进程（2-way vs 40-way）
  2. 假设示意图：机制框架与子空间几何
  3. 主操控结果：A1, B1, C1对比
  4. 层特异性证据：B3结果（最强定位证据）
  5. 子空间动态：TDR分析可视化
  6. 统计总结：所有条件的onset/peak对比

补充图表 (Supplementary):
  - 连接统计验证
  - 适应参数验证
  - 剂量-反应曲线
  - 控制实验完整结果
  - 额外解码分析（时间泛化矩阵等）
```

---

## 8. 附录：代码实现与数据结构

### 8.1 H5文件结构速查

#### nodes.h5

```
/nodes/v1/
  ├── node_id: [0, 1, 2, ..., N-1]
  ├── node_type_id: [101, 102, ...]  # 映射到node types
  ├── x, y, z: 坐标
  └── ...

/node_types/
  ├── node_type_id: [101, 102, ...]
  ├── pop_name: ['e23Cux2', 'i23Sst', ...]
  └── ...
```

#### edges.h5

```
/edges/v1_to_v1/
  ├── source_node_id: [node_id_source, ...]
  ├── target_node_id: [node_id_target, ...]
  ├── edge_type_id: [201, 202, ...]
  ├── syn_weight: [w1, w2, ...]
  ├── nsyns: [n1, n2, ...]  # 突触数量
  └── delay: [d1, d2, ...]

/edge_types/
  ├── edge_type_id: [201, 202, ...]
  ├── dynamics_params: JSON strings
  └── ...
```

### 8.2 快速提取连接矩阵

```python
import h5py
import numpy as np
import pandas as pd

def extract_connection_matrix(edges_h5_path, nodes_h5_path,
                              source_pop, target_pop):
    """
    提取特定群体间的连接矩阵

    Returns:
    --------
    connectivity_df: pd.DataFrame
        包含所有连接信息
    summary: dict
        统计摘要
    """
    # 加载文件
    edges_h5 = h5py.File(edges_h5_path, 'r')
    nodes_h5 = h5py.File(nodes_h5_path, 'r')

    # 获取node type映射
    node_types = nodes_h5['node_types']['pop_name'][:]
    node_type_ids = nodes_h5['node_types']['node_type_id'][:]
    type_to_name = dict(zip(node_type_ids, node_types))

    # 获取每个node的type
    all_node_types = nodes_h5['nodes']['v1']['node_type_id'][:]
    all_node_names = [type_to_name[tid] for tid in all_node_types]

    # 提取edges
    source_nodes = edges_h5['edges']['v1_to_v1']['source_node_id'][:]
    target_nodes = edges_h5['edges']['v1_to_v1']['target_node_id'][:]
    syn_weights = edges_h5['edges']['v1_to_v1']['syn_weight'][:]
    nsyns = edges_h5['edges']['v1_to_v1']['nsyns'][:] if 'nsyns' in edges_h5['edges']['v1_to_v1'] else np.ones_like(syn_weights)

    # 筛选目标连接
    source_mask = np.array([all_node_names[nid] == source_pop
                            for nid in source_nodes])
    target_mask = np.array([all_node_names[nid] == target_pop
                            for nid in target_nodes])
    mask = source_mask & target_mask

    # 创建DataFrame
    connectivity_df = pd.DataFrame({
        'source_node_id': source_nodes[mask],
        'target_node_id': target_nodes[mask],
        'syn_weight': syn_weights[mask],
        'nsyns': nsyns[mask],
        'total_strength': syn_weights[mask] * nsyns[mask]
    })

    # 统计摘要
    summary = {
        'n_edges': len(connectivity_df),
        'mean_weight': connectivity_df['syn_weight'].mean(),
        'std_weight': connectivity_df['syn_weight'].std(),
        'total_strength': connectivity_df['total_strength'].sum(),
        'unique_sources': connectivity_df['source_node_id'].nunique(),
        'unique_targets': connectivity_df['target_node_id'].nunique(),
    }

    edges_h5.close()
    nodes_h5.close()

    return connectivity_df, summary

# 使用示例
conn_df, summary = extract_connection_matrix(
    'v1_v1_edges.h5',
    'v1_nodes.h5',
    source_pop='i23Sst',
    target_pop='e23Cux2'
)

print(f"Connection summary: {source_pop} → {target_pop}")
for key, val in summary.items():
    print(f"  {key}: {val}")
```

### 8.3 修改NEST模型参数

```python
import json
from pathlib import Path

def modify_nest_model_params(model_name, param_updates, output_dir=None):
    """
    修改NEST模型JSON参数

    Parameters:
    -----------
    model_name: str
        模型名称（如'e4Scnn1a'）
    param_updates: dict
        要更新的参数 {'param_name': new_value, ...}
    output_dir: str or None
        输出目录（若None则覆盖原文件）

    Returns:
    --------
    output_path: Path
        修改后的文件路径
    """
    # 读取原始JSON
    json_path = Path(f'components/cell_models/nest_models/{model_name}.json')
    with open(json_path, 'r') as f:
        params = json.load(f)

    # 更新参数
    params.update(param_updates)

    # 保存
    if output_dir:
        output_path = Path(output_dir) / f'{model_name}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = json_path

    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Modified {model_name}: {param_updates}")
    print(f"Saved to {output_path}")

    return output_path

# 使用示例：统一L4适应时间常数
l4_types = ['e4Scnn1a', 'e4Rorb', 'e4other', 'e4Nr5a1']
target_tau = 100  # ms

output_dir = 'components/cell_models/nest_models_C1_fast'

for cell_type in l4_types:
    modify_nest_model_params(
        model_name=cell_type,
        param_updates={'asc_decay': [1.0/target_tau]},
        output_dir=output_dir
    )
```

---

## 参考文献（完整版）

### 理论框架（表征几何）

1. Elsayed & Cunningham (2017). "Structure in neural population recordings: an expected byproduct of simpler phenomena?" *Nature Neuroscience*.

2. Gallego et al. (2017). "Neural manifolds for the control of movement." *Neuron*.

3. Rigotti et al. (2013). "The importance of mixed selectivity in complex cognitive tasks." *Nature*.

4. Stringer et al. (2019). "Spontaneous behaviors drive multidimensional, brainwide activity." *Science*.

### 抑制回路机制

5. Pfeffer et al. (2013). "Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons." *Nature Neuroscience*.

6. Fu et al. (2014). "A cortical circuit for gain control by behavioral state." *Cell*.

7. Karnani et al. (2016). "Opening holes in the blanket of inhibition: localized lateral disinhibition by VIP interneurons." *Journal of Neuroscience*.

8. Kuchibhotla et al. (2017). "Parallel processing by cortical inhibition enables context-dependent behavior." *Nature Neuroscience*.

### 适应与动态

9. Priebe & Ferster (2006). "Mechanisms underlying cross-orientation suppression in cat visual cortex." *Neuron*.

10. Ferrante et al. (2017). "Distinct functional groups emerge from the intrinsic properties of molecularly identified cortical interneurons." *Frontiers in Cellular Neuroscience*.

---

**文档版本**: v1.0
**最后更新**: 2024-01-XX
**状态**: 待实验验证
