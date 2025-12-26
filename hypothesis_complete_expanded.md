# Allen V1模型中500ms时间滞后的神经机制：完整假设与实验方案

## 目录

1. [核心发现与数据基础](#核心发现与数据基础)
2. [主要假设：L4亚型时间分离机制](#主要假设l4亚型时间分离机制)
3. [文献支持与推理链](#文献支持与推理链)
4. [实验方案详细设计](#实验方案详细设计)
   - [实验1：减弱e4Scnn1a（快速亚型）](#实验1减弱e4scnn1a快速亚型)
   - [实验2：减弱e4Rorb/other/Nr5a1（慢速亚型）](#实验2减弱e4rorbothernr5a1慢速亚型)
   - [实验3：实测网络时间动态](#实验3实测网络时间动态)
   - [实验4：时间泛化解码](#实验4时间泛化解码)
   - [实验5：放大L23 E→E测试累积效应](#实验5放大l23-ee测试累积效应)
   - [实验6：放大L5→L23测试选择性](#实验6放大l5l23测试选择性)
   - [实验7：选择性调节抑制子型](#实验7选择性调节抑制子型)
5. [辅助假设与机制](#辅助假设与机制)
6. [总结与验证路线图](#总结与验证路线图)

---

## 核心发现与数据基础

### 观测现象

在Allen Institute V1点神经元模型中，使用L2/3兴奋性神经元（e23Cux2）活动解码视觉刺激类别时，发现：

- **精细分类（40类）**：解码准确率在 ~150-200ms 达到峰值
- **抽象分类（2类：有生命/无生命）**：解码准确率在 ~500-700ms 达到峰值
- **时间间距Δt**：约 **500ms**

**关键问题**：两种分类信息都从**同一群L23兴奋性神经元**解码，为何时间差距如此之大？

---

### 发现1：L4→L23绝对主导 ⭐⭐⭐⭐⭐

**数据来源**：H5文件分析（`v1_v1_edges.h5`）

| 通路 | 连接数 | 平均权重 | 总强度 | 相对比例 |
|------|--------|---------|--------|----------|
| **L4 E → L23 E** | 7,441,406 | 2.815 | 20,941,021 | **100%** (基准) |
| L23 E → L23 E | 10,434,976 | 0.390 | 4,064,535 | 19% |
| L5 E → L23 E | 820,060 | 1.503 | 1,228,763 | 6% |

**关键结论**：
- ✅ L4前馈是L23兴奋性输入的**绝对主导来源**（100%基准）
- ✅ L23横向连接虽然数量多（10M+），但单突触权重很弱（仅为L4的14%）
- ✅ L5反馈极弱（仅6%）
- ⚠️ **时间差异必须来自L4内部的异质性**，因为L4是主要信息源

---

### 发现2：L4亚型有233ms适应时间差异 ⭐⭐⭐⭐⭐

**数据来源**：神经元模型参数（`components/cell_models/nest_models/*.json`）

| L4亚型 | asc_decay[0] | τ_adapt_slow | 到L23连接数 | 总强度 | 比例 |
|--------|-------------|-------------|-----------|--------|------|
| **e4Scnn1a** | 0.01 | **100 ms** | 2,232,541 | 6,279,926 | 30% |
| e4Rorb | 0.003 | **333 ms** | 1,935,148 | 5,452,255 | 26% |
| e4other | 0.003 | **333 ms** | 2,382,776 | 6,707,915 | 32% |
| e4Nr5a1 | 0.003 | **333 ms** | 890,941 | 2,501,926 | 12% |

**关键计算**：
```python
# 适应时间常数 = 1 / asc_decay
τ_adapt_slow = 1 / asc_decay[0]

e4Scnn1a:  τ = 1/0.01  = 100 ms
e4Rorb:    τ = 1/0.003 = 333 ms
e4other:   τ = 1/0.003 = 333 ms
e4Nr5a1:   τ = 1/0.003 = 333 ms

# 时间差异
Δτ = 333 - 100 = 233 ms
```

**关键结论**：
- ✅ **e4Scnn1a（快速亚型）占L4→L23总强度的30%**
- ✅ **e4Rorb/other/Nr5a1（慢速亚型）占70%**
- ✅ **233ms适应差异可以部分解释500ms时间滞后**
- ✅ 这是**实际模型参数**，不是推测

---

### 发现3：传输延迟无法解释时间差异

**数据来源**：`Allen_V1_Complete_Properties_Tables.md`

| 通路 | 平均延迟 | 延迟范围 |
|------|---------|---------|
| L4 E → L23 E | 2.30 ms | 2.30 - 2.30 ms |
| L23 E → L23 E | 1.60 ms | 1.60 - 1.60 ms |
| L5 E → L23 E | 3.00 ms | 3.00 - 3.00 ms |
| PV → L23 E | 0.90 ms | 0.90 - 0.90 ms |
| Sst → L23 E | 1.50 ms | 1.50 - 1.50 ms |

**关键结论**：
- ❌ 所有传输延迟都 <5ms
- ❌ L4前馈（2.3ms）vs L5反馈（3.0ms）差异仅0.7ms
- ❌ **传输延迟无法解释500ms时间滞后**

---

## 主要假设：L4亚型时间分离机制

### 机制描述

```
刺激呈现 (t=0)
    ↓
LGN → L4快速激活 (t=0-50ms)
    ↓
L4内部分化：

┌─────────────────────────────┐  ┌──────────────────────────────────┐
│ [快速通路 - e4Scnn1a]       │  │ [慢速通路 - e4Rorb/other/Nr5a1] │
│ τ_adapt = 100 ms            │  │ τ_adapt = 333 ms                 │
│ 占L4→L23总强度的30%         │  │ 占L4→L23总强度的70%              │
│                             │  │                                  │
│ 快速克服spike频率适应       │  │ 需要更长时间克服适应             │
│ (~150ms达到稳定)           │  │ (~400ms达到稳定)                │
│          ↓                  │  │          ↓                       │
│ 驱动L23早期状态             │  │ 驱动L23晚期状态                  │
│ (提供30%兴奋性输入)         │  │ (提供70%兴奋性输入)              │
│          ↓                  │  │          ↓                       │
│ 精细分类可解码              │  │ 抽象分类可解码                   │
│ (~150-200ms)               │  │ (~500-700ms)                    │
└─────────────────────────────┘  └──────────────────────────────────┘
```

### 关键机制要素

1. **同一L23群体，不同时间窗口的输入组成**
   - 所有L4亚型都投射到同一L23 e23Cux2群体
   - 早期（0-200ms）：e4Scnn1a主导输入（30%）
   - 晚期（400-700ms）：e4Rorb/other/Nr5a1主导输入（70%）

2. **适应电流控制时间演化**
   - e4Scnn1a：100ms适应 → 快速恢复发放 → 早期高活性
   - e4Rorb/other/Nr5a1：333ms适应 → 慢速恢复发放 → 晚期高活性

3. **L23表征的时间演化**
   - L23神经元在不同时间整合不同的L4输入组合
   - 早期：主要反映e4Scnn1a编码的特征（精细）
   - 晚期：主要反映e4Rorb/other/Nr5a1编码的特征（抽象）

### 功能分工推测

**核心推测**（需要实验验证）：

- **e4Scnn1a（快速亚型）**：
  - 编码精细分类所需的特征
  - 局部、快速响应的特征
  - 例如：纹理、边缘细节、局部对比度

- **e4Rorb/other/Nr5a1（慢速亚型）**：
  - 编码抽象分类所需的特征
  - 需要更长时间整合的全局特征
  - 例如：整体形状、空间布局、类别原型

**注意**：这个功能分工推测**没有直接文献支持**，是基于时间尺度的合理类比，需要通过实验1-2直接验证。

---

## 文献支持与推理链

### 直接支持1：适应电流可达数百毫秒 ⭐⭐⭐⭐⭐

**文献**：[Multiple Time Scales of Temporal Response in Pyramidal Neurons (J Neurophysiol, 2006)](https://journals.physiology.org/doi/abs/10.1152/jn.00453.2006)

**引用原文**：
> "Pyramidal neurons exhibit adaptation/facilitation processes covering a wide range of timescales ranging from tens of milliseconds to seconds"

**直接支持的内容**：
- ✅ 锥体神经元的适应可以达到**数百毫秒到秒级**
- ✅ Allen V1模型中L4亚型的100ms vs 333ms在这个范围内

**推理链**：
```
文献事实：适应可达数百ms
    ↓
Allen V1数据：e4Scnn1a=100ms, e4Rorb/other/Nr5a1=333ms
    ↓
推论：这两个时间尺度都在生物学可行范围内
    ↓
支持：233ms差异是合理的时间滞后来源
```

---

### 直接支持2：适应支持时间整合计算 ⭐⭐⭐⭐⭐

**文献**：[Spike frequency adaptation supports network computations (eLife, 2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8313230/)

**引用原文**：
> "Spike frequency adaptation supports network computations on temporally dispersed information"

**直接支持的内容**：
- ✅ 适应电流（spike frequency adaptation）对时间分散信息的网络计算至关重要
- ✅ 不同的适应时间常数可以支持不同时间尺度的计算

**推理链**：
```
文献事实：适应支持时间整合
    ↓
Allen V1数据：L4亚型有不同适应（100 vs 333ms）
    ↓
推论：不同亚型可能整合不同时间尺度的信息
    ↓
假设：e4Scnn1a整合快速特征（精细分类）
       e4Rorb/other/Nr5a1整合慢速整合特征（抽象分类）
```

---

### 间接支持：循环处理细化表征 ⭐⭐⭐⭐

**文献**：[The logic of recurrent circuits in V1 (Nature Neurosci, 2024)](https://www.nature.com/articles/s41593-023-01510-5)

**引用原文**：
> "Recurrent cortical activity sculpts visual perception by refining, amplifying or suppressing visual input"

**间接支持的内容**：
- 循环处理对精细化表征至关重要
- 在V1中，循环活动可以细化视觉输入

**推理链（间接）**：
```
文献事实：V1的循环处理细化表征
    ↓
Allen V1数据：L23 E→E横向连接存在（虽弱，19%）
    ↓
推测：L23 E→E可能帮助整合慢速L4亚型的输入
    ↓
间接支持：抽象分类可能依赖L23内部的横向整合
```

**注意**：这是**间接支持**，因为文献没有直接提到"适应差异"或"亚型分离"。

---

### 缺少直接支持的部分 ⚠️⚠️⚠️

**核心推测**：e4Scnn1a（100ms适应）主要编码精细分类所需的特征，e4Rorb/other/Nr5a1（333ms适应）主要编码抽象分类所需的特征

**目前证据**：
- ✅ 有233ms适应差异（Allen V1模型参数，**事实**）
- ✅ 适应可达数百毫秒（文献支持）
- ✅ 早期视觉皮层可以支持高级分类（文献支持）
- ⚠️ **缺少直接证据**：
  - ❌ 没有文献直接表明L4神经元亚型的适应参数与分类层级有关
  - ❌ 没有文献表明Scnn1a vs Rorb有不同的功能分工（精细vs抽象）
  - ❌ 没有实验证据表明这些亚型编码不同层次的信息

**为什么仍然值得测试**？

1. ✅ 233ms适应差异是**实际存在的**（不是假设）
2. ✅ 这个差异的时间尺度接近观测的500ms滞后
3. ✅ 实验设计可以**直接证伪**这个假设
4. ✅ 如果正确，这将是**新发现**（文献中没有类似报道）
5. ✅ 如果错误，也能排除一个机制，帮助找到真正的原因

---

## 实验方案详细设计

### 实验设计总览

| 实验 | 目的 | 操作 | 预期结果（如果假设正确） | 优先级 |
|-----|------|------|------------------------|--------|
| 实验1 | 验证e4Scnn1a支持精细分类 | 减弱e4Scnn1a→L23 | 精细分类延迟/受损，抽象分类不变 | ⭐⭐⭐⭐⭐ |
| 实验2 | 验证慢速亚型支持抽象分类 | 减弱e4Rorb/other/Nr5a1→L23 | 抽象分类延迟/受损，精细分类不变 | ⭐⭐⭐⭐⭐ |
| 实验3 | 实测L4亚型时间动态 | 记录PSTH和L23膜电位 | e4Scnn1a峰值~150ms，慢速亚型峰值~400ms | ⭐⭐⭐⭐⭐ |
| 实验4 | 验证L23表征时间演化 | 时间泛化矩阵 | 早期/晚期解码器不泛化 | ⭐⭐⭐⭐ |
| 实验5 | 测试L23 E→E累积效应 | 放大L23 E→E | 抽象分类加快（如果横向整合重要） | ⭐⭐⭐ |
| 实验6 | 测试L5反馈选择性 | 放大L5→L23 | 抽象分类改善（如果L5重要） | ⭐⭐ |
| 实验7 | 测试抑制子型时间分工 | 选择性减弱PV/Sst | PV影响精细，Sst影响抽象 | ⭐⭐⭐ |

---

### 实验1：减弱e4Scnn1a（快速亚型）

#### 实验目的

验证e4Scnn1a（100ms适应）是否主要支持精细分类（40类）。

#### 详细操作步骤

**步骤1：识别e4Scnn1a到L23的连接**

```python
import h5py
import numpy as np
import pandas as pd

# 读取节点类型信息
node_types = pd.read_csv('network/v1_node_types.csv')
nodes_df = pd.read_csv('network/v1_nodes.h5')  # 或用h5py读取

# 识别e4Scnn1a和L23兴奋性神经元的node_type_id
e4scnn1a_type_ids = node_types[node_types['pop_name'] == 'e4Scnn1a']['node_type_id'].values
l23_exc_type_ids = node_types[node_types['pop_name'] == 'e23Cux2']['node_type_id'].values

print(f"e4Scnn1a node_type_ids: {e4scnn1a_type_ids}")
print(f"L23 excitatory node_type_ids: {l23_exc_type_ids}")

# 建立node_id到node_type_id的映射
# 从v1_nodes.h5读取
with h5py.File('network/v1_nodes.h5', 'r') as f:
    node_ids = f['nodes']['v1']['node_id'][:]
    node_type_ids = f['nodes']['v1']['node_type_id'][:]

node_id_to_type = dict(zip(node_ids, node_type_ids))

# 识别属于e4Scnn1a和L23 E的具体node_id
e4scnn1a_node_ids = [nid for nid, ntype in node_id_to_type.items()
                      if ntype in e4scnn1a_type_ids]
l23_exc_node_ids = [nid for nid, ntype in node_id_to_type.items()
                     if ntype in l23_exc_type_ids]

print(f"Found {len(e4scnn1a_node_ids)} e4Scnn1a neurons")
print(f"Found {len(l23_exc_node_ids)} L23 excitatory neurons")
```

**步骤2：筛选e4Scnn1a→L23 E的连接**

```python
# 读取边信息
with h5py.File('network/v1_v1_edges.h5', 'r+') as f:
    edges_group = f['edges']['v1_to_v1']

    # 读取源和目标节点ID
    source_node_id = edges_group['source_node_id'][:]
    target_node_id = edges_group['target_node_id'][:]

    # 创建mask：源是e4Scnn1a，目标是L23 E
    mask_e4scnn1a_to_l23e = (
        np.isin(source_node_id, e4scnn1a_node_ids) &
        np.isin(target_node_id, l23_exc_node_ids)
    )

    print(f"Found {mask_e4scnn1a_to_l23e.sum()} connections from e4Scnn1a to L23E")

    # 获取edge_type_id，找到对应的权重组
    edge_type_id = edges_group['edge_type_id'][:]

    # Allen V1模型中，权重存储在edge_type_id对应的子组中
    # 需要找到所有涉及的edge_type
    unique_edge_types = np.unique(edge_type_id[mask_e4scnn1a_to_l23e])
    print(f"Edge types involved: {unique_edge_types}")
```

**步骤3：减弱连接权重**

```python
# 定义减弱因子
weakening_factors = [0.0, 0.2, 0.5, 0.8, 1.0]  # 0=完全消除, 1.0=不变

for factor in weakening_factors:
    print(f"\n{'='*70}")
    print(f"Testing weakening factor: {factor}")
    print(f"{'='*70}")

    # 修改权重
    with h5py.File('network/v1_v1_edges.h5', 'r+') as f:
        edges_group = f['edges']['v1_to_v1']

        # 对每个edge_type子组
        for edge_type in unique_edge_types:
            edge_type_str = str(edge_type)
            if edge_type_str in edges_group:
                # 获取该edge_type的权重数据
                syn_weight = edges_group[edge_type_str]['syn_weight']

                # 找到属于e4Scnn1a→L23E的连接索引
                edge_type_mask = (edge_type_id == edge_type)
                combined_mask = mask_e4scnn1a_to_l23e & edge_type_mask

                # 获取原始权重
                original_weights = syn_weight[combined_mask]

                # 修改权重
                syn_weight[combined_mask] = original_weights * factor

                print(f"  Edge type {edge_type}: Modified {combined_mask.sum()} connections")
                print(f"    Original mean weight: {original_weights.mean():.4f}")
                print(f"    New mean weight: {(original_weights * factor).mean():.4f}")
```

**步骤4：运行模拟**

```python
from bmtk.simulator import pointnet
import json

# 配置模拟参数
config = {
    'run': {
        'tstop': 1000.0,  # 模拟1000ms
        'dt': 0.1,        # 0.1ms时间步长
        'dL': 20.0,
        'overwrite_output_dir': True
    },
    'conditions': {
        'celsius': 34.0,
        'v_init': -80.0
    },
    'inputs': {
        # 这里需要配置LGN输入或虚拟输入
        # 根据实际的刺激协议配置
        'LGN_spikes': {
            'input_type': 'spikes',
            'module': 'nwb',
            'input_file': 'inputs/lgn_spikes.nwb',  # 预先准备的LGN输入
            'node_set': 'lgn'
        }
    },
    'reports': {
        # 记录L23兴奋性神经元的发放
        'spikes': {
            'file_name': f'output/spikes_e4scnn1a_factor{factor}.h5',
            'module': 'spikes_report',
            'cells': 'all'
        },
        # 记录L23神经元的膜电位
        'membrane_potential': {
            'file_name': f'output/v_report_factor{factor}.h5',
            'module': 'membrane_report',
            'variable_name': 'v',
            'cells': 'L23_exc',
            'sections': 'soma',
            'buffer_size': 10000
        }
    },
    'output': {
        'output_dir': f'output/exp1_factor{factor}',
        'log_file': f'log_factor{factor}.txt',
        'spikes_file': f'spikes_factor{factor}.h5',
        'spikes_file_csv': f'spikes_factor{factor}.csv'
    },
    'network': 'network/v1_network_config.json',
    'components': {
        'synaptic_models_dir': 'components/synaptic_models',
        'point_neuron_models_dir': 'components/cell_models',
        'morphologies_dir': None
    }
}

# 保存配置文件
with open(f'config_exp1_factor{factor}.json', 'w') as f:
    json.dump(config, f, indent=2)

# 运行模拟
print(f"Running simulation with factor={factor}...")
pointnet.run_pointnet(f'config_exp1_factor{factor}.json')
```

**步骤5：解码分析**

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

def decode_categories(spike_data, time_windows, labels_fine, labels_abstract):
    """
    对每个时间窗口进行解码

    参数:
    - spike_data: shape (n_trials, n_neurons, n_timebins)
    - time_windows: list of (start_ms, end_ms) tuples
    - labels_fine: 40-class labels
    - labels_abstract: 2-class labels

    返回:
    - results_fine: 精细分类准确率时间序列
    - results_abstract: 抽象分类准确率时间序列
    """
    results_fine = []
    results_abstract = []

    for t_start, t_end in time_windows:
        # 提取该时间窗口的发放率
        start_bin = int(t_start / 0.1)  # 0.1ms时间步长
        end_bin = int(t_end / 0.1)

        # 计算每个trial在该窗口的平均发放率
        firing_rates = spike_data[:, :, start_bin:end_bin].mean(axis=2)

        # 精细分类（40类）
        clf_fine = LogisticRegressionCV(
            Cs=10,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=-1
        )
        scores_fine = cross_val_score(
            clf_fine, firing_rates, labels_fine,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        results_fine.append(scores_fine.mean())

        # 抽象分类（2类）
        clf_abstract = LogisticRegressionCV(
            Cs=10,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1
        )
        scores_abstract = cross_val_score(
            clf_abstract, firing_rates, labels_abstract,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        results_abstract.append(scores_abstract.mean())

    return np.array(results_fine), np.array(results_abstract)

# 定义时间窗口（每50ms滑动窗口）
time_windows = [(t, t+50) for t in range(0, 700, 10)]
time_centers = [(t[0] + t[1])/2 for t in time_windows]

# 对每个factor运行解码
results_all = {}
for factor in weakening_factors:
    # 加载spike数据
    spike_data = load_spike_data(f'output/exp1_factor{factor}/spikes.h5')

    # 解码
    acc_fine, acc_abstract = decode_categories(
        spike_data, time_windows, labels_fine, labels_abstract
    )

    results_all[factor] = {
        'fine': acc_fine,
        'abstract': acc_abstract
    }

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 精细分类
ax = axes[0]
for factor in weakening_factors:
    ax.plot(time_centers, results_all[factor]['fine'],
            label=f'Factor={factor}', linewidth=2)
ax.axhline(1/40, color='k', linestyle='--', label='Chance (40-way)')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Decoding Accuracy', fontsize=12)
ax.set_title('Fine-grained (40-way) Categorization', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 抽象分类
ax = axes[1]
for factor in weakening_factors:
    ax.plot(time_centers, results_all[factor]['abstract'],
            label=f'Factor={factor}', linewidth=2)
ax.axhline(0.5, color='k', linestyle='--', label='Chance (2-way)')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Decoding Accuracy', fontsize=12)
ax.set_title('Abstract (2-way) Categorization', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/exp1_decoding_results.pdf', dpi=300)
plt.show()
```

#### 预期结果

**如果假设正确**（e4Scnn1a主要支持精细分类）：

**定量预测**：

| 减弱因子 | 精细分类峰值时间 | 精细分类峰值准确率 | 抽象分类峰值时间 | 抽象分类峰值准确率 |
|---------|-----------------|-------------------|-----------------|-------------------|
| 1.0（基线） | ~150-200ms | ~0.30 (40类) | ~500-700ms | ~0.85 (2类) |
| 0.8 | ~180-230ms (+30ms) | ~0.25 (-17%) | ~500-700ms (不变) | ~0.83 (-2%) |
| 0.5 | ~250-300ms (+100ms) | ~0.18 (-40%) | ~500-700ms (不变) | ~0.80 (-6%) |
| 0.2 | ~350-450ms (+200ms) | ~0.10 (-67%) | ~550-750ms (+50ms) | ~0.75 (-12%) |
| 0.0（完全消除） | >700ms或无法解码 | ~0.05 (接近chance) | ~600-800ms (+100ms) | ~0.68 (-20%) |

**关键观察**：
- ✅ **精细分类显著延迟**：峰值时间从150ms→350ms以上
- ✅ **精细分类准确率下降**：特别是早期时间窗口（0-300ms）
- ✅ **抽象分类相对保持**：峰值时间基本不变或轻微延迟
- ✅ **时间间距减小**：精细分类"追上"抽象分类

**判据**：
```python
# 计算关键指标
def compute_metrics(results):
    # 精细分类峰值时间
    peak_time_fine = time_centers[np.argmax(results['fine'])]
    # 抽象分类峰值时间
    peak_time_abstract = time_centers[np.argmax(results['abstract'])]
    # 时间间距
    delta_t = peak_time_abstract - peak_time_fine

    return peak_time_fine, peak_time_abstract, delta_t

# 对每个factor计算
for factor in weakening_factors:
    t_fine, t_abstract, delta = compute_metrics(results_all[factor])
    print(f"Factor {factor}: Fine peak={t_fine:.0f}ms, "
          f"Abstract peak={t_abstract:.0f}ms, Δt={delta:.0f}ms")

# 判据
baseline_delta = compute_metrics(results_all[1.0])[2]
weak_delta = compute_metrics(results_all[0.2])[2]

if baseline_delta > 400 and weak_delta < 250:
    print("\n✅ 强烈支持假设：e4Scnn1a主要支持精细分类")
elif weak_delta > baseline_delta:
    print("\n❌ 不支持假设：减弱e4Scnn1a反而增大时间间距")
else:
    print("\n⚠️ 结果不明确，需要进一步分析")
```

---

**如果假设错误**：

可能的其他情况：
1. **两种分类都严重受损**：
   - 说明e4Scnn1a对两者都重要（无选择性）
   - 需要重新考虑机制

2. **两种分类都基本不变**：
   - 说明e4Scnn1a不重要，或50%减弱幅度太小
   - 需要更大幅度调节（尝试factor=0或0.1）

3. **抽象分类受损更严重**：
   - 与假设相反
   - 说明e4Scnn1a可能主要支持抽象分类
   - 需要重新审视假设

#### 补充分析

**分析1：时间分辨率更高的解码**

```python
# 使用更短的时间窗口（10ms）
time_windows_fine = [(t, t+10) for t in range(0, 700, 5)]

# 重复解码分析...
```

**分析2：单神经元贡献分析**

```python
# 计算每个L23神经元对精细vs抽象分类的贡献
from sklearn.linear_model import LogisticRegression

def compute_neuron_selectivity(spike_data, labels_fine, labels_abstract):
    """
    计算每个神经元对精细vs抽象分类的选择性
    """
    n_neurons = spike_data.shape[1]
    selectivity_fine = np.zeros(n_neurons)
    selectivity_abstract = np.zeros(n_neurons)

    # 对每个神经元单独解码
    for i in range(n_neurons):
        # 使用整个时间窗口的发放率
        firing_rates = spike_data[:, i, :].mean(axis=1, keepdims=True)

        # 精细分类
        clf = LogisticRegression(max_iter=1000)
        scores_fine = cross_val_score(clf, firing_rates, labels_fine, cv=5)
        selectivity_fine[i] = scores_fine.mean()

        # 抽象分类
        scores_abstract = cross_val_score(clf, firing_rates, labels_abstract, cv=5)
        selectivity_abstract[i] = scores_abstract.mean()

    return selectivity_fine, selectivity_abstract

# 对基线和减弱条件分析
sel_fine_baseline, sel_abstract_baseline = compute_neuron_selectivity(
    spike_data_baseline, labels_fine, labels_abstract
)

sel_fine_weak, sel_abstract_weak = compute_neuron_selectivity(
    spike_data_weak, labels_fine, labels_abstract
)

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(sel_fine_baseline, sel_abstract_baseline,
           alpha=0.3, label='Baseline')
ax.scatter(sel_fine_weak, sel_abstract_weak,
           alpha=0.3, label='e4Scnn1a weakened')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('Fine-grained selectivity', fontsize=12)
ax.set_ylabel('Abstract selectivity', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('results/exp1_neuron_selectivity.pdf', dpi=300)
```

---

### 实验2：减弱e4Rorb/other/Nr5a1（慢速亚型）

#### 实验目的

验证慢速亚型（333ms适应）是否主要支持抽象分类（2类）。

#### 详细操作步骤

**与实验1类似，但筛选慢速亚型：**

```python
# 识别三个慢速亚型的节点
slow_pops = ['e4Rorb', 'e4other', 'e4Nr5a1']
slow_type_ids = node_types[node_types['pop_name'].isin(slow_pops)]['node_type_id'].values

slow_node_ids = [nid for nid, ntype in node_id_to_type.items()
                  if ntype in slow_type_ids]

print(f"Found {len(slow_node_ids)} slow L4 neurons")
print(f"  e4Rorb: {len([n for n in slow_node_ids if node_id_to_type[n] in node_types[node_types['pop_name']=='e4Rorb']['node_type_id'].values])}")
print(f"  e4other: {len([n for n in slow_node_ids if node_id_to_type[n] in node_types[node_types['pop_name']=='e4other']['node_type_id'].values])}")
print(f"  e4Nr5a1: {len([n for n in slow_node_ids if node_id_to_type[n] in node_types[node_types['pop_name']=='e4Nr5a1']['node_type_id'].values])}")

# 创建mask
mask_slow_to_l23e = (
    np.isin(source_node_id, slow_node_ids) &
    np.isin(target_node_id, l23_exc_node_ids)
)

print(f"Found {mask_slow_to_l23e.sum()} connections from slow L4 to L23E")

# 修改权重（使用相同的weakening_factors）
# ... (代码与实验1类似)
```

#### 预期结果

**如果假设正确**（慢速亚型主要支持抽象分类）：

**定量预测**：

| 减弱因子 | 精细分类峰值时间 | 精细分类峰值准确率 | 抽象分类峰值时间 | 抽象分类峰值准确率 |
|---------|-----------------|-------------------|-----------------|-------------------|
| 1.0（基线） | ~150-200ms | ~0.30 | ~500-700ms | ~0.85 |
| 0.8 | ~150-200ms (不变) | ~0.28 (-7%) | ~550-750ms (+50ms) | ~0.78 (-8%) |
| 0.5 | ~150-200ms (不变) | ~0.25 (-17%) | ~650-850ms (+150ms) | ~0.65 (-24%) |
| 0.2 | ~150-250ms (+50ms) | ~0.20 (-33%) | >800ms (+300ms) | ~0.58 (-32%) |
| 0.0（完全消除） | ~100-200ms (可能提前) | ~0.18 (-40%) | >1000ms或无法解码 | ~0.52 (接近chance) |

**关键观察**：
- ✅ **抽象分类显著延迟**：峰值时间从500ms→800ms以上
- ✅ **抽象分类准确率显著下降**：特别是晚期窗口（400-700ms）
- ✅ **精细分类相对保持**：峰值时间基本不变或轻微延迟
- ✅ **时间间距增大**：抽象分类进一步延迟

#### 对比实验1和2的判据

**综合判据**（最关键）：

```python
def comprehensive_test(results_exp1, results_exp2):
    """
    综合测试L4亚型时间分离假设
    """
    # 实验1：减弱e4Scnn1a
    # 预期：精细受损 > 抽象受损
    fine_damage_exp1 = (results_exp1[1.0]['fine'].max() -
                        results_exp1[0.2]['fine'].max())
    abstract_damage_exp1 = (results_exp1[1.0]['abstract'].max() -
                            results_exp1[0.2]['abstract'].max())

    # 实验2：减弱慢速亚型
    # 预期：抽象受损 > 精细受损
    fine_damage_exp2 = (results_exp2[1.0]['fine'].max() -
                        results_exp2[0.2]['fine'].max())
    abstract_damage_exp2 = (results_exp2[1.0]['abstract'].max() -
                            results_exp2[0.2]['abstract'].max())

    # 计算选择性指数
    selectivity_exp1 = (fine_damage_exp1 - abstract_damage_exp1) / \
                       (fine_damage_exp1 + abstract_damage_exp1 + 1e-6)
    selectivity_exp2 = (abstract_damage_exp2 - fine_damage_exp2) / \
                       (abstract_damage_exp2 + fine_damage_exp2 + 1e-6)

    print(f"\n{'='*70}")
    print("综合测试结果")
    print(f"{'='*70}")
    print(f"\n实验1（减弱e4Scnn1a）:")
    print(f"  精细分类受损: {fine_damage_exp1:.3f}")
    print(f"  抽象分类受损: {abstract_damage_exp1:.3f}")
    print(f"  选择性指数: {selectivity_exp1:.3f} (>0表示对精细分类选择性)")

    print(f"\n实验2（减弱慢速亚型）:")
    print(f"  精细分类受损: {fine_damage_exp2:.3f}")
    print(f"  抽象分类受损: {abstract_damage_exp2:.3f}")
    print(f"  选择性指数: {selectivity_exp2:.3f} (>0表示对抽象分类选择性)")

    # 判据
    if selectivity_exp1 > 0.3 and selectivity_exp2 > 0.3:
        print(f"\n{'='*70}")
        print("✅ ✅ ✅ 强烈支持L4亚型时间分离假设 ✅ ✅ ✅")
        print(f"{'='*70}")
        print("\n证据:")
        print("  1. e4Scnn1a选择性支持精细分类")
        print("  2. e4Rorb/other/Nr5a1选择性支持抽象分类")
        print("  3. 233ms适应差异是主要机制")
        return "STRONG_SUPPORT"
    elif selectivity_exp1 < 0 or selectivity_exp2 < 0:
        print(f"\n{'='*70}")
        print("❌ 不支持假设")
        print(f"{'='*70}")
        print("\n证据:")
        print("  - 选择性指数为负，说明效应与预期相反")
        return "REJECTED"
    else:
        print(f"\n{'='*70}")
        print("⚠️ 部分支持，需要进一步分析")
        print(f"{'='*70}")
        return "PARTIAL_SUPPORT"

# 运行综合测试
result = comprehensive_test(results_exp1_all, results_exp2_all)
```

---

### 实验3：实测网络时间动态

#### 实验目的

**不依赖解码**，直接测量L4亚型的实际激活时间差异，验证适应参数是否能预测实际时间动态。

#### 详细操作步骤

**步骤1：配置高时间分辨率记录**

```python
config_exp3 = {
    'run': {
        'tstop': 1000.0,
        'dt': 0.1,  # 0.1ms高精度
        'dL': 20.0
    },
    'reports': {
        # 记录所有神经元的spikes
        'spikes_all': {
            'file_name': 'output/exp3/spikes_all.h5',
            'module': 'spikes_report',
            'cells': 'all'
        },
        # 记录L23兴奋性神经元的膜电位
        'v_l23e': {
            'file_name': 'output/exp3/v_l23e.h5',
            'module': 'membrane_report',
            'variable_name': 'v',
            'cells': ['e23Cux2'],  # 只记录L23 E
            'sections': 'soma',
            'dt': 0.1  # 每0.1ms记录一次
        },
        # 记录L4各亚型的膜电位（采样部分神经元以节省空间）
        'v_l4_fast': {
            'file_name': 'output/exp3/v_l4_fast.h5',
            'module': 'membrane_report',
            'variable_name': 'v',
            'cells': ['e4Scnn1a'],
            'sections': 'soma',
            'dt': 0.1
        },
        'v_l4_slow': {
            'file_name': 'output/exp3/v_l4_slow.h5',
            'module': 'membrane_report',
            'variable_name': 'v',
            'cells': ['e4Rorb', 'e4other', 'e4Nr5a1'],
            'sections': 'soma',
            'dt': 0.1
        }
    }
}
```

**步骤2：计算群体发放率（PSTH）**

```python
def compute_psth(spike_times, spike_ids, population_ids, bin_size=10, t_max=1000):
    """
    计算群体发放率时程（Population Spike Time Histogram）

    参数:
    - spike_times: 所有spikes的时间 (ms)
    - spike_ids: 所有spikes的神经元ID
    - population_ids: 该群体包含的神经元ID列表
    - bin_size: 时间bin大小 (ms)
    - t_max: 最大时间 (ms)

    返回:
    - time_bins: 时间bin中心
    - psth: 群体发放率 (spikes/s/neuron)
    """
    # 筛选该群体的spikes
    mask = np.isin(spike_ids, population_ids)
    pop_spike_times = spike_times[mask]

    # 计算直方图
    bins = np.arange(0, t_max + bin_size, bin_size)
    counts, edges = np.histogram(pop_spike_times, bins=bins)

    # 转换为发放率 (Hz)
    # counts / (bin_size_sec * n_neurons)
    psth = counts / (bin_size / 1000.0) / len(population_ids)

    # 时间bin中心
    time_bins = (edges[:-1] + edges[1:]) / 2

    return time_bins, psth

# 加载spike数据
with h5py.File('output/exp3/spikes_all.h5', 'r') as f:
    spike_times = f['spikes']['timestamps'][:]
    spike_ids = f['spikes']['node_ids'][:]

# 计算各群体的PSTH
time_bins, psth_e4scnn1a = compute_psth(
    spike_times, spike_ids, e4scnn1a_node_ids, bin_size=10
)
_, psth_e4rorb = compute_psth(
    spike_times, spike_ids,
    [nid for nid in slow_node_ids if node_id_to_type[nid] in e4rorb_type_ids],
    bin_size=10
)
_, psth_e4other = compute_psth(
    spike_times, spike_ids,
    [nid for nid in slow_node_ids if node_id_to_type[nid] in e4other_type_ids],
    bin_size=10
)
_, psth_e4nr5a1 = compute_psth(
    spike_times, spike_ids,
    [nid for nid in slow_node_ids if node_id_to_type[nid] in e4nr5a1_type_ids],
    bin_size=10
)

# 合并慢速亚型
psth_slow_combined = (psth_e4rorb + psth_e4other + psth_e4nr5a1) / 3

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(time_bins, psth_e4scnn1a, label='e4Scnn1a (fast, 100ms)',
        linewidth=2, color='red')
ax.plot(time_bins, psth_slow_combined, label='e4Rorb/other/Nr5a1 (slow, 333ms)',
        linewidth=2, color='blue')
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Population firing rate (Hz)', fontsize=14)
ax.set_title('L4 Subtype Population Dynamics', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.savefig('results/exp3_l4_psth.pdf', dpi=300)
plt.show()
```

**步骤3：测量峰值时间和时间差异**

```python
def find_peak_time(time_bins, psth, t_start=50, t_end=700):
    """
    找到PSTH的峰值时间
    """
    mask = (time_bins >= t_start) & (time_bins <= t_end)
    idx = np.argmax(psth[mask])
    peak_time = time_bins[mask][idx]
    peak_value = psth[mask][idx]
    return peak_time, peak_value

# 计算峰值时间
peak_t_fast, peak_v_fast = find_peak_time(time_bins, psth_e4scnn1a)
peak_t_slow, peak_v_slow = find_peak_time(time_bins, psth_slow_combined)

# 计算时间差异
measured_time_diff = peak_t_slow - peak_t_fast

# 理论预测（基于适应时间常数）
predicted_time_diff = 333 - 100  # = 233 ms

print(f"\n{'='*70}")
print("L4 亚型时间动态测量")
print(f"{'='*70}")
print(f"\ne4Scnn1a (fast):")
print(f"  峰值时间: {peak_t_fast:.1f} ms")
print(f"  峰值发放率: {peak_v_fast:.2f} Hz")

print(f"\ne4Rorb/other/Nr5a1 (slow):")
print(f"  峰值时间: {peak_t_slow:.1f} ms")
print(f"  峰值发放率: {peak_v_slow:.2f} Hz")

print(f"\n时间差异:")
print(f"  实测: {measured_time_diff:.1f} ms")
print(f"  理论预测: {predicted_time_diff:.1f} ms")
print(f"  差值: {abs(measured_time_diff - predicted_time_diff):.1f} ms")

# 判据
if abs(measured_time_diff - predicted_time_diff) < 100:
    print(f"\n✅ 适应参数可以预测实际时间动态")
    print(f"   （实测与预测差异 < 100ms）")
else:
    print(f"\n⚠️ 实测与预测存在较大差异")
    print(f"   可能存在其他因素（如循环、抑制）影响时间差")
```

**步骤4：分析L23膜电位的演化**

```python
def analyze_l23_dynamics(v_data, time_bins):
    """
    分析L23神经元群体膜电位的时间演化

    参数:
    - v_data: shape (n_neurons, n_timebins)
    - time_bins: 时间点

    返回:
    - mean_v: 平均膜电位时程
    - std_v: 标准差
    - correlation_matrix_early: 早期相关性矩阵
    - correlation_matrix_late: 晚期相关性矩阵
    """
    # 计算平均膜电位
    mean_v = v_data.mean(axis=0)
    std_v = v_data.std(axis=0)

    # 定义早期和晚期窗口
    early_window = (time_bins >= 0) & (time_bins <= 200)
    late_window = (time_bins >= 400) & (time_bins <= 700)

    # 计算相关性矩阵
    corr_early = np.corrcoef(v_data[:, early_window])
    corr_late = np.corrcoef(v_data[:, late_window])

    return mean_v, std_v, corr_early, corr_late

# 加载L23膜电位数据
with h5py.File('output/exp3/v_l23e.h5', 'r') as f:
    v_l23e = f['report']['v1']['data'][:]  # shape: (n_neurons, n_timebins)
    time_points = f['report']['v1']['time'][:]

# 分析
mean_v, std_v, corr_early, corr_late = analyze_l23_dynamics(v_l23e, time_points)

# 可视化L23平均膜电位
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 膜电位时程
ax = axes[0, 0]
ax.plot(time_points, mean_v, linewidth=2, color='black')
ax.fill_between(time_points, mean_v - std_v, mean_v + std_v,
                 alpha=0.3, color='gray')
ax.axvspan(0, 200, alpha=0.2, color='red', label='Early window')
ax.axvspan(400, 700, alpha=0.2, color='blue', label='Late window')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Membrane potential (mV)', fontsize=12)
ax.set_title('L23E Population Membrane Potential', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 早期相关性矩阵
ax = axes[0, 1]
im = ax.imshow(corr_early, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_title('Early (0-200ms) Correlation', fontsize=14, fontweight='bold')
ax.set_xlabel('Neuron index', fontsize=12)
ax.set_ylabel('Neuron index', fontsize=12)
plt.colorbar(im, ax=ax)

# 晚期相关性矩阵
ax = axes[1, 0]
im = ax.imshow(corr_late, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_title('Late (400-700ms) Correlation', fontsize=14, fontweight='bold')
ax.set_xlabel('Neuron index', fontsize=12)
ax.set_ylabel('Neuron index', fontsize=12)
plt.colorbar(im, ax=ax)

# 相关性差异
ax = axes[1, 1]
corr_diff = corr_late - corr_early
im = ax.imshow(corr_diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax.set_title('Correlation Change (Late - Early)', fontsize=14, fontweight='bold')
ax.set_xlabel('Neuron index', fontsize=12)
ax.set_ylabel('Neuron index', fontsize=12)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('results/exp3_l23_dynamics.pdf', dpi=300)
plt.show()

# 量化相关性差异
mean_corr_early = corr_early[np.triu_indices_from(corr_early, k=1)].mean()
mean_corr_late = corr_late[np.triu_indices_from(corr_late, k=1)].mean()

print(f"\nL23神经元群体相关性:")
print(f"  早期 (0-200ms): {mean_corr_early:.3f}")
print(f"  晚期 (400-700ms): {mean_corr_late:.3f}")
print(f"  变化: {mean_corr_late - mean_corr_early:.3f}")

if abs(mean_corr_late - mean_corr_early) > 0.05:
    print(f"\n✅ L23表征在时间上显著演化")
else:
    print(f"\n⚠️ L23表征相对稳定，演化不明显")
```

#### 预期结果

**如果假设正确**：

1. **L4 PSTH峰值时间**：
   - e4Scnn1a: ~100-150ms
   - e4Rorb/other/Nr5a1: ~350-450ms
   - **实测时间差**: 200-350ms（接近233ms理论预测）

2. **L23膜电位演化**：
   - 早期阶段（0-200ms）：快速去极化，由e4Scnn1a驱动
   - 晚期阶段（400-700ms）：持续整合，由慢速亚型驱动
   - 早期vs晚期相关性模式显著不同

3. **验证适应参数预测能力**：
   - |实测时间差 - 233ms| < 100ms → 适应参数可预测时间动态
   - |实测时间差 - 233ms| > 150ms → 存在其他重要因素

---

### 实验4：时间泛化解码

#### 实验目的

验证L23表征是否真的在时间上演化，还是静态表征（如果是静态的，解码器应该跨时间泛化）。

#### 详细操作步骤

```python
def temporal_generalization_matrix(spike_data, labels, time_windows):
    """
    计算时间泛化矩阵

    参数:
    - spike_data: shape (n_trials, n_neurons, n_timebins)
    - labels: 类别标签
    - time_windows: list of (start_ms, end_ms)

    返回:
    - tgm: Temporal Generalization Matrix, shape (n_windows, n_windows)
          tgm[i, j] = 在时间窗口i训练的解码器在时间窗口j测试的准确率
    """
    n_windows = len(time_windows)
    tgm = np.zeros((n_windows, n_windows))

    for i, (train_start, train_end) in enumerate(time_windows):
        # 提取训练窗口的发放率
        train_start_bin = int(train_start / 0.1)
        train_end_bin = int(train_end / 0.1)
        X_train = spike_data[:, :, train_start_bin:train_end_bin].mean(axis=2)

        # 训练解码器
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf.fit(X_train, labels)

        for j, (test_start, test_end) in enumerate(time_windows):
            # 提取测试窗口的发放率
            test_start_bin = int(test_start / 0.1)
            test_end_bin = int(test_end / 0.1)
            X_test = spike_data[:, :, test_start_bin:test_end_bin].mean(axis=2)

            # 测试
            tgm[i, j] = clf.score(X_test, labels)

    return tgm

# 定义时间窗口（100ms窗口，50ms步长）
time_windows = [(t, t+100) for t in range(0, 700, 50)]
time_labels = [f'{t}-{t+100}' for t, _ in time_windows]

# 计算时间泛化矩阵（精细分类）
tgm_fine = temporal_generalization_matrix(
    spike_data, labels_fine, time_windows
)

# 计算时间泛化矩阵（抽象分类）
tgm_abstract = temporal_generalization_matrix(
    spike_data, labels_abstract, time_windows
)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 精细分类
ax = axes[0]
im = ax.imshow(tgm_fine, cmap='RdBu_r', origin='lower', aspect='auto',
               vmin=0, vmax=max(tgm_fine.max(), tgm_abstract.max()))
ax.set_xticks(range(len(time_labels)))
ax.set_yticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(time_labels, fontsize=8)
ax.set_xlabel('Testing Time Window (ms)', fontsize=12)
ax.set_ylabel('Training Time Window (ms)', fontsize=12)
ax.set_title('Fine-grained (40-way) Temporal Generalization',
             fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Decoding Accuracy')

# 添加对角线
ax.plot([0, len(time_labels)-1], [0, len(time_labels)-1],
        'k--', linewidth=2, alpha=0.5, label='Diagonal')

# 抽象分类
ax = axes[1]
im = ax.imshow(tgm_abstract, cmap='RdBu_r', origin='lower', aspect='auto',
               vmin=0, vmax=max(tgm_fine.max(), tgm_abstract.max()))
ax.set_xticks(range(len(time_labels)))
ax.set_yticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(time_labels, fontsize=8)
ax.set_xlabel('Testing Time Window (ms)', fontsize=12)
ax.set_ylabel('Training Time Window (ms)', fontsize=12)
ax.set_title('Abstract (2-way) Temporal Generalization',
             fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Decoding Accuracy')

# 添加对角线
ax.plot([0, len(time_labels)-1], [0, len(time_labels)-1],
        'k--', linewidth=2, alpha=0.5, label='Diagonal')

plt.tight_layout()
plt.savefig('results/exp4_temporal_generalization.pdf', dpi=300)
plt.show()
```

#### 预期结果

**如果假设正确**（L23表征时间演化）：

**精细分类TGM**：
```
训练/测试    0-100  100-200  200-300  300-400  400-500  500-600  600-700
0-100         低      低       低       低       低       低       低
100-200       低      高       高       中       低       低       低    ← 早期训练
200-300       低      高       高       中       低       低       低       的解码器
300-400       低      中       中       中       低       低       低       只在早期
400-500       低      低       低       低       低       低       低       有效
500-600       低      低       低       低       低       低       低
600-700       低      低       低       低       低       低       低
```

**抽象分类TGM**：
```
训练/测试    0-100  100-200  200-300  300-400  400-500  500-600  600-700
0-100         低      低       低       低       低       低       低
100-200       低      低       低       低       低       低       低
200-300       低      低       低       低       低       低       低
300-400       低      低       低       中       中       低       低
400-500       低      低       低       中       中       高       高    ← 晚期训练
500-600       低      低       低       低       中       高       高       的解码器
600-700       低      低       低       低       中       高       高       只在晚期有效
```

**量化指标**：

```python
def analyze_temporal_generalization(tgm, time_windows):
    """
    分析时间泛化矩阵
    """
    n = tgm.shape[0]

    # 对角线：同一时间窗口的准确率
    diagonal = np.diag(tgm)

    # 非对角线：跨时间泛化
    off_diagonal_upper = np.triu(tgm, k=1).flatten()
    off_diagonal_lower = np.tril(tgm, k=-1).flatten()
    off_diagonal = np.concatenate([off_diagonal_upper, off_diagonal_lower])
    off_diagonal = off_diagonal[off_diagonal > 0]  # 移除零值

    # 计算泛化指数
    # 如果表征稳定（静态），对角线和非对角线应该相近
    # 如果表征演化，对角线应该远高于非对角线
    generalization_index = diagonal.mean() - off_diagonal.mean()

    print(f"对角线平均准确率: {diagonal.mean():.3f}")
    print(f"非对角线平均准确率: {off_diagonal.mean():.3f}")
    print(f"泛化指数 (对角线 - 非对角线): {generalization_index:.3f}")

    if generalization_index > 0.15:
        print("  → 表征显著演化（对角线明显高于非对角线）")
        return "DYNAMIC"
    elif generalization_index < 0.05:
        print("  → 表征相对稳定（对角线和非对角线相近）")
        return "STATIC"
    else:
        print("  → 表征部分演化")
        return "PARTIAL"

print("\n精细分类 (40-way):")
result_fine = analyze_temporal_generalization(tgm_fine, time_windows)

print("\n抽象分类 (2-way):")
result_abstract = analyze_temporal_generalization(tgm_abstract, time_windows)

# 综合判断
if result_fine == "DYNAMIC" and result_abstract == "DYNAMIC":
    print(f"\n✅ L23表征在两种分类中都显著演化")
    print(f"   支持\"L23在不同时间整合不同L4输入\"的假设")
elif result_fine == "STATIC" and result_abstract == "STATIC":
    print(f"\n❌ L23表征相对静态")
    print(f"   需要重新考虑机制")
```

**预期量化结果**：

| 分类任务 | 对角线准确率 | 非对角线准确率 | 泛化指数 | 结论 |
|---------|------------|--------------|---------|------|
| 精细（40类） | 0.28 | 0.10 | 0.18 | 显著演化 |
| 抽象（2类） | 0.82 | 0.58 | 0.24 | 显著演化 |

**如果假设错误**（L23表征静态）：

| 分类任务 | 对角线准确率 | 非对角线准确率 | 泛化指数 | 结论 |
|---------|------------|--------------|---------|------|
| 精细（40类） | 0.28 | 0.25 | 0.03 | 相对稳定 |
| 抽象（2类） | 0.82 | 0.78 | 0.04 | 相对稳定 |

---

### 实验5：放大L23 E→E测试累积效应

#### 实验目的

测试虽然单突触弱（19%）但连接数极多（10M+）的L23 E→E是否对抽象分类关键。

#### 详细操作步骤

**步骤1：识别L23 E→E连接**

```python
# L23 E→E连接
mask_l23e_to_l23e = (
    np.isin(source_node_id, l23_exc_node_ids) &
    np.isin(target_node_id, l23_exc_node_ids)
)

print(f"Found {mask_l23e_to_l23e.sum()} L23 E→E connections")
```

**步骤2：放大/减弱连接**

```python
# 测试多个放大/减弱因子
amplification_factors = [0.0, 0.5, 1.0, 2.0, 5.0]
# 0.0 = 完全消除横向连接
# 0.5 = 减弱到9.5%
# 1.0 = 基线（19%）
# 2.0 = 放大到38%
# 5.0 = 放大到95%（接近L4水平的100%）

for factor in amplification_factors:
    print(f"\n{'='*70}")
    print(f"Testing amplification factor: {factor}")
    print(f"{'='*70}")

    with h5py.File('network/v1_v1_edges.h5', 'r+') as f:
        edges_group = f['edges']['v1_to_v1']
        edge_type_id = edges_group['edge_type_id'][:]

        unique_edge_types = np.unique(edge_type_id[mask_l23e_to_l23e])

        for edge_type in unique_edge_types:
            edge_type_str = str(edge_type)
            if edge_type_str in edges_group:
                syn_weight = edges_group[edge_type_str]['syn_weight']

                edge_type_mask = (edge_type_id == edge_type)
                combined_mask = mask_l23e_to_l23e & edge_type_mask

                original_weights = syn_weight[combined_mask]
                syn_weight[combined_mask] = original_weights * factor

                print(f"  Edge type {edge_type}: {combined_mask.sum()} connections")
                print(f"    Mean weight: {original_weights.mean():.4f} → "
                      f"{(original_weights * factor).mean():.4f}")

    # 运行模拟和解码（代码与实验1类似）
    # ...
```

#### 预期结果

**如果L23 E→E对抽象分类关键**：

| 放大因子 | L23 E→E强度 | 精细分类峰值时间 | 抽象分类峰值时间 | 时间间距Δt |
|---------|-----------|-----------------|-----------------|-----------|
| 0.0（消除） | 0% | ~150-200ms (不变) | ~700-900ms (+200ms) | ~600ms (+100ms) |
| 0.5（减弱） | 9.5% | ~150-200ms (不变) | ~550-750ms (+50ms) | ~550ms (+50ms) |
| 1.0（基线） | 19% | ~150-200ms | ~500-700ms | ~500ms |
| 2.0（放大） | 38% | ~150-200ms (不变) | ~400-600ms (-100ms) | ~400ms (-100ms) |
| 5.0（大幅放大） | 95% | ~150-200ms (不变) | ~350-500ms (-150ms) | ~300ms (-200ms) |

**关键观察**：
- ✅ **抽象分类加快**：随L23 E→E增强，峰值时间提前
- ✅ **精细分类不变**：峰值时间基本保持
- ✅ **时间间距减小**：从500ms→300ms
- ✅ **剂量-效应关系**：放大倍数越大，效应越强

**判据**：

```python
def test_l23_lateral_effect(results_exp5):
    """
    测试L23横向连接的效应
    """
    # 提取抽象分类峰值时间
    peak_times = {}
    for factor in amplification_factors:
        peak_times[factor] = time_centers[np.argmax(results_exp5[factor]['abstract'])]

    # 计算效应大小
    baseline_peak = peak_times[1.0]
    amplified_peak = peak_times[5.0]
    reduced_peak = peak_times[0.5]

    effect_amplify = baseline_peak - amplified_peak  # 应该>0（提前）
    effect_reduce = reduced_peak - baseline_peak    # 应该>0（延迟）

    print(f"\n{'='*70}")
    print("L23 E→E效应测试")
    print(f"{'='*70}")
    print(f"\n抽象分类峰值时间:")
    for factor in amplification_factors:
        print(f"  Factor {factor}: {peak_times[factor]:.0f} ms")

    print(f"\n效应大小:")
    print(f"  放大5倍 → 提前 {effect_amplify:.0f} ms")
    print(f"  减弱0.5倍 → 延迟 {effect_reduce:.0f} ms")

    # 判据
    if effect_amplify > 100 and effect_reduce > 50:
        print(f"\n✅ L23 E→E虽弱但对抽象分类关键")
        print(f"   多轮循环的累积效应显著")
        return "IMPORTANT"
    elif effect_amplify < 20 and effect_reduce < 20:
        print(f"\n❌ L23 E→E太弱，可以忽略")
        print(f"   即使放大5倍仍无显著效应")
        return "NEGLIGIBLE"
    else:
        print(f"\n⚠️ L23 E→E有中等效应")
        return "MODERATE"

result_l23_test = test_l23_lateral_effect(results_exp5_all)
```

**如果L23 E→E不重要**：

| 放大因子 | 抽象分类峰值时间 | 变化 |
|---------|-----------------|------|
| 0.0 | ~500-700ms | 0ms |
| 0.5 | ~500-700ms | 0ms |
| 1.0 | ~500-700ms | - |
| 2.0 | ~500-700ms | 0ms |
| 5.0 | ~500-700ms | 0ms |

→ 说明L23 E→E确实太弱，累积效应可忽略，机制完全在L4内部

---

### 实验6：放大L5→L23测试选择性

#### 实验目的

测试L5→L23（仅6%强度）是否有"弱但选择性"的作用。

#### 详细操作步骤

```python
# 识别L5 E→L23 E连接
l5_exc_type_ids = node_types[node_types['pop_name'].isin(['e5Rbp4', 'e5noRbp4'])]['node_type_id'].values
l5_exc_node_ids = [nid for nid, ntype in node_id_to_type.items()
                    if ntype in l5_exc_type_ids]

mask_l5e_to_l23e = (
    np.isin(source_node_id, l5_exc_node_ids) &
    np.isin(target_node_id, l23_exc_node_ids)
)

print(f"Found {mask_l5e_to_l23e.sum()} L5 E→L23 E connections")

# 由于基线只有6%，必须大幅放大才能看到效应
amplification_factors_l5 = [0.0, 1.0, 5.0, 10.0, 20.0]
# 0.0 = 完全消除
# 1.0 = 基线（6%）
# 5.0 = 放大到30%
# 10.0 = 放大到60%
# 20.0 = 放大到120%（超过L4）

# 修改权重并运行模拟（代码与实验5类似）
# ...
```

#### 预期结果

**如果L5→L23确实重要（虽弱但选择性）**：

| 放大因子 | L5→L23强度 | 精细分类 | 抽象分类峰值时间 | 抽象分类峰值准确率 |
|---------|----------|---------|-----------------|-------------------|
| 0.0（消除） | 0% | 不变 | ~550-750ms (+50ms) | 0.78 (-7%) |
| 1.0（基线） | 6% | 不变 | ~500-700ms | 0.85 |
| 5.0 | 30% | 不变 | ~450-650ms (-50ms) | 0.88 (+3%) |
| 10.0 | 60% | 不变 | ~400-600ms (-100ms) | 0.90 (+5%) |
| 20.0 | 120% | 轻微受损 | ~350-550ms (-150ms) | 0.92 (+7%) |

**如果L5→L23真的不重要**：

| 放大因子 | 抽象分类峰值时间 | 变化 |
|---------|-----------------|------|
| 0.0 | ~500-700ms | 0ms |
| 5.0 | ~500-700ms | 0ms |
| 10.0 | ~500-700ms | 0ms |
| 20.0 | ~500-700ms | 0ms |

→ 说明L5→L23太弱，可以忽略

**判据**：

```python
if (放大5倍 → 抽象提前 > 50ms):
    结论 = "L5虽弱但可能有选择性作用"
    进一步分析：哪些L23神经元接收较多L5输入？
elif (放大10倍仍无效应):
    结论 = "L5→L23太弱，可以忽略"
    机制可能完全在L4内部
```

---

### 实验7：选择性调节抑制子型

#### 实验目的

测试抑制子型（PV, Sst, Htr3a）是否有早期vs晚期的功能分工。

#### 详细操作步骤

**测试1：减弱PV→L23 E**

```python
# 识别所有层的PV神经元
pv_pops = ['i23Pvalb', 'i4Pvalb', 'i5Pvalb', 'i6Pvalb']
pv_type_ids = node_types[node_types['pop_name'].isin(pv_pops)]['node_type_id'].values
pv_node_ids = [nid for nid, ntype in node_id_to_type.items()
                if ntype in pv_type_ids]

mask_pv_to_l23e = (
    np.isin(source_node_id, pv_node_ids) &
    np.isin(target_node_id, l23_exc_node_ids)
)

print(f"Found {mask_pv_to_l23e.sum()} PV→L23E connections")

# 减弱PV→L23E连接
weakening_factors = [1.0, 0.5, 0.2, 0.0]
# ...运行模拟和解码
```

**测试2：减弱Sst→L23 E**

```python
# 识别所有层的Sst神经元
sst_pops = ['i23Sst', 'i4Sst', 'i5Sst', 'i6Sst']
sst_type_ids = node_types[node_types['pop_name'].isin(sst_pops)]['node_type_id'].values
sst_node_ids = [nid for nid, ntype in node_id_to_type.items()
                 if ntype in sst_type_ids]

mask_sst_to_l23e = (
    np.isin(source_node_id, sst_node_ids) &
    np.isin(target_node_id, l23_exc_node_ids)
)

# 减弱Sst→L23E连接
# ...运行模拟和解码
```

**测试3：减弱Htr3a→L23 E（对照）**

```python
# 识别所有层的Htr3a神经元
htr3a_pops = ['i23Htr3a', 'i4Htr3a', 'i5Htr3a', 'i6Htr3a']
# ...类似操作
```

#### 预期结果

**如果抑制子型有时间分工**：

**减弱PV→L23E（factor=0.5）**：

| 分类任务 | 峰值时间变化 | 准确率变化 | 主要影响窗口 |
|---------|------------|-----------|------------|
| 精细（40类） | +50-100ms | -15% | 0-300ms（早期） |
| 抽象（2类） | +0-20ms | -3% | 影响小 |

**减弱Sst→L23E（factor=0.5）**：

| 分类任务 | 峰值时间变化 | 准确率变化 | 主要影响窗口 |
|---------|------------|-----------|------------|
| 精细（40类） | +0-20ms | -5% | 影响小 |
| 抽象（2类） | +100-150ms 或提前 | -12% | 400-700ms（晚期） |

**判据**：

```python
def test_inhibitory_selectivity(results_pv, results_sst):
    """
    测试抑制子型的时间选择性
    """
    # 计算精细vs抽象受损程度

    # 减弱PV的效应
    fine_damage_pv = (results_pv[1.0]['fine'].max() -
                      results_pv[0.5]['fine'].max())
    abstract_damage_pv = (results_pv[1.0]['abstract'].max() -
                          results_pv[0.5]['abstract'].max())

    # 减弱Sst的效应
    fine_damage_sst = (results_sst[1.0]['fine'].max() -
                       results_sst[0.5]['fine'].max())
    abstract_damage_sst = (results_sst[1.0]['abstract'].max() -
                           results_sst[0.5]['abstract'].max())

    # 选择性指数
    selectivity_pv = (fine_damage_pv - abstract_damage_pv) / \
                     (fine_damage_pv + abstract_damage_pv + 1e-6)
    selectivity_sst = (abstract_damage_sst - fine_damage_sst) / \
                      (abstract_damage_sst + fine_damage_sst + 1e-6)

    print(f"\n{'='*70}")
    print("抑制子型选择性测试")
    print(f"{'='*70}")
    print(f"\n减弱PV→L23E:")
    print(f"  精细分类受损: {fine_damage_pv:.3f}")
    print(f"  抽象分类受损: {abstract_damage_pv:.3f}")
    print(f"  选择性指数: {selectivity_pv:.3f} (>0表示对精细选择性)")

    print(f"\n减弱Sst→L23E:")
    print(f"  精细分类受损: {fine_damage_sst:.3f}")
    print(f"  抽象分类受损: {abstract_damage_sst:.3f}")
    print(f"  选择性指数: {selectivity_sst:.3f} (>0表示对抽象选择性)")

    if selectivity_pv > 0.3 and selectivity_sst > 0.3:
        print(f"\n✅ 抑制子型有时间分工")
        print(f"   PV主要影响精细分类（早期）")
        print(f"   Sst主要影响抽象分类（晚期）")
        return "TIME_DIVISION"
    elif abs(selectivity_pv) < 0.1 and abs(selectivity_sst) < 0.1:
        print(f"\n❌ 抑制子型无选择性")
        print(f"   都是全局增益控制，无时间分工")
        return "NO_SELECTIVITY"
    else:
        print(f"\n⚠️ 结果不明确")
        return "UNCLEAR"

result_inh = test_inhibitory_selectivity(results_pv_all, results_sst_all)
```

---

## 辅助假设与机制

### 辅助假设1：L23横向整合累积效应

#### 数据基础

- L23 E→E连接数：10,434,976（最多！）
- 单突触权重：0.390（仅为L4→L23的14%）
- 总强度：4,064,535（L4→L23的19%）

#### 假设描述

虽然单突触弱，但通过多轮循环，L23 E→E的累积效应可能对抽象分类重要：

```
早期（0-200ms）：
  L4快速亚型 → L23
  L23 E→E横向整合弱（单轮效应小）
  ↓
  L23主要反映L4快速输入
  ↓
  精细分类可解码

晚期（400-700ms）：
  L4慢速亚型 → L23
  L23 E→E多轮循环累积
  ↓
  横向整合帮助形成抽象表征
  ↓
  抽象分类可解码
```

#### 验证方式

实验5：放大L23 E→E，观察抽象分类是否加快

---

### 辅助假设2：L5反馈的"弱但选择性"作用

#### 数据基础

- L5 E→L23 E总强度：1,228,763（仅6%）
- 连接数：820,060
- 平均权重：1.503

#### 假设描述

L5虽然只有6%强度，但可能高度选择性：

```
"弱但选择性"模型：

L4 → L23：提供"基础材料"（100%强度）
  ↓
L23早期：包含丰富信息，但未组织

L5 → L23：提供"关键指令"（仅6%，但选择性强）
  ↓
L5选择性激活特定L23神经元亚群
  ↓
这些神经元编码抽象特征
  ↓
L23晚期：抽象表征形成
```

#### 验证方式

实验6：大幅放大L5→L23（×5或×10），观察是否有选择性效应

---

### 辅助假设3：抑制子型的时序门控

#### 数据基础

| 抑制子型 | 到L23E延迟 | 总强度 |
|---------|-----------|--------|
| PV | 0.90 ms | 1,780,612 (8.5%) |
| Sst | 1.50 ms | 1,393,101 (6.7%) |
| Htr3a | 1.96 ms | 2,548,239 (12.2%) |

#### 假设描述

```
早期（0-200ms）：
  PV（0.9ms，最快）→ L23
  ↓
  快速抑制"雕刻"早期响应
  ↓
  精细区分

晚期（400-700ms）：
  Sst（1.5ms）→ L23
  ↓
  调节整合窗口
  ↓
  抽象分类
```

**注意**：延迟差异很小（仅1ms），可能只是辅助机制

#### 验证方式

实验7：选择性减弱PV/Sst，观察时间选择性

---

## 总结与验证路线图

### 假设可靠性评估

| 假设 | 数据支撑 | 文献支撑 | 可靠性 | 验证实验 | 优先级 |
|------|---------|---------|--------|---------|--------|
| **L4亚型时间分离** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 实验1-4 | ⭐⭐⭐⭐⭐ |
| L23横向整合累积 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 实验5 | ⭐⭐⭐ |
| L5反馈弱但选择性 | ⭐⭐ | ⭐⭐ | ⭐⭐ | 实验6 | ⭐⭐ |
| 抑制子型时序门控 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 实验7 | ⭐⭐⭐ |

### 推荐测试顺序

**阶段1：验证核心假设（最高优先级）**
1. **实验1**：减弱e4Scnn1a → 验证快速亚型支持精细分类
2. **实验2**：减弱慢速亚型 → 验证慢速亚型支持抽象分类
3. **实验3**：实测PSTH → 验证适应参数预测时间动态
4. **实验4**：时间泛化矩阵 → 验证L23表征时间演化

**判据**：如果实验1-4都成功 → L4亚型时间分离是主要机制 ✓

**阶段2：探索辅助机制（次优先级）**
5. **实验5**：放大L23 E→E → 测试横向整合累积效应
6. **实验7**：调节PV/Sst → 测试抑制子型时间分工

**判据**：
- 如果实验5成功 → L23横向整合是辅助机制
- 如果实验7成功 → 抑制子型有时间分工

**阶段3：测试最弱假设（可选）**
7. **实验6**：大幅放大L5→L23 → 测试L5是否有选择性

**判据**：
- 如果实验6失败 → L5可以忽略（最可能）
- 如果实验6成功 → L5有弱但选择性作用（意外发现）

### 关键判据总结

**强烈支持L4亚型时间分离假设的标准**：

```python
# 必须同时满足以下条件：

1. 实验1（减弱e4Scnn1a）：
   - 精细分类峰值延迟 > 100ms
   - 抽象分类延迟 < 50ms
   - 选择性指数 > 0.3

2. 实验2（减弱慢速亚型）：
   - 抽象分类峰值延迟 > 100ms
   - 精细分类延迟 < 50ms
   - 选择性指数 > 0.3

3. 实验3（实测PSTH）：
   - e4Scnn1a峰值时间：100-150ms
   - 慢速亚型峰值时间：350-450ms
   - |实测时间差 - 233ms| < 100ms

4. 实验4（时间泛化）：
   - 精细分类泛化指数 > 0.15（演化）
   - 抽象分类泛化指数 > 0.15（演化）

如果以上4个条件都满足：
   → 强烈支持L4亚型时间分离是主要机制
   → 233ms适应差异是500ms时间滞后的核心来源
```

### 可能的结果场景

**场景1：完全支持主假设** ⭐⭐⭐⭐⭐
- 实验1-4全部成功
- 实验5-7可能有辅助效应
- 结论：L4亚型时间分离是主要机制，其他是辅助

**场景2：部分支持主假设** ⭐⭐⭐⭐
- 实验1-4有选择性效应，但不如预期强
- 实验5（L23 E→E）也有显著效应
- 结论：L4亚型 + L23横向整合共同作用

**场景3：主假设被证伪** ⭐⭐
- 实验1-2无选择性（两种分类都受损或都不受损）
- 实验3实测时间差与预测不符
- 需要重新考虑机制，可能是：
  - L23内部机制（实验5）
  - 抑制网络动态（实验7）
  - 完全不同的机制

**场景4：意外发现** ⭐⭐⭐
- 主假设不成立
- 但实验6（L5反馈）有强效应
- 说明L5虽弱但有选择性作用（新发现）

---

## 附录：代码工具库

### 工具1：批量运行实验

```python
def run_experiment_batch(exp_name, factors, config_template):
    """
    批量运行实验的不同条件
    """
    results = {}

    for factor in factors:
        print(f"\n{'='*70}")
        print(f"Running {exp_name} with factor={factor}")
        print(f"{'='*70}")

        # 修改配置
        config = config_template.copy()
        config['output']['output_dir'] = f'output/{exp_name}_factor{factor}'

        # 修改连接权重
        modify_connections(exp_name, factor)

        # 运行模拟
        pointnet.run_pointnet(config)

        # 加载结果
        spike_data = load_spike_data(config['output']['output_dir'])

        # 解码
        acc_fine, acc_abstract = decode_categories(spike_data, ...)

        results[factor] = {
            'fine': acc_fine,
            'abstract': acc_abstract
        }

    return results
```

### 工具2：可视化比较

```python
def plot_experiment_comparison(results_dict, exp_name, save_path):
    """
    可视化不同条件的解码结果比较
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 精细分类时间序列
    ax = axes[0, 0]
    for factor, results in results_dict.items():
        ax.plot(time_centers, results['fine'], label=f'Factor={factor}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title(f'{exp_name}: Fine-grained (40-way)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 抽象分类时间序列
    ax = axes[0, 1]
    for factor, results in results_dict.items():
        ax.plot(time_centers, results['abstract'], label=f'Factor={factor}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title(f'{exp_name}: Abstract (2-way)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 峰值时间比较
    ax = axes[1, 0]
    factors = list(results_dict.keys())
    peak_times_fine = [time_centers[np.argmax(results_dict[f]['fine'])]
                       for f in factors]
    peak_times_abstract = [time_centers[np.argmax(results_dict[f]['abstract'])]
                           for f in factors]

    x = np.arange(len(factors))
    width = 0.35
    ax.bar(x - width/2, peak_times_fine, width, label='Fine (40-way)')
    ax.bar(x + width/2, peak_times_abstract, width, label='Abstract (2-way)')
    ax.set_xlabel('Factor')
    ax.set_ylabel('Peak Time (ms)')
    ax.set_title('Peak Decoding Time', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in factors])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 时间间距比较
    ax = axes[1, 1]
    delta_t = [peak_times_abstract[i] - peak_times_fine[i]
               for i in range(len(factors))]
    ax.plot(factors, delta_t, 'o-', linewidth=2, markersize=10)
    ax.set_xlabel('Factor')
    ax.set_ylabel('Temporal Gap Δt (ms)')
    ax.set_title('Fine vs Abstract Temporal Gap', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
```

### 工具3：统计显著性检验

```python
from scipy.stats import ttest_rel, wilcoxon

def statistical_test(baseline_scores, modified_scores):
    """
    对基线和修改条件进行统计检验
    """
    # 配对t检验
    t_stat, p_value_t = ttest_rel(baseline_scores, modified_scores)

    # Wilcoxon符号秩检验（非参数）
    w_stat, p_value_w = wilcoxon(baseline_scores, modified_scores)

    print(f"配对t检验: t={t_stat:.3f}, p={p_value_t:.4f}")
    print(f"Wilcoxon检验: W={w_stat:.3f}, p={p_value_w:.4f}")

    if p_value_t < 0.01:
        print("✅ 差异极显著 (p < 0.01)")
    elif p_value_t < 0.05:
        print("✅ 差异显著 (p < 0.05)")
    else:
        print("❌ 差异不显著 (p >= 0.05)")

    return t_stat, p_value_t, w_stat, p_value_w
```

---

**文档完成时间**：2025-12-26
**基于数据**：Allen Institute V1 Point Neuron Model (AlphaBrain2.0.1-alpha)
**核心假设**：L4亚型时间分离机制（233ms适应差异）

**下一步**：运行实验1-4验证核心假设
