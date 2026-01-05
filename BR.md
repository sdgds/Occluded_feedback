# Binocular Rivalry模型中的自适应增益调节机制

## 1. 概述

本文档详细说明了双眼竞争（Binocular Rivalry）模拟中使用的 `changed_beta_func` 函数的数学原理、物理意义和实现细节。该函数通过动态调节神经增益参数β，实现Face区域与Place区域之间的竞争性感知切换。

---

## 2. 核心物理原理

### 2.1 随机Hopfield网络中的β参数

在随机Hopfield神经网络中，神经元的激活概率由以下sigmoid函数决定：

```
P(s_i = +1 | h_i) = 1 / (1 + exp(-β · h_i))
```

其中：
- `s_i`：第i个神经元的状态（+1或-1）
- `h_i`：局部场强（来自其他神经元的输入）
- `β`：**逆温度参数**（inverse temperature），控制系统的确定性

### 2.2 β参数的物理意义

| β值 | 物理意义 | 神经动力学效果 |
|-----|---------|---------------|
| **高β** (≈15) | 高噪声、低增益 | 神经元状态不稳定，难以维持激活 → **抑制该区域** |
| **低β** (≈2) | 低噪声、高增益 | 神经元状态确定，容易维持激活 → **增强该区域** |

### 2.3 竞争机制

Face和Place两个区域通过**反相调节各自的β值**实现对抗性竞争：
- 当Face主导时：`β_Face ≈ 15`（高），`β_Place ≈ 2`（低）
- 当Place主导时：`β_Place ≈ 15`（高），`β_Face ≈ 2`（低）

---

## 3. 函数签名与参数

```python
def changed_beta_func(self, region_avg_dynamics_state, mask, beta, tao, adaptation_lower_bound=2):
    """
    Face vs Place竞争版本：通过自适应增益调节实现感知切换
    """
```

### 3.1 输入参数

| 参数名 | 类型 | 物理意义 | 典型值 |
|--------|------|---------|--------|
| `self` | object | Hopfield网络实例 | - |
| `region_avg_dynamics_state` | ndarray | 当前区域的激活历史时间序列 | shape: (n_timesteps,) |
| `mask` | ndarray | 当前神经元所属的区域掩码 | face_mask/place_mask/... |
| `beta` | float | 基础β值（本函数中未使用） | 15 |
| `tao` | int | 统计时间窗口大小 | 1 |
| `adaptation_lower_bound` | float | β的最小值（对应最大增益） | 2 |

### 3.2 返回值

| 返回类型 | 说明 |
|---------|------|
| float | 为当前区域当前时刻计算得到的β值 |

---

## 4. 核心计算流程

### 4.1 总体架构

```
输入: region_avg_dynamics_state, mask
    ↓
计算统计量: r (平均激活率), r_1 (一阶差分)
    ↓
判断系统状态 (|r_1| < threshold ?)
    ↓
├─ 稳态 → beta_r 策略
│          ├─ 高激活 → 神经适应 (增加β)
│          └─ 低激活 → 对手抑制 (根据opponent决定β)
│
└─ 过渡态 → beta_first_order_difference 策略
           ├─ 上升 → 施加制动 (高β)
           └─ 下降 → 加速切换 (低β)
    ↓
返回 β 值
```

---

### 4.2 第一步：计算激活统计量

#### 4.2.1 平均激活率 r

**数学公式：**
```
r = (1/τ) · Σ[i=1 to τ] firing_rate(t-i)
```

**代码实现：**
```python
r = np.zeros(tao)
for i in range(1, tao + 1):
    r[i - 1] = get_firing_rate(region_avg_dynamics_state, -i)
r = np.mean(r)  # 过去τ个时间步的平均激活
```

**物理意义：**
- 衡量该区域当前是否处于高激活状态（主导感知）
- `r > 0.78`：该区域主导感知（例如正在"看到"Face）
- `r < 0.78`：该区域被抑制（例如正在"看到"Place）

---

#### 4.2.2 一阶时间差分 r₁

**数学公式：**
```
r_1 = (1/τ) · Σ[i=1 to τ] [firing_rate(t-i) - firing_rate(t-i-1)]
```

**代码实现：**
```python
r_1 = np.zeros(tao)
for i in range(1, tao + 1):
    r_1[i - 1] = first_order_difference(region_avg_dynamics_state, -i, -(i + 1))
r_1 = np.mean(r_1)  # 平均变化速率
```

**物理意义：**
- 检测该区域激活是否正在快速变化
- `r_1 > 0.03`：激活正在上升（正在获得主导）
- `r_1 < -0.03`：激活正在下降（正在失去主导）
- `|r_1| < 0.03`：激活稳定（处于稳态）

---

### 4.3 第二步：决策树选择调节策略

```python
decision_threshold = 0.03 + np.random.normal(0, 0.018)

if np.abs(r_1) < decision_threshold:
    return beta_r(r)  # 稳态调节
else:
    return beta_first_order_difference(r_1)  # 过渡态调节
```

**判断逻辑：**

| 条件 | 策略 | 物理意义 |
|------|------|---------|
| `|r_1| < 0.03 ± 0.018` | `beta_r(r)` | 系统稳定，基于激活率调节 |
| `|r_1| ≥ 0.03 ± 0.018` | `beta_first_order_difference(r_1)` | 系统过渡，基于变化率调节 |

---

## 5. 策略1：稳态调节 (beta_r)

当系统处于稳态时（`|r_1| < 0.03`），使用激活率驱动的β调节。

### 5.1 分支A：高激活自适应（Neural Adaptation）

**触发条件：**
```
r > 0.78 + N(0, 0.09)
```
其中 `N(μ, σ)` 表示均值为μ、标准差为σ的正态分布。

**计算公式：**
```
β(r) = [(15 - 2) / (1 + exp(25·(r - 0.5)))] + 2 + N(0, 1.1)
```

**代码实现：**
```python
adaptation_threshold = 0.78 + np.random.normal(0, 0.09)

if r > adaptation_threshold:
    base_beta = (15 - adaptation_lower_bound) / (1 + np.exp(25 * (r - 0.5))) + adaptation_lower_bound
    return base_beta + np.random.normal(0, 1.1)
```

**物理意义：**
- 模拟**神经适应（Neural Adaptation）**现象
- 当区域长时间主导感知时，逐渐增加β（增加噪声），削弱响应
- 对应生物学中的**感受器疲劳**（Sensory Fatigue）

**Sigmoid函数特性：**

| r值 | β值 | 意义 |
|-----|-----|------|
| 0.4 | ≈ 2.0 | 最小噪声，强激活 |
| 0.5 | ≈ 8.5 | 中等噪声 |
| 0.9 | ≈ 15.0 | 最大噪声，弱激活 |

---

### 5.2 分支B：对手抑制机制（Opponent Inhibition）

**触发条件：**
```
r ≤ 0.78 + N(0, 0.09)
```

#### 5.2.1 计算对手区域激活

```python
# 确定对手区域
if (mask == face_mask).all():
    opponent_mask = place_mask  # Face的对手是Place
elif (mask == place_mask).all():
    opponent_mask = face_mask   # Place的对手是Face

# 计算对手的平均激活
opponent_region_avg_dynamics_state = self.avg_activation_in_mask_timeserise(
    self.dynamics_state, opponent_mask
)

opponent_r = np.zeros(tao)
for i in range(1, tao + 1):
    opponent_r[i - 1] = get_firing_rate(opponent_region_avg_dynamics_state, -i)
opponent_r = np.mean(opponent_r)
```

**数学表达：**
```
r_opponent = (1/τ) · Σ[i=1 to τ] firing_rate_opponent(t-i)
```

---

#### 5.2.2 关键切换判断

这是**产生Gamma分布感知持续时间的核心机制**。

```python
opponent_threshold = 0.8 + np.random.uniform(-0.1, 0.05)

if opponent_r < opponent_threshold:
    return 15 + np.random.normal(0, 1.6)  # 对手弱 → 保持被抑制
else:
    return adaptation_lower_bound + np.random.normal(0, 0.9)  # 对手强 → 准备翻转
```

**数学表达：**

```
β_current = {
    15 + N(0, 1.6)  if r_opponent < 0.8 + U(-0.1, 0.05)
    2 + N(0, 0.9)   otherwise
}
```
其中 `U(a, b)` 表示均匀分布在 [a, b] 区间。

**物理意义分析：**

| 对手状态 | opponent_r | 当前区域β | 结果 |
|---------|-----------|----------|------|
| 对手主导 | > 0.8 | **2 ± 0.9** | 当前区域获得低β（高增益），准备夺回主导权 |
| 对手未主导 | < 0.8 | **15 ± 1.6** | 当前区域维持高β（高噪声），继续被抑制 |

**关键创新：**
- `np.random.uniform(-0.1, 0.05)` 产生**不对称的随机阈值**
- 阈值范围：[0.7, 0.85]
- 跨度 = 0.15（相对于激活范围[-1, 1]非常大）
- **让切换时机具有高度随机性 → Gamma分布**

---

## 6. 策略2：过渡态调节 (beta_first_order_difference)

当系统处于快速变化状态时（`|r_1| > 0.03`）。

```python
def beta_first_order_difference(r_1):
    threshold = 0.03 + np.random.normal(0, 0.025)
    if r_1 > threshold:
        return 15 + np.random.normal(0, 1.2)  # 正在上升 → 抑制
    else:
        return adaptation_lower_bound + np.random.normal(0, 0.7)  # 正在下降 → 增强
```

**物理意义：**

| 变化方向 | r₁ | β值 | 作用 |
|---------|-----|-----|------|
| 激活上升 | > 0.03 | 15 ± 1.2 | 施加制动，防止过快切换（惯性） |
| 激活下降 | < 0.03 | 2 ± 0.7 | 加速切换，完成状态转换 |

**生物学对应：**
- 模拟神经系统的**惯性与制动机制**
- 防止感知状态频繁抖动

---

## 7. 关键参数详解

### 7.1 阈值参数

| 参数名 | 均值 | 噪声分布 | 物理意义 | 调优方向 |
|--------|------|---------|---------|---------|
| `adaptation_threshold` | 0.78 | N(0, 0.09) | 触发神经适应的激活阈值 | ↑增大：延迟适应发生 |
| **`opponent_threshold`** | 0.8 | **U(-0.1, 0.05)** | **触发感知切换的对手激活阈值** | ↓降低：更易发生切换 |
| `decision_threshold` | 0.03 | N(0, 0.018) | 判断稳态/过渡态的边界 | ↑增大：更多稳态判断 |

---

### 7.2 噪声参数（控制随机性）

| 添加位置 | 噪声类型 | 标准差/范围 | 作用 |
|---------|---------|-----------|------|
| `adaptation_threshold` | Gaussian | σ = 0.09 | 自适应触发时机的变异 |
| **`opponent_threshold`** | **Uniform** | **[-0.1, 0.05]** | **🔥切换判断的强随机性** |
| `decision_threshold` | Gaussian | σ = 0.018 | 稳态判断的变异 |
| β输出（高β分支） | Gaussian | σ = 1.6 | 抑制状态的随机波动 |
| β输出（低β分支） | Gaussian | σ = 0.9 | 增强状态的随机波动 |

---

### 7.3 Sigmoid函数参数

**完整表达式：**
```
β(r) = (β_max - β_min) / (1 + exp(k·(r - r_0))) + β_min + ε
```

| 参数符号 | 代码中的值 | 物理意义 |
|---------|-----------|---------|
| β_max | 15 | β的最大值（最小增益） |
| β_min | 2 (`adaptation_lower_bound`) | β的最小值（最大增益） |
| k | 25 | Sigmoid斜率（控制陡峭程度） |
| r_0 | 0.5 | Sigmoid中心点 |
| ε | N(0, 1.1) | 输出噪声 |

**斜率演化：**
- 原始版本：`k = 100` → 几乎阶跃函数（过于确定）
- 当前版本：`k = 25` → 平滑过渡（允许渐进式适应）

---

## 8. 完整工作流程示例

### 8.1 场景：Face主导 → Place夺权

| 时间 | Face状态 (r) | Place状态 (r) | Face的β | Place的β | 主导区域 |
|------|-------------|--------------|---------|----------|---------|
| t=0 | 0.85 | 0.25 | 自适应：β≈14 | 对手弱：β≈15 | **Face** |
| t=10 | 0.82 | 0.35 | 自适应：β≈13 | 对手弱：β≈15 | **Face** |
| t=20 | 0.75 | 0.78 | 对手强：β≈**2** | 自适应：β≈12 | **切换中** |
| t=30 | 0.30 | 0.85 | 对手弱：β≈15 | 自适应：β≈14 | **Place** |

---

### 8.2 关键时刻详解（t=20）

**Face区域的计算：**
1. Face的 `r = 0.75 < 0.78` → 进入"对手抑制"分支
2. 查询Place的激活：`r_opponent = 0.78`
3. 生成随机阈值：`threshold = 0.8 + U(-0.1, 0.05) = 0.72`（假设）
4. 判断：`0.78 > 0.72` → Face获得 `β = 2`（准备让出主导）

**Place区域的计算：**
1. Place的 `r = 0.78 > 0.78` → 触发自适应分支
2. Place获得 `β ≈ 12`（适度抑制，但仍在主导）

**系统行为：**
- Face获得低β → 降低噪声，提高对竞争的敏感性
- Place维持较高β → 由于长时间主导而开始疲劳
- 几个时间步后，Face完全夺回主导权

---

## 9. 产生Gamma分布的数学机制

### 9.1 随机性来源

感知持续时间的变异来自以下**多层级随机过程的叠加**：

1. **切换判断的强随机性**
   ```python
   opponent_threshold = 0.8 + np.random.uniform(-0.1, 0.05)
   ```
   - 每次判断切换的阈值都不同
   - 范围 [0.7, 0.85]，跨度 = 0.15（非常大）

2. **所有β输出都添加噪声**
   ```python
   return 15 + np.random.normal(0, 1.6)
   return adaptation_lower_bound + np.random.normal(0, 0.9)
   ```
   - 即使激活状态相同，β值也有变异
   - 导致停留时间的额外随机性

3. **级联阈值的复合随机性**
   - `decision_threshold`、`adaptation_threshold` 都有噪声
   - 多个随机判断的叠加效应

---

### 9.2 从随机过程到Gamma分布

**数学原理：**

假设感知状态的切换由泊松过程驱动，但切换率 λ 本身是随机的：

```
λ(t) = λ_0 · f(r(t), r_opponent(t), noise)
```

当切换率具有随机性时，等待时间分布从**指数分布**演化为**Gamma分布**：

```
P(T = t) = (λ^k / Γ(k)) · t^(k-1) · exp(-λ·t)
```

其中：
- `k`：形状参数（与噪声强度相关）
- `λ`：尺度参数（与平均切换率相关）
- `Γ(k)`：Gamma函数

**关键因素：**
- `opponent_threshold` 的大范围随机性 → 控制 k 参数
- 神经适应机制 → 引入记忆效应，增强Gamma特性

---

## 10. 调优指南

### 10.1 获得更理想的Gamma分布

**目标：**增加感知持续时间的变异性，产生右偏的Gamma分布。

#### 方案1：增强切换随机性

```python
# 原始
opponent_threshold = 0.8 + np.random.uniform(-0.1, 0.05)

# 优化
opponent_threshold = 0.73 + np.random.uniform(-0.15, 0.08)  # 范围扩大到 [0.58, 0.81]
```

**效果：**
- 切换判断更加随机
- 产生更长的尾部（允许偶尔的超长感知）

---

#### 方案2：增强β输出噪声

```python
# 原始
if opponent_r < opponent_threshold:
    return 15 + np.random.normal(0, 1.6)
else:
    return adaptation_lower_bound + np.random.normal(0, 0.9)

# 优化
if opponent_r < opponent_threshold:
    return 15 + np.random.normal(0, 2.0)  # σ: 1.6 → 2.0
else:
    return adaptation_lower_bound + np.random.normal(0, 1.5)  # σ: 0.9 → 1.5
```

**效果：**
- 即使在相同激活条件下，β值也有更大波动
- 增加短时感知的概率（Gamma分布的峰值更尖锐）

---

#### 方案3：调整阈值均值

```python
# 降低切换难度，增加长时感知的概率
opponent_threshold = 0.70 + np.random.uniform(-0.12, 0.08)  # 原来是 0.8
```

**效果：**
- 切换更难触发
- Gamma分布的尾部更长

---

### 10.2 参数调优表

| 目标 | 参数 | 调整方向 | 效果 |
|------|------|---------|------|
| 增加切换频率 | `opponent_threshold` 均值 | ↑ 增大 | 更易切换，缩短平均持续时间 |
| 减少切换频率 | `opponent_threshold` 均值 | ↓ 降低 | 更难切换，延长平均持续时间 |
| 增加持续时间变异 | `opponent_threshold` 范围 | ↑ 扩大 | Gamma分布更宽，形状参数k减小 |
| 增加短时感知 | β输出噪声 | ↑ 增大 | 峰值更尖锐 |
| 增加长时感知 | `opponent_threshold` 均值 | ↓ 降低 | 尾部更长 |

---

## 11. 生物学对应关系

| 计算模型组件 | 神经生物学机制 | 文献支持 |
|------------|---------------|---------|
| 高β → 抑制 | 抑制性神经递质（GABA） | Wilson (2003) |
| 低β → 增强 | 兴奋性神经递质（Glutamate） | Tong et al. (2006) |
| 神经适应 | 感受器疲劳、突触抑制 | Blake & Logothetis (2002) |
| 对手抑制 | 互抑制网络（Mutual Inhibition） | Lehky (1988) |
| Gamma分布 | 实验测得的感知持续时间分布 | Levelt (1967) |

---

## 12. 参考文献

1. **Blake, R., & Logothetis, N. K. (2002).** Visual competition. *Nature Reviews Neuroscience*, 3(1), 13-21.

2. **Levelt, W. J. (1967).** Note on the distribution of dominance times in binocular rivalry. *British Journal of Psychology*, 58(1‐2), 143-145.

3. **Lehky, S. R. (1988).** An astable multivibrator model of binocular rivalry. *Perception*, 17(2), 215-228.

4. **Tong, F., Meng, M., & Blake, R. (2006).** Neural bases of binocular rivalry. *Trends in Cognitive Sciences*, 10(11), 502-511.

5. **Wilson, H. R. (2003).** Computational evidence for a rivalry hierarchy in vision. *Proceedings of the National Academy of Sciences*, 100(24), 14499-14503.

---

## 附录A：完整代码

```python
def changed_beta_func(self, region_avg_dynamics_state, mask, beta, tao, adaptation_lower_bound=2):
    """
    Face vs Place竞争版本：通过自适应增益调节实现感知切换

    参数:
        region_avg_dynamics_state: 当前区域的激活历史时间序列
        mask: 当前神经元所属的区域掩码
        beta: 基础β值（本函数中未使用）
        tao: 统计时间窗口大小
        adaptation_lower_bound: β的最小值（最大增益）

    返回:
        float: 为当前区域计算得到的β值
    """

    place_mask = self.place_mask
    limb_mask = self.limb_mask
    face_mask = self.face_mask
    object_mask = self.object_mask

    def get_firing_rate(dynamics_state, t):
        if dynamics_state.shape[0] >= 2:
            temp = dynamics_state[t]
        else:
            temp = dynamics_state.mean(axis=0)
        return temp

    def first_order_difference(dynamics_state, t2, t1):
        temp = get_firing_rate(dynamics_state, t2) - get_firing_rate(dynamics_state, t1)
        return temp

    def beta_first_order_difference(r_1):
        threshold = 0.03 + np.random.normal(0, 0.025)
        if r_1 > threshold:
            return 15 + np.random.normal(0, 1.2)
        else:
            return adaptation_lower_bound + np.random.normal(0, 0.7)

    def beta_r(r):
        # Object和Limb区域：简单处理
        if (mask == object_mask).all() or (mask == limb_mask).all():
            return (15 - adaptation_lower_bound) / (1 + np.exp(30 * (r - 0.5))) + adaptation_lower_bound

        # Face和Place区域：竞争机制
        else:
            adaptation_threshold = 0.78 + np.random.normal(0, 0.09)

            if r > adaptation_threshold:
                base_beta = (15 - adaptation_lower_bound) / (1 + np.exp(25 * (r - 0.5))) + adaptation_lower_bound
                return base_beta + np.random.normal(0, 1.1)
            else:
                # Face vs Place opponent logic
                if (mask == face_mask).all():
                    opponent_mask = place_mask
                elif (mask == place_mask).all():
                    opponent_mask = face_mask
                else:
                    return 15

                opponent_region_avg_dynamics_state = self.avg_activation_in_mask_timeserise(
                    self.dynamics_state, opponent_mask
                )

                opponent_r = np.zeros(tao)
                for i in range(1, tao + 1):
                    opponent_r[i - 1] = get_firing_rate(opponent_region_avg_dynamics_state, -i)
                opponent_r = np.mean(opponent_r)

                opponent_threshold = 0.8 + np.random.uniform(-0.1, 0.05)

                if opponent_r < opponent_threshold:
                    return 15 + np.random.normal(0, 1.6)
                else:
                    return adaptation_lower_bound + np.random.normal(0, 0.9)

    # 计算统计量
    r = np.zeros(tao)
    r_1 = np.zeros(tao)

    for i in range(1, tao + 1):
        r[i - 1] = get_firing_rate(region_avg_dynamics_state, -i)
        r_1[i - 1] = first_order_difference(region_avg_dynamics_state, -i, -(i + 1))

    r_1 = np.mean(r_1)
    r = np.mean(r)

    # 决策
    decision_threshold = 0.03 + np.random.normal(0, 0.018)

    if np.abs(r_1) < decision_threshold:
        return beta_r(r)
    else:
        return beta_first_order_difference(r_1)
```

---

**文档版本:** 1.0
**最后更新:** 2026-01-05
**作者:** Claude Code Analysis
