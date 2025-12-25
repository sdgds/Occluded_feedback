# V1抽象vs精细分类时间滞后效应的机制假设

## 现象
- 抽象分类(2分类：生命vs非生命)的解码正确率超过随机水平的时间点**晚于**精细分类(40分类)
- 已测试：调节全局抑制性连接强度 → **无效**

## 基于文献的新假设

### 假设1: 神经元时间常数差异 ⭐⭐⭐ (最推荐)

**理论依据**:
- 抑制性神经元时间常数(τI) > 兴奋性神经元时间常数(τE)
- 抽象信息需要更长的时间整合窗口
- 文献: [PMC12680323](https://pmc.ncbi.nlm.nih.gov/articles/PMC12680323/)

**测试方案**:
1. **调节膜时间常数** (在cell models中):
   - 增大抑制性神经元的τm → 预期抽象分类更晚
   - 减小抑制性神经元的τm → 预期时间滞后减少

2. **具体参数**:
   ```python
   # 在 components/cell_models/*.json 中修改
   # 例如: IntFire1_inh.json
   {
       "tau_m": 10.0,  # 默认值，尝试 15.0, 20.0 (增大) 或 5.0, 7.0 (减小)
       "tau_ref": 2.0,
       ...
   }
   ```

3. **预测**:
   - τI增大 → 时间滞后增大
   - τI减小 → 时间滞后减小
   - τE/τI比值是关键

---

### 假设2: 抑制性亚型特异性 ⭐⭐⭐

**理论依据**:
- Pvalb: 快速抑制，短时间常数 → 精细分类
- Sst: 树突抑制，长时间常数 → 抽象分类（需要长时程整合）
- 文献: [PMC9915496](https://pmc.ncbi.nlm.nih.gov/articles/PMC9915496/)

**测试方案**:
1. **选择性调节Sst连接**:
   ```python
   # 使用你的scale_v1_synaptic_weights函数
   scale_v1_synaptic_weights(
       edges_h5_path="network/v1_v1_edges.h5",
       nodes_h5_path="network/v1_nodes.h5",
       node_types_csv_path="network/v1_node_types.csv",
       pop_names=["i23Sst", "i4Sst", "i5Sst", "i6Sst"],  # 只调节Sst
       apply_to="source",  # 从Sst发出的连接
       scale_factor=0.5,   # 或 1.5, 2.0
   )
   ```

2. **对比测试Pvalb**:
   ```python
   scale_v1_synaptic_weights(
       pop_names=["i23Pvalb", "i4Pvalb", "i5Pvalb", "i6Pvalb"],
       scale_factor=0.5,
   )
   ```

3. **预测**:
   - 减弱Sst → 抽象分类变快，时间滞后减小
   - 增强Sst → 抽象分类更慢，时间滞后增大
   - Pvalb调节 → 可能影响精细分类速度，但不影响时间滞后

---

### 假设3: 层间反馈连接 ⭐⭐

**理论依据**:
- 深层(L5/6)处理抽象信息
- 深层→浅层的反馈对抽象分类至关重要
- 文献: [Nature s41593-023-01510-5](https://www.nature.com/articles/s41593-023-01510-5)

**测试方案**:
1. **增强L5→L2/3连接**:
   ```python
   # 需要先确定L5和L2/3的node_type_id
   # 然后选择性调节跨层连接

   # 思路：从v1_v1_edges中筛选 source层=L5, target层=L2/3 的边
   # 这需要写一个新函数，基于层的连接调节
   ```

2. **测试L6→L4连接**:
   - L6是反馈的主要来源
   - L4接收丘脑输入并向浅层投射

3. **预测**:
   - 增强深→浅反馈 → 抽象信息更快到达浅层
   - 减弱深→浅反馈 → 抽象分类延迟更明显

---

### 假设4: 循环连接vs前馈连接 ⭐⭐

**理论依据**:
- 抽象分类需要更多循环处理
- 局部循环连接支持时间整合
- 文献: [PMC11188075](https://pmc.ncbi.nlm.nih.gov/articles/PMC11188075/)

**测试方案**:
1. **识别循环连接**:
   - 同一层内的连接 (L2/3→L2/3, L5→L5)
   - vs 前馈连接 (L4→L2/3, L2/3→L5)

2. **调节策略**:
   ```python
   # 增强同层循环（特别是L5）
   scale_v1_synaptic_weights(
       layer="L5",
       apply_to="source",  # 从L5发出
       # 需要额外筛选target也在L5
       scale_factor=1.5,
   )
   ```

3. **预测**:
   - 增强循环 → 时间滞后可能增大（更多时间整合）
   - 减弱循环 → 时间滞后减小

---

### 假设5: LGN输入的时间特性 ⭐

**理论依据**:
- 不同V1细胞类型接收不同时间特性的LGN输入
- Magno vs Parvo通路的时间差异

**测试方案**:
1. **查看LGN→V1的delay**:
   ```python
   # lgn_v1_edge_types.csv 显示所有delay=1.7ms
   # 可能需要调节特定细胞类型的LGN输入delay
   ```

2. **调节特定细胞类型的LGN输入强度**:
   - 兴奋性vs抑制性细胞接收的LGN输入比例

---

## 推荐测试顺序

1. **首选**: 假设2 (Sst vs Pvalb) - 最容易实现，生物学基础最强
2. **次选**: 假设1 (时间常数) - 需要修改cell models
3. **第三**: 假设3 (层间连接) - 需要更复杂的边筛选
4. **探索**: 假设4 (循环连接)

## 实验设计建议

对于每个假设，建议：
1. **梯度测试**: 不要只测试开/关，而是测试连续的scale_factor (0.25, 0.5, 1.0, 1.5, 2.0)
2. **对照组**: 同时测试相反的调节方向
3. **特异性**: 测试是否只影响时间滞后，而不改变整体解码性能
4. **剂量-反应关系**: 检查效果是否与调节强度成比例

## 参考文献

1. [Hierarchical temporal processing in mouse visual system](https://pmc.ncbi.nlm.nih.gov/articles/PMC11373856/)
2. [Recurrent circuits in primary visual cortex](https://www.nature.com/articles/s41593-023-01510-5)
3. [Animacy processing in temporal cortex](https://pmc.ncbi.nlm.nih.gov/articles/PMC11671252/)
4. [Neural time constants and E/I balance](https://pmc.ncbi.nlm.nih.gov/articles/PMC12680323/)
5. [Input-specific synaptic depression](https://pmc.ncbi.nlm.nih.gov/articles/PMC9915496/)
6. [Early visual cortex contributions to categorization](https://pmc.ncbi.nlm.nih.gov/articles/PMC10312552/)
7. [Recurrent cortical networks encode temporal statistics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11188075/)
