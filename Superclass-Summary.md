# Superclass-AlexNet 实验总结文档

## 研究目标

验证是否加入抽象类别信息有助于DCNN最后一层的分类正确率提升。

核心思路：将ImageNet的1000个细分类别聚合为k个抽象大类，利用大类信息作为top-down信号来约束和引导最后一层的分类。

---

## 第一步：构建抽象类别层级（Word Hierarchy）

### 1.1 基于WordNet的语义层级

**方法**：
- 使用WordNet的词语层级关系，计算ImageNet 1000类之间的语义相似性
- 通过`path_similarity`计算任意两个类别的相似度，构建1000×1000的相似性矩阵

**实现**：
```python
def get_word_RSM(class_index1, class_index2):
    synset1 = wn.synset_from_pos_and_offset('n', class_index1)
    synset2 = wn.synset_from_pos_and_offset('n', class_index2)
    return synset1.path_similarity(synset2)
```

**聚类**：
- 使用K-means将1000类聚为k个大类
- 将相似性矩阵转换为距离矩阵（1 - 相似度）后进行聚类

**结果**：
- 成功构建了基于语义的类别层级结构
- 相似性矩阵显示出明显的块状结构（见可视化图）
<img width="436" height="410" alt="image" src="https://github.com/user-attachments/assets/545ce8aa-7267-4acc-b2a2-7657dfad7139" />


### 1.2 基于BERT的词嵌入层级

**方法**：
- 使用BERT模型（bert-base-uncased）为每个类别名称生成词嵌入
- 计算词嵌入之间的相关系数作为相似性度量
- 使用K-means聚类生成抽象大类

**实现**：
```python
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()
```

**对比WordNet**：
- BERT能够捕捉更细粒度的语义关系
- 相似性矩阵结构与WordNet有所不同

---

## 第二步：验证抽象大类的有效性上限（Analysis AlexNet）

### 2.1 Oracle实验：假设大类分类完美

**核心思路**：
在给定完美大类标签的前提下，计算AlexNet新分类正确率的理论上限。

**算法逻辑**：
1. 对每张图片，计算AlexNet的top-k预测（k通常为9-10）
2. 获取每个预测类别对应的大类标签
3. 确定图片的正确大类标签
4. 在正确大类内，选择激活值最高的小类作为最终预测

**关键代码逻辑**：
```python
for i in range(labels.shape[0]):
    pt = predicted_top5[i]  # top-k预测
    l = labels[i]  # 真实标签
    if l in pt:  # 如果真实标签在top-k中
        loc = torch.where(pt==l)[0].item()  # 找到位置
        # 检查该位置前是否有重复大类
        if cluster_labels[loc] not in cluster_labels[pt[:loc]]:
            count += 1  # 可以被救回
```

**实验设置**：
- 模型：AlexNet（Top-1准确率 ~56%）
- Top-k范围：k=10
- 聚类方法：WordNet语义聚类
- 测试集：ImageNet验证集（50,000张图片）

**实验结果**：

| 大类数量 k | Oracle准确率 | 提升幅度 |
|-----------|-------------|---------|
| k=2 | 70.80% | +14.25% |
| k=3 | 75.29% | +18.74% |
| k=4 | 77.64% | +21.09% |
| k=5 | 77.52% | +20.97% |
| k=6 | 79.71% | +23.16% |
| k=7 | 79.44% | +22.89% |
| k=8 | 79.36% | +22.81% |
| k=9 | 80.25% | +23.70% |
| k=10 | 81.45% | +24.90% |

**关键发现**：
- ✅ **巨大的提升潜力**：在k=10时，准确率从56%提升到81%，增幅高达25个百分点
- ✅ **k值选择**：k在6-10之间效果最好，太小（k<5）限制太强，太大则约束不足
- ✅ **top-k覆盖率高**：AlexNet的top-10预测包含了大量正确答案，为top-down策略提供了基础

**意义**：
这个Oracle实验证明了"如果能准确预测大类，就能显著提升分类性能"的假设，为后续尝试提供了理论上限目标。

---

## 第三步：Top-Down策略实现尝试

### 3.1 尝试1：基于大类激活值总和的Top-Down

**方法**：
1. 获取AlexNet的top-k预测及激活值
2. 计算每个大类在top-k中的总激活值（所有属于该大类的小类激活值之和）
3. 选择总激活值最大的大类
4. 在该大类中选择激活值最高的小类作为最终预测

**实验设置**：
- 模型：ResNet50（Baseline: 76.14%）
- 大类数量：k=6
- Top-k：10

**结果**：
```
Baseline (Top-1):  76.14%
Top-Down:          67.19%
改进：              -8.95%
```

**问题分析**：
- ❌ **性能反而下降**：准确率降低了约9个百分点
- 原因：大类预测准确率不足，错误的大类选择导致性能恶化

**详细统计**：
- 大类正确时：top-down准确率65.93%
- 大类错误时：top-down准确率仅22.60%
- 救回（Rescued）：628个样本
- 破坏（Broke）：5,105个样本
- 净损失：-4,477个样本

---

## 第四步：训练独立的大类分类器

### 4.1 方案A：基于AlexNet logits的分类器

**架构**：
```python
class AlexNetSuperclassClassifier(nn.Module):
    def __init__(self, pretrained_alexnet, num_superclasses=10):
        # 冻结AlexNet
        self.feature_extractor = pretrained_alexnet

        # 可训练的大类分类头
        self.superclass_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, 512),
            nn.ReLU(),
            nn.Linear(512, num_superclasses)
        )
```

**训练设置**：
- Backbone：ResNet50（冻结）
- 输入：ResNet50的1000维logits
- 输出：k个大类
- 优化器：SGD, lr=0.001
- 训练集：ImageNet完整训练集
- 大类标签：基于WordNet聚类（k=10）

#### 置信度门控策略
<img width="1589" height="1189" alt="image" src="https://github.com/user-attachments/assets/49e178fa-dd30-485b-b194-caf8c28c3294" />
<img width="1588" height="1189" alt="image" src="https://github.com/user-attachments/assets/fe064d75-692e-4d67-aab2-c9f32627de48" />


**改进思路**：
只在AlexNet低置信度时且大类分类器比较准确时使用top-down，高置信度时保持原预测。

**训练结果**：
```
大类分类器验证集准确率（目前）：87%
加入top-down策略（小类confidence<0.6，大类confidence>0.8）:59.12% （Improvement: +2.57%）
```



### 4.3 尝试：在困难样本上fine-tune大类分类器

**思路**：
专门针对AlexNet低置信度的困难样本训练大类分类器

**实现**：
```python
# 识别困难样本（置信度 < 0.5）
hard_sample_indices = identify_hard_samples(
    alexnet, train_loader, device,
    confidence_threshold=0.5
)

# 只在困难样本上训练
hard_sample_loader = create_hard_sample_loader(
    train_dataset, hard_sample_indices, batch_size=128
)

# 使用加权损失处理类别不平衡
class_weights = compute_class_weights(...)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**结果**：
- 困难样本比例：43.9%（21,974个）
- 困难样本训练集大类准确率：有提升
- **但top-down策略仍然失败**

**原因分析**：
即使大类分类器在困难样本上表现更好，但Case 2（破坏正确预测）的问题依然严重。

---
