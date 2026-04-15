# AOI 多通道缺陷检测与分割 (100寸面板 ADC 项目)

本项目基于魔改版 YOLO (Ultralytics) 框架，专为工业 AOI 多通道图像（DOT 9通道 / MURA 8通道）设计的端到端缺陷检测与分割流水线。实现了“原图多通道直出 + 纯表征学习”，免去传统的繁杂图像预处理。

---

## 1. 标准工作流 (Workflow)

为了方便长期维护，请严格遵循以下操作顺序：

1.  **数据归档**：将标注好的数据存放（或软链接）至 `dataset/raw_DOT_by_defect` 和 `dataset/raw_MURA_by_defect`。
2.  **质量检查**：使用 `dataset/dataset-check.ipynb` 可视化原始 LabelMe 标注，检查笔误或误标。
3.  **格式转换与融合**：运行 `dataset/convert_labelme_to_yolo.py`。
    -   将同一个 Unique ID 的多 Pattern 图像堆叠为 `.npy` (uint8)。
    -   跨 Pattern 融合标注（IoU > 0.5 且同标签则合并）。
    -   按 UID 划分 Train/Val，防止数据泄露。
4.  **数据清洗**：运行 `dataset/clean_dataset.py`。
    -   执行合并规则（如合并各种暗点、线类缺陷）。
    -   过滤总数 < 10 的极稀有类别。
    -   生成最终的 `cleaned_YOLO_DOT` 和 `cleaned_YOLO_MURA` 训练集。
5.  **数据均衡 (可选)**：运行 `dataset/oversample.py` 对稀有类别进行过采样。
6.  **启动训练**：根据显卡类型，在 `train-predict/` 下使用 `screen` 启动一键训练脚本。

---

## 2. 环境部署与显卡适配

### 2.1 现代显卡 (A100 / RTX 3090 / RTX 4090)
**启动脚本**：`train_aoi_screen.sh` (默认开启 AMP)。
**环境配置**：
```bash
cd /roing-workspace/ultralytics-aoi/
pip install -e .  # 以开发者模式安装魔改版源码
```

### 2.2 老旧显卡 (Tesla P100)
**启动脚本**：`train_aoi_screen-p100.sh` (强制关闭 AMP)。
**环境配置**：由于 P100 (sm_60) 与现代 PyTorch 存在兼容性问题，需运行专用配置脚本：
```bash
cd /roing-workspace/train-predict/
bash setup_aoi_env-p100.sh
```

---

## 3. 数据集管理 (软链接与存储)

建议将原始数据集存放于挂载盘（如 `data14`），通过软链接引入，以保护系统盘空间：
```bash
# 示例
ln -s /data14/7B新线数据-标注/raw_DOT_by_defect /roing-workspace/dataset/raw_DOT_by_defect
ln -s /data14/7B新线数据-标注/raw_MURA_by_defect /roing-workspace/dataset/raw_MURA_by_defect
```
**注意**：`.npy` 转换脚本默认使用符号链接。迁移数据集时，请确保软链接指向的物理路径依然有效。

---

## 4. 关键功能说明

### 4.1 通道数调整 (例如从 9 扩充至 12)
本项目采用代码驱动配置。若需增加通道，**只需修改两个脚本的常量并重新跑一遍流程**，无需手动修改 YAML：
1.  打开 `dataset/convert_labelme_to_yolo.py`，在 `DOT_PATTERNS` 或 `MURA_PATTERNS` 列表中增加新的通道名称。

2.  打开 `dataset/clean_dataset.py`，将顶部的 `DOT_CHANNELS` 或 `MURA_CHANNELS` 常量改为新的数值（如 12）。

3.  **重新生成数据**：
    依次运行：
    ```bash
    python convert_labelme_to_yolo.py
    python clean_dataset.py
    ```
    清洗脚本会自动在 `cleaned_YOLO_DOT/dataset.yaml` 中写入正确的 `channels: 12`。

    > **底层原理**：训练开始时，YOLO 框架会读取 YAML 中的 `channels` 字段来构建第一层卷积。如果该数值与 `.npy` 文件的实际通道深度不匹配，程序会报错。只要这两个脚本的设置是一致的，训练就能正常启动。

    > **通道顺序 (严格约束)**：
    `.npy` 文件的通道深度顺序严格遵循 `dataset/convert_labelme_to_yolo.py` 中 `DOT_PATTERNS` / `MURA_PATTERNS` 的列表索引。
    - **警告**：训练好的模型是与该顺序绑定的。**严禁在已有训练权重的基础上打乱列表顺序**，否则模型将无法正确识别特征。
    - **扩展建议**：新增通道时，请添加到列表末尾，不要改变已有通道的相对位置。



### 4.2 数据均衡 (`oversample.py`)
用于解决长尾分布问题。
- **逻辑**：识别样本数少于阈值的类别，通过复制 `.npy/.txt` 对进行扩充。
- **用法**：`python oversample.py --target-factor 0.3` (将少数类扩充至最大类别的 30%)。

### 4.3 权重初始化逻辑
当输入通道（如 9/8）与预训练权重（3通道）不符时，底层 `BaseModel.load` 已被魔改为执行 **Mean-Broadcast (均值广播)**：
- 将 RGB 3 通道的权重取均值，广播给所有新通道。这确保了多通道模型能继承强大的表征提取能力。

---

## 5. 开发者备忘录 (故障排查)

- **训练意外停止**：两套脚本均支持断点续训。
  - 设置 `export DOT_RESUME=True` 或 `SEG_RESUME=True` 后运行脚本。
- **显存优化 (P100)**：P100 只有 16G 显存，建议：
  - DOT: `BATCH=16`
  - MURA: `BATCH=4` (因分割头占用大)
  - 必须设置 `amp=False`。
- **NumPy 兼容性**：若报错 `Numpy is not available`，请确保版本为 `numpy<2`。