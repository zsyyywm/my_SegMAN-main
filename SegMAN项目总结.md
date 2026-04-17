# SegMAN 项目总结

本文档概括 **`mamba-main/SegMAN-main/SegMAN-main`** 仓库的定位、依赖、官方论文结果，以及本环境中 **DataA / DataB 二分类语义分割** 的 **测试集评估 JSON** 与相关实验说明。更细的环境与命令见同目录下的 [`SEGMAN_EXPERIMENT_SETUP.md`](SEGMAN_EXPERIMENT_SETUP.md) 与 [`README.md`](README.md)。

---

## 1. 项目是什么

**SegMAN**（**Seg**mentation with **M**amba and **A**ttention **N**etwork）是 **CVPR 2025** 官方 PyTorch 实现，论文题目为 *SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation*。

- **论文 / 预印本**：[arXiv:2412.11890](https://arxiv.org/abs/2412.11890)  
- **核心思想**：在语义分割中结合 **状态空间模型（SSM）** 与 **局部注意力（NATTEN 等）**，做多尺度上下文建模；分割框架基于 **MMSegmentation v0.30.0**。

仓库根目录主要包含：

| 路径 | 说明 |
|------|------|
| `segmentation/` | MMSeg 工程与训练、测试、配置（**日常命令在此目录执行**） |
| `kernels/selective_scan/` | VMamba 风格的 **Selective Scan 2D** CUDA 扩展，需本地编译安装 |
| `assets/` | 论文配图（模型结构、主结果图等） |
| `pretrained/` | ImageNet-1k 预训练骨干权重放置位置（如 `SegMAN_Encoder_t.pth.tar`） |

实现致谢与依赖栈：**MMSegmentation**、**Natten**、**VMamba**、**SegFormer** 等（详见 `README.md`）。

---

## 2. 环境与依赖要点

- **Python**：建议 3.10（`README` / `SEGMAN_EXPERIMENT_SETUP` 一致）。  
- **PyTorch**：示例为 `torch==2.1.2` + 对应 `torchvision`；若 **torch ≥ 2.1**，需按 `README` 修改 mmcv 中 `_functions.py` 一处（避免多卡 stream 报错）。  
- **MMSegmentation**：**v0.30.0**；通过 `openmim` 安装 `mmcv-full` 后，在 `segmentation` 下 `pip install -v -e .`。  
- **Natten**：须与 CUDA / PyTorch 版本匹配的 wheel（见 `README`）。  
- **selective_scan**：`cd kernels/selective_scan && pip install .`（需可用 CUDA 编译环境）。  
- **其他**：`requirements.txt` 中列出 `mmcv-full==1.7.2`、`mmengine` 等；二分类实验说明中强调 **NumPy 1.x** 与数据路径（`DataA-B/DataA`、`DataB`）见 `SEGMAN_EXPERIMENT_SETUP.md`。

---

## 3. 官方论文表格结果（标准数据集）

以下为仓库 `README.md` 中公开的 **单尺度语义分割 mIoU** 及参数量、FLOPs（与论文 / 官方权重一致，便于与文献对比）。

### 3.1 ADE20K

| 模型 | 骨干 (ImageNet Top-1) | mIoU | Params | FLOPs |
|------|------------------------|------|--------|-------|
| SegMAN-T | Encoder-T (76.2) | 43.0 | 6.4M | 6.2G |
| SegMAN-S | Encoder-S (84.0) | 51.3 | 29.4M | 25.3G |
| SegMAN-B | Encoder-B (85.1) | 52.6 | 51.8M | 58.1G |
| SegMAN-L | Encoder-L (85.5) | 53.2 | 92.6M | 97.1G |

### 3.2 Cityscapes

| 模型 | 骨干 (ImageNet Top-1) | mIoU | Params | FLOPs |
|------|------------------------|------|--------|-------|
| SegMAN-T | Encoder-T (76.2) | 80.3 | 6.4M | 52.5G |
| SegMAN-S | Encoder-S (84.0) | 83.2 | 29.4M | 218.4G |
| SegMAN-B | Encoder-B (85.1) | 83.8 | 51.8M | 479.0G |
| SegMAN-L | Encoder-L (85.5) | 84.2 | 92.6M | 769.0G |

### 3.3 COCO-Stuff-164K

| 模型 | 骨干 (ImageNet Top-1) | mIoU | Params | FLOPs |
|------|------------------------|------|--------|-------|
| SegMAN-T | Encoder-T (76.2) | 41.3 | 6.4M | 6.2G |
| SegMAN-S | Encoder-S (84.0) | 47.5 | 29.4M | 25.3G |
| SegMAN-B | Encoder-B (85.1) | 48.4 | 51.8M | 58.1G |
| SegMAN-L | Encoder-L (85.5) | 48.8 | 92.6M | 97.1G |

权重与配置文件链接见原 [`README.md`](README.md) 表格。

---

## 4. 本仓库内自定义二分类实验（DataA / DataB）

在 `segmentation/local_configs/segman_binary/` 下提供 **512×512**、**前景 / 背景** 二分类配置（如 `segman_t_dataa_512_iou.py`、`segman_t_datab_512_iou.py`）。数据根目录默认指向与 TransNeXt 对比实验一致的 **`DataA-B/DataA`** 与 **`DataA-B/DataB`**（相对路径说明见 `SEGMAN_EXPERIMENT_SETUP.md`）。

与 Mask2Former / TransNeXt 侧的约定对齐说明（双 checkpoint、按 IoU 或 val loss 存 best 等）同样写在 `SEGMAN_EXPERIMENT_SETUP.md`。

---

## 5. 本环境测试集评估结果（`tools/test.py` 导出 JSON）

以下数值来自 **`segmentation/checkpoints/`** 下由 MMSeg 测试脚本写入的 **`eval_single_scale_*.json`**（**单尺度测试**；指标为 0–1 小数，**百分比 = 数值 × 100**）。

### 5.1 DataB（配置：`segman_t_datab_512_iou.py`）

**结果文件**：`segmentation/checkpoints/test_datab_iou61/eval_single_scale_20260417_141621.json`

| 指标 | 数值 |
|------|------|
| aAcc | 0.9868 |
| **mIoU** | **0.8703** |
| mAcc | 0.9701 |
| mFscore | 0.9266 |
| mPrecision | 0.8911 |
| mRecall | 0.9701 |
| IoU.background | 0.9863 |
| **IoU.foreground** | **0.7544** |
| Acc.background | 0.9884 |
| Acc.foreground | 0.9519 |
| Fscore.foreground | 0.8600 |
| Precision.foreground | 0.7844 |
| Recall.foreground | 0.9519 |

### 5.2 DataA（配置：`segman_t_dataa_512_iou.py`）

**结果文件**：`segmentation/checkpoints/test_dataa_iou63/eval_single_scale_20260417_141431.json`

| 指标 | 数值 |
|------|------|
| aAcc | 0.9917 |
| **mIoU** | **0.7365** |
| mAcc | 0.9356 |
| mFscore | 0.8229 |
| mPrecision | 0.7573 |
| mRecall | 0.9356 |
| IoU.background | 0.9916 |
| **IoU.foreground** | **0.4814** |
| Acc.background | 0.9927 |
| Acc.foreground | 0.8786 |
| Fscore.foreground | 0.6500 |
| Precision.foreground | 0.5158 |
| Recall.foreground | 0.8786 |

> **说明**：目录名中的 `iou61` / `iou63` 仅为文件夹命名习惯；**以对应 JSON 内 `metric` 为准**。若需复现测试，请使用与训练一致的 checkpoint，并在 `segmentation` 目录下执行 `README` 中的 `tools/test.py` 命令，指向上述 config 与权重。

---

## 6. 训练与测试命令索引

- **训练**（官方 ADE20K 等）：`cd segmentation` 后使用 `tools/train.py` 或 `tools/dist_train.sh`（见 `README.md`）。  
- **二分类 DataA/DataB**：见 `SEGMAN_EXPERIMENT_SETUP.md` 第八节示例命令。  
- **测试 / 评估**：`tools/test.py` 或 `tools/dist_test.sh` + 配置文件 + checkpoint 路径（见 `README.md`）。  
- **可视化**：`segmentation/image_demo.py`（`README.md`「Visualization」一节）。

---

## 7. 引用

```bibtex
@inproceedings{SegMAN,
    title={SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation},
    author={Yunxiang Fu and Meng Lou and Yizhou Yu},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```

---

*文档生成依据：仓库 `README.md`、`SEGMAN_EXPERIMENT_SETUP.md`、`requirements.txt`，以及 `segmentation/checkpoints/test_*/*.json` 中的实测指标。*
