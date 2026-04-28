# SegMAN 语义分割（本仓库说明）

本目录为 **[SegMAN (CVPR 2025)](https://arxiv.org/abs/2412.11890)** 官方 PyTorch 实现的一份可运行拷贝：在 **MMSegmentation v0.30.0** 上实现 **SegMANEncoder + SegMANDecoder**（**SSM + 局部注意力 + 全局注意**），并扩展了与课题一致的 **DataA / DataB 电线二分类** 配置与产物目录约定。

**工程根**：内含 **`segmentation/`**、**`kernels/`**、**`requirements.txt`** 的目录（常见克隆路径：`my_MambaVision-main/my_SegMAN-main-main/my_SegMAN-main-main/`）。

---

## 文档分工（请先读本节）

| 文档 | 用途 |
|------|------|
| **本文 `README.md`** | **项目综述**：任务与目录、官方结果表（节选）、二分类说明、与 BEVANet 等课题对照、评测占位。**不写长安装与长命令块。** |
| **[`SETUP.md`](SETUP.md)** | **环境与依赖**：Conda、PyTorch、MMSeg / mmcv、NATTEN、selective_scan、`pip install`、**数据与预训练权重路径**、自检。 |
| **[`SegMAN.md`](SegMAN.md)** | **命令速查**：`cd segmentation` 后可直接复制的 **训练 / 测试 / 多卡 / 可视化** 命令。 |
| **[`segmentation/SegMAN.md`](segmentation/SegMAN.md)** | 二分类 **WIRE 三方案** 与 Hook 实现索引（实现细节，非日常命令入口）。 |

---

## 重要目录与产物

| 路径 | 说明 |
|------|------|
| **`segmentation/`** | **日常命令在此执行**：`tools/train.py`、`tools/test.py`、`local_configs/`。 |
| **`kernels/selective_scan/`** | Selective Scan 2D CUDA 扩展，需 **`pip install .`**（见 SETUP）。 |
| **`data/checkpoints1/train_<时间戳>/`** | 二分类训练默认 **work_dir**（日志、TensorBoard、best 权重等，相对 **工程根**）。 |
| **`data/checkpoints2/test_<时间戳>/`** | 二分类 **`tools/test.py`** 未指定 `--work-dir` 时的默认输出目录。 |
| **`../../../DataA-B/DataA`（及 DataB）** | 相对 **`segmentation/`** 启动时的默认数据根；可用环境变量覆盖（见 SETUP）。 |
| **`segmentation/pretrained/`** | 二分类常用 **`segman_t_ade.pth`**（ADE20K 整网）等；下载见 SETUP。 |

**云服务器（AutoDL）**：工作区与自建 Conda 环境多在数据盘 **`/root/autodl-tmp`**（例如 **`/root/autodl-tmp/conda_envs/segman`**）；详见 [`SETUP.md`](SETUP.md)「云服务器与数据盘」一节。

**Git**：勿提交大数据集、整目录 `data/checkpoints*`、体积过大的 `*.pth`；请依赖本地 `.gitignore`。

### 预训练权重（仓库不含，需自行下载）

本仓库 **不附带** 任何 `*.pth` 文件。二分类配置默认通过 **`load_from = 'pretrained/segman_t_ade.pth'`** 从 **ADE20K 上训练好的 SegMAN-T 整网** 初始化（加载为 `strict=False`，与二分类头不兼容的层会跳过）；**训练前须自备该文件**。

| 用途 | 文件名 | 下载地址 | 放置位置（相对 **`segmentation/`** 工作目录） |
|------|--------|----------|-----------------------------------------------|
| 二分类默认 `load_from` | `segman_t_ade.pth` | [Google Drive：SegMAN-T \| ADE20K](https://drive.google.com/file/d/1d0wp7C83YjImeQmL5_CIo1qMRjPd_-8a/view?usp=sharing) | **`pretrained/segman_t_ade.pth`**（即 **`$REPO/segmentation/pretrained/segman_t_ade.pth`**） |

**落地步骤**：将下载得到的权重保存为上述文件名并放入 **`segmentation/pretrained/`**（可先 `mkdir -p segmentation/pretrained`）。命令行下载示例（需 [gdown](https://github.com/wkentaro/gdown)）见 **[`SETUP.md`](SETUP.md) §6.1**。

**其他权重**：标准多数据集分割 checkpoint 见本文 **§3** 各表 **Download** 列（与 **`local_configs/segman/`** 对应）；ImageNet 骨干与官方分割权重汇总目录见 **[`SETUP.md`](SETUP.md) §6.2–6.3**。

---

## 1. 任务与模型概要

| 项目 | 说明 |
|------|------|
| **标准语义分割** | ADE20K、Cityscapes、COCO-Stuff 等；配置在 **`segmentation/local_configs/segman/`**。 |
| **二分类（本课题）** | **前景 / 背景**，配置在 **`segmentation/local_configs/segman_binary/`**；与 BEVANet / TransNeXt 侧 **DataA-B** 约定对齐（mask **0/1**）。 |
| **骨干** | **SegMANEncoder**（T/S/B/L）；依赖 **NATTEN** 与 **selective_scan**。 |
| **解码器** | **SegMANDecoder**。 |

实现致谢：**MMSegmentation**、**NATTEN**、**VMamba**、**SegFormer** 等（详见论文与上游仓库）。

---

## 2. 二分类三方案（与 BEVANet 对齐；与 DataA / DataB 正交）

| 方案 | 配置示例 | 输入 | 验证解码 |
|------|-----------|------|----------|
| **一** | `segman_t_dataa_512_iou.py` / `segman_t_datab_512_iou.py` | 512×512 | **argmax**；IoU best / 早停。 |
| **二** | `segman_t_dataa_512_scheme2.py` / `segman_t_datab_512_scheme2.py` | 512×512 | 前景 **P>0.55**；`save_best` 对该解码下的前景 IoU。 |
| **三** | `segman_t_dataa_256_scheme3.py` | 256×256 | 前景 **P>0.5**。 |

**可选（非上表「方案」编号）**：`segman_t_dataa_512_wire_dual.py`（同训 argmax + 0.55 双指标）；`segman_t_dataa_512_val_loss.py`（TransNeXt 风格 IoU + val loss 双轨，`data/checkpoints2`）。详见 **`segmentation/SegMAN.md`**。

**命令**：一律见 **[`SegMAN.md`](SegMAN.md)**。

**指标提示**：日志中的 **mIoU** 多为 **两类 IoU 的算术平均**；**选优 / 早停** 在二分类配置中通常与 **前景 IoU**（如 **`val/IoU`**）对齐，勿与「仅看 mIoU」混读（与 BEVANet `README` 中口径说明一致）。

---

## 3. 官方多数据集结果与权重（节选）

完整配置路径见 **`segmentation/local_configs/segman/`**；**下载链接**见 **[`SETUP.md`](SETUP.md) §6** 与下表 **Download** 列。

### ADE20K

| Model | Backbone (ImageNet Top-1) | mIoU | Params | FLOPs | Config | Download |
|-------|---------------------------|------|--------|-------|--------|----------|
| SegMAN-T | Encoder-T (76.2) | 43.0 | 6.4M | 6.2G | [config](segmentation/local_configs/segman/tiny/segman_t_ade.py) | [Google Drive](https://drive.google.com/file/d/1d0wp7C83YjImeQmL5_CIo1qMRjPd_-8a/view?usp=sharing) |
| SegMAN-S | Encoder-S (84.0) | 51.3 | 29.4M | 25.3G | [config](segmentation/local_configs/segman/small/segman_s_ade.py) | [Google Drive](https://drive.google.com/file/d/1VguPfxr_XSLWFuhopb0Ff-oA9QJGZMD7/view?usp=sharing) |
| SegMAN-B | Encoder-B (85.1) | 52.6 | 51.8M | 58.1G | [config](segmentation/local_configs/segman/base/segman_b_ade.py) | [Google Drive](https://drive.google.com/file/d/19C1lpTTqHZZvLdf4SbKcp8SMiIDPQcoO/view?usp=sharing) |
| SegMAN-L | Encoder-L (85.5) | 53.2 | 92.6M | 97.1G | [config](segmentation/local_configs/segman/large/segman_l_ade.py) | [Google Drive](https://drive.google.com/file/d/18OFmbr8rklYXqO93tU9UDYKsmobGFSR6/view?usp=sharing) |

### Cityscapes

| Model | Backbone | mIoU | Params | FLOPs | Config | Download |
|-------|----------|------|--------|-------|--------|----------|
| SegMAN-T | Encoder-T (76.2) | 80.3 | 6.4M | 52.5G | [config](segmentation/local_configs/segman/tiny/segman_t_cityscapes.py) | [Google Drive](https://drive.google.com/file/d/1GivXciIZ7hdDsY0IDvV-v1dejGCK2VLX/view?usp=sharing) |
| SegMAN-S | Encoder-S (84.0) | 83.2 | 29.4M | 218.4G | [config](segmentation/local_configs/segman/small/segman_s_cityscapes.py) | [Google Drive](https://drive.google.com/file/d/1VOpcMY9rTiHcx13nkFYLlX6llxAAZTEK/view?usp=sharing) |
| SegMAN-B | Encoder-B (85.1) | 83.8 | 51.8M | 479.0G | [config](segmentation/local_configs/segman/base/segman_b_cityscapes.py) | [Google Drive](https://drive.google.com/file/d/1k34JM9WVBYBIcCv8FOKvDUAHPDhjj00t/view?usp=sharing) |
| SegMAN-L | Encoder-L (85.5) | 84.2 | 92.6M | 769.0G | [config](segmentation/local_configs/segman/large/segman_l_cityscapes.py) | [Google Drive](https://drive.google.com/file/d/1SPaXL-faXlZyEPXl5bILLMlHG0OaFo1j/view?usp=sharing) |

### COCO-Stuff-164K

| Model | Backbone | mIoU | Params | FLOPs | Config | Download |
|-------|----------|------|--------|-------|--------|----------|
| SegMAN-T | Encoder-T (76.2) | 41.3 | 6.4M | 6.2G | [config](segmentation/local_configs/segman/tiny/segman_t_coco.py) | [Google Drive](https://drive.google.com/file/d/18P-e5hxWkISfiDZRnNTMow2-Fphk4H4t/view?usp=sharing) |
| SegMAN-S | Encoder-S (84.0) | 47.5 | 29.4M | 25.3G | [config](segmentation/local_configs/segman/small/segman_s_coco.py) | [Google Drive](https://drive.google.com/file/d/1LEa7PSs9H1yovjFp0Ylu-izqbje0LDDf/view?usp=sharing) |
| SegMAN-B | Encoder-B (85.1) | 48.4 | 51.8M | 58.1G | [config](segmentation/local_configs/segman/base/segman_b_coco.py) | [Google Drive](https://drive.google.com/file/d/1NHnNSBMQOw3y4FzjS66XcBrf-v5BpM0b/view?usp=sharing) |
| SegMAN-L | Encoder-L (85.5) | 48.8 | 92.6M | 97.1G | [config](segmentation/local_configs/segman/large/segman_l_coco.py) | [Google Drive](https://drive.google.com/file/d/18kVKvgZwESK-oixRpOjByg-97TWt8AA7/view?usp=sharing) |

主结果图见仓库 **`assets/`**（**`assets/model.png`**、**`assets/SegMAN_performance.png`** 等）。

---

## 4. 本环境二分类测试记录（DataA / DataB，六个方案运行）

以下数值来自在 **`segmentation/`** 下运行 **`tools/test.py`** 导出的  
`data/checkpoints2/test_20260428_15*/eval_single_scale_*.json`（单尺度；指标为 **0–1** 小数，`×100` 为百分比）。

### DataA

| 训练目录 | 配置（方案） | mIoU | IoU.fg | Acc.fg | Fscore.fg | Precision.fg | Recall.fg |
|------|------|------|------|------|------|------|------|
| `train_20260428_095310` | `segman_t_dataa_512_iou.py`（方案一：512 / argmax） | 0.7352 | 0.4792 | 0.9082 | 0.6479 | 0.5035 | 0.9082 |
| `train_20260428_101215` | `segman_t_dataa_256_scheme3.py`（方案三：256 / P>0.5） | 0.6518 | 0.3185 | 0.7933 | 0.4831 | 0.3473 | 0.7933 |
| `train_20260428_130151` | `segman_t_dataa_512_scheme2.py`（方案二：512 / P>0.55） | 0.7361 | 0.4805 | 0.8759 | 0.6491 | 0.5156 | 0.8759 |

### DataB

| 训练目录 | 配置（方案） | mIoU | IoU.fg | Acc.fg | Fscore.fg | Precision.fg | Recall.fg |
|------|------|------|------|------|------|------|------|
| `train_20260428_101242` | `segman_t_datab_512_iou.py`（方案一：512 / argmax） | 0.8648 | 0.7440 | 0.9443 | 0.8532 | 0.7781 | 0.9443 |
| `train_20260428_101253` | `segman_t_datab_512_scheme2.py`（方案二：512 / P>0.55） | 0.8659 | 0.7461 | 0.9455 | 0.8546 | 0.7796 | 0.9455 |
| `train_20260428_101306` | `segman_t_datab_256_scheme3.py`（方案三：256 / P>0.5） | 0.8072 | 0.6382 | 0.9456 | 0.7792 | 0.6625 | 0.9456 |

**测试命令**见 [`SegMAN.md`](SegMAN.md) §5（已支持“直接传 `train_*` 目录自动找 config + 权重”）。

---

## 5. 与课题内 BEVANet 的对照（简要）

| 项目 | BEVANet | 本仓库 SegMAN 二分类 |
|------|---------|----------------------|
| 入口 | `tools/wire.py` + yacs | **`segmentation/tools/train.py`** + mmcv config |
| 训练产物根 | `data/checkpoints1/train_*` | 同上（工程根下） |
| 测试产物根 | `data/checkpoints2/test_*` | 同上 |

口径上均强调 **前景 IoU 与 mIoU(mean) 勿混读**；SegMAN 侧由 **`SegmanEvalHook`** 与配置中的 **`evaluation.save_best`** 体现。

---

## 6. 引用

```bibtex
@inproceedings{SegMAN,
    title={SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation},
    author={Yunxiang Fu and Meng Lou and Yizhou Yu},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```
