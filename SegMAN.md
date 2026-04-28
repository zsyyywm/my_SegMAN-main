# SegMAN — 训练 / 测试命令速查

> **角色**：本文只有 **可直接复制的命令** 与极简说明。  
> **项目综述**（任务、目录、指标含义）见 `[README.md](README.md)`。  
> **环境安装、数据路径、预训练下载**见 `[SETUP.md](SETUP.md)`。

论文：[SegMAN (CVPR 2025)](https://arxiv.org/abs/2412.11890)

**重要**：文档里的 `**/path/to/...` 是占位符，不能原样复制**。必须改成你机器上 **工程根** 的**真实绝对路径**（下面「AutoDL 本机」已写好当前常见路径）。`**conda activate segman`** 只有在你用 `conda create -n segman` 建了**同名**环境时才有效；本仓库在 AutoDL 上常用的是 **路径型**环境 `**/root/autodl-tmp/conda_envs/segman`**。

---

## 1. 工程根与工作目录

工程根指内含 `**segmentation/**`、`**kernels/**`、`**requirements.txt**` 的目录。以下**每次开新终端**建议先执行其一整段，再跑后面各节的训练/测试命令。

**AutoDL / 数据盘在 `~/autodl-tmp` 时（可直接复制）**

```bash
export REPO=/root/autodl-tmp/my_MambaVision-main/my_SegMAN-main-main/my_SegMAN-main-main
conda activate /root/autodl-tmp/conda_envs/segman
# 若上一行报错，可试：source /root/autodl-tmp/conda_envs/segman/bin/activate
cd "$REPO/segmentation"
pwd   # 应看到 .../my_SegMAN-main-main/segmentation
```

**其他机器（务必自行改路径）**

```bash
export REPO=你的工程根绝对路径   # 例：/home/you/projects/my_SegMAN-main-main

conda activate 你的环境名          # 例：segman；或用 conda activate /完整路径/到/env
cd "$REPO/segmentation"
```

**二分类训练与测试必须在 `segmentation/` 下执行**，否则相对路径（数据、`pretrained/`、`load_from`）可能错误。

**后面第 3 节起**：若已在本节设置好 `**REPO`** 并 `**cd "$REPO/segmentation"**`，命令里只写 `python tools/...` 即可；若新开终端，请先重复本节。

---

## 2. 产物目录（二分类 `segman_use_binary_ckpt_layout`）


| 路径（相对 **工程根** `$REPO`）                           | 说明                                                            |
| ------------------------------------------------ | ------------------------------------------------------------- |
| `**data/checkpoints1/train_<YYYYMMDD_HHMMSS>/`** | 训练：日志、config 快照、TensorBoard、`best_*.pth`、`checkpoint` 等。      |
| `**data/checkpoints2/test_<YYYYMMDD_HHMMSS>/**`  | 测试：默认写入 `**eval_single_scale_*.json**` 等（未指定 `--work-dir` 时）。 |


自定义输出目录：训练加 `**--work-dir /绝对路径**`；测试同样。

---

## 3. 二分类训练（DataA）

配置位于 `**local_configs/segman_binary/**`。与 BEVANet 三方案对齐：


| 方案    | 输入      | 验证解码          | 配置                              |
| ----- | ------- | ------------- | ------------------------------- |
| **一** | 512×512 | **argmax**    | `segman_t_dataa_512_iou.py`     |
| **二** | 512×512 | 前景 **P>0.55** | `segman_t_dataa_512_scheme2.py` |
| **三** | 256×256 | 前景 **P>0.5**  | `segman_t_dataa_256_scheme3.py` |


以上三者 **互斥**；另可选「同训双指标」见下节。

**方案一（512，argmax + IoU 选优 / 早停）**

```bash
# 已在 §1 中 cd 到 segmentation/ 的前提下：
python tools/train.py local_configs/segman_binary/segman_t_dataa_512_iou.py
```

**方案二（512，固定 512；验证阈值 0.55）**

```bash
cd "$REPO/segmentation"
python tools/train.py local_configs/segman_binary/segman_t_dataa_512_scheme2.py
```

**方案三（256，固定 256；验证阈值 0.5）**

```bash
cd "$REPO/segmentation"
python tools/train.py local_configs/segman_binary/segman_t_dataa_256_scheme3.py
```

**可选：512 同训 argmax + 0.55 双路指标（单卡，与 BEVA「双路」类似但一次前向两套解码）**

```bash
cd "$REPO/segmentation"
python tools/train.py local_configs/segman_binary/segman_t_dataa_512_wire_dual.py
```

**可选：TransNeXt 风格 IoU best + val loss 双轨（`data/checkpoints2`，非上表「方案二」）**

```bash
cd "$REPO/segmentation"
python tools/train.py local_configs/segman_binary/segman_t_dataa_512_val_loss.py
```

---

## 4. 二分类训练（DataB）

将配置名中的 `**dataa**` 改为 `**datab**`：

```bash
cd "$REPO/segmentation"
python tools/train.py local_configs/segman_binary/segman_t_datab_512_iou.py
python tools/train.py local_configs/segman_binary/segman_t_datab_512_scheme2.py
python tools/train.py local_configs/segman_binary/segman_t_datab_256_scheme3.py
python tools/train.py local_configs/segman_binary/segman_t_datab_512_val_loss.py
```

---

## 5. 二分类测试（`tools/test.py`）

将 `**CHECKPOINT**` 换成训练得到的 `**best_*.pth**` 或 `**iter_*.pth**` 的绝对路径；`**CONFIG**` 与训练时 **一致**。  
本仓库 `tools/test.py` 默认会评估 `mIoU + mFscore`，因此会直接输出每类 **IoU / Acc / Fscore / Precision / Recall**（无需再手动加 `--eval`）。

也支持**只给训练目录**（会自动在目录内找 config 与最佳权重）：

```bash
cd "$REPO/segmentation"
python tools/test.py \
  local_configs/segman_binary/segman_t_dataa_512_iou.py \
  --checkpoint "$CHECKPOINT"
```

```bash
cd "$REPO/segmentation"
python tools/test.py \
  "$REPO/data/checkpoints1/train_20260428_095310"
```

按当前课题常用的 6 个 run，可直接批量复制：

```bash
cd "$REPO/segmentation"
python tools/test.py "$REPO/data/checkpoints1/train_20260428_095310"
python tools/test.py "$REPO/data/checkpoints1/train_20260428_101215"
python tools/test.py "$REPO/data/checkpoints1/train_20260428_101242"
python tools/test.py "$REPO/data/checkpoints1/train_20260428_101253"
python tools/test.py "$REPO/data/checkpoints1/train_20260428_101306"
python tools/test.py "$REPO/data/checkpoints1/train_20260428_130151"
```

默认评测结果 JSON 写入 `**$REPO/data/checkpoints2/test_<时间戳>/**`。指定目录示例：

```bash
python tools/test.py \
  local_configs/segman_binary/segman_t_datab_512_iou.py \
  --checkpoint "$CHECKPOINT" \
  --work-dir "$REPO/data/checkpoints2/my_eval_run"
```

多卡：使用 `**tools/dist_test.sh**`（参数以 MMSeg 文档为准）。

---

## 6. 官方数据集训练 / 测试（ADE20K 等）

需先完成 `[SETUP.md](SETUP.md)` 中的 MMSeg、NATTEN、selective_scan 与 `**pip install -e segmentation/**`。

**训练示例（SegMAN-B | ADE20K，单卡）**

```bash
cd "$REPO/segmentation"
python tools/train.py local_configs/segman/base/segman_b_ade.py --work-dir outputs/EXP_NAME
```

**多卡**

```bash
bash tools/dist_train.sh local_configs/segman/base/segman_b_ade.py <GPU_NUM> --work-dir outputs/EXP_NAME
```

**测试示例**

```bash
cd "$REPO/segmentation"
python tools/test.py local_configs/segman/base/segman_b_ade.py /path/to/checkpoint.pth
```

**多卡测试**

```bash
bash tools/dist_test.sh local_configs/segman/base/segman_b_ade.py /path/to/checkpoint.pth <GPU_NUM>
```

论文脚本目录：`**segmentation/scripts/**`（如 `scripts/train_segman-s.sh` 等，按文件名选用）。

---

## 7. 可视化

若本地已包含 MMSeg 官方 `**demo/image_demo.py**`，可在 `**segmentation/**` 下按 [get_started](segmentation/docs/zh_cn/get_started.md) 中的示例调用；参数为 **图片路径、config、checkpoint、palette、输出路径**。

---

## 8. ImageNet 编码器预训练（可选）

仓库根目录 `**train.py`** 与 `**scripts/train_*.sh**` 用于 **ImageNet 分类预训练**（非 `segmentation/tools/train.py`）。使用前请编辑脚本内 `**--data-dir`**、`**--output**` 等路径。

```bash
cd "$REPO"
bash scripts/train_small.sh
```

---

## 9. 引用

```bibtex
@inproceedings{SegMAN,
    title={SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation},
    author={Yunxiang Fu and Meng Lou and Yizhou Yu},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```

