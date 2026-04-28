# 环境与依赖（SETUP）

> **角色**：从零安装 Conda / PyTorch / MMSeg / NATTEN / selective_scan、放置预训练权重、确认数据路径。**不写长训练命令**（见根目录 [`SegMAN.md`](SegMAN.md)）。  
> **项目综述**（任务、目录、`data/checkpoints1/2` 含义）见 [`README.md`](README.md)。

---

## 工程根目录

下文「**工程根**」指内含 **`segmentation/`**、**`kernels/`**、**`requirements.txt`** 的目录。**不要**使用字面量 **`/path/to/...`**，应换成你的**真实绝对路径**。AutoDL 数据盘上常见为：

```bash
export REPO=/root/autodl-tmp/my_MambaVision-main/my_SegMAN-main-main/my_SegMAN-main-main
```

多数训练与测试命令在 **`$REPO/segmentation/`** 下执行（`tools/train.py` 会解析仓库根并写入 `data/checkpoints*`）。

---

## 云服务器与数据盘（以 AutoDL 为例）

在 **AutoDL** 等平台上，**数据盘**通常挂载在 **`/root/autodl-tmp`**（工作区、大文件、自建 Conda 环境多放此处，避免占满系统盘 **`/`**）。

**当前实例上已存在的路径**（便于你对照；换机后请以控制台「数据盘」实际挂载点为准）：

| 路径 | 说明 |
|------|------|
| `/root/autodl-tmp/my_MambaVision-main/` | 课题相关**代码**（含本仓库 `.../my_SegMAN-main-main/my_SegMAN-main-main/`）。 |
| `/root/autodl-tmp/conda_envs/segman` | **SegMAN / MMSeg** 用 Conda 环境（体积较大，已在数据盘）。 |
| `/root/autodl-tmp/conda_envs/bevanet` | **BEVANet** 对照实验环境。 |
| `/root/autodl-tmp/conda_pkgs` | Conda / pip 包缓存，减轻系统盘压力。 |

**一键对齐本机 `REPO` + 激活 SegMAN 环境（可复制）**：

```bash
export REPO=/root/autodl-tmp/my_MambaVision-main/my_SegMAN-main-main/my_SegMAN-main-main
conda activate /root/autodl-tmp/conda_envs/segman
# 若上式无效，可改用：
# source /root/autodl-tmp/conda_envs/segman/bin/activate
```

若你的平台把数据盘挂在 **`/data`** 等其他目录，只需把上述 **`/root/autodl-tmp`** 换成控制台说明的**数据盘根路径**，并相应修改 `export REPO=...`。

---

## 1. Conda 与 Python

若 **已在数据盘上建好** `conda_envs/segman` 等，可**跳过** `conda create`，直接 `conda activate` 该路径即可。

```bash
conda create -n segman python=3.10 -y
conda activate segman

pip install torch==2.1.2 torchvision==0.16.2
```

请按本机 **CUDA 版本** 从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择匹配的 `torch` / `torchvision` 安装命令（上式仅为示例）。

---

## 2. MMSegmentation v0.30.0 与 mmcv

按 [MMSeg v0.30.0 安装说明](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/get_started.md) 准备环境。下列组合在作者环境中可用：

```bash
pip install -U openmim
mim install mmcv-full
cd "$REPO/segmentation"
pip install -v -e .
```

若使用 **torch ≥ 2.1.0**，可能需修改 conda 环境中 **`mmcv/parallel/_functions.py`**（约第 75 行附近）以适配 stream 获取方式，例如：

```python
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.1.0'):
    streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]
else:
    streams = [_get_stream(device) for device in target_gpus]
```

具体行号以你安装的 mmcv 版本为准。

---

## 3. NATTEN 与 Selective Scan

**NATTEN**：按 [NATTEN 发布页](https://github.com/SHI-Labs/NATTEN) 选择与 **PyTorch + CUDA** 匹配的 wheel，例如：

```bash
pip install natten==0.17.3+torch210cu121 -f https://shi-labs.com/natten/wheels/
```

**Selective Scan 2D**（VMamba 系）：

```bash
cd "$REPO/kernels/selective_scan"
pip install .
cd "$REPO"
```

需本机可编译 CUDA 扩展。

---

## 4. 其余 Python 依赖

```bash
pip install -r "$REPO/requirements.txt"
```

二分类与部分脚本建议 **NumPy 1.x**；若遇兼容问题可固定 `numpy<2`。

---

## 5. 数据（DataA / DataB）

默认在 **`segmentation/`** 目录下启动训练时，数据根为相对路径 **`../../../DataA-B/DataA`**（及 DataB 对应配置），即与工程根上数级目录同级的 **`DataA-B`**（常见布局：`my_MambaVision-main/DataA-B/DataA`、`.../DataB`），目录内需有 **`image/{train,val}`**、**`mask/{train,val}`**，mask 为 **0/1**。

自定义路径：

```bash
export SEG_DATAA_ROOT=/绝对路径/DataA
export SEG_DATAB_ROOT=/绝对路径/DataB
```

（以各 `_base_/datasets/*.py` 中环境变量名为准。）

---

## 6. 预训练与论文权重

### 6.1 二分类配置常用：`segman_t_ade.pth`

`segmentation/local_configs/segman_binary/*.py` 中 **`load_from = 'pretrained/segman_t_ade.pth'`** 表示 **ADE20K 上训练好的 SegMAN-T 整网**（含解码器；加载时 `strict=False`，二分类头不兼容层会跳过）。

- **Google Drive（SegMAN-T | ADE20K）**：  
  https://drive.google.com/file/d/1d0wp7C83YjImeQmL5_CIo1qMRjPd_-8a/view?usp=sharing  

下载后放到 **`$REPO/segmentation/pretrained/`**，并命名为 **`segman_t_ade.pth`**。

命令行示例（需已安装 [gdown](https://github.com/wkentaro/gdown)）：

```bash
mkdir -p "$REPO/segmentation/pretrained"
gdown --fuzzy 'https://drive.google.com/file/d/1d0wp7C83YjImeQmL5_CIo1qMRjPd_-8a/view?usp=sharing' \
  -O "$REPO/segmentation/pretrained/segman_t_ade.pth"
```

### 6.2 ImageNet 骨干（整网训练 / 论文复现）

- 文件夹：[Google Drive（ImageNet-1k 预训练）](https://drive.google.com/drive/folders/1QYU7nhpe0ddH7bPxI7VH4drc__07uEHs?usp=sharing)  
  将所需文件放入 **`segmentation/pretrained/`**（或按各 `local_configs/segman/*/*.py` 中路径修改）。

### 6.3 官方分割训练权重汇总

- [Google Drive（分割权重汇总）](https://drive.google.com/drive/folders/1C2bmb7KP7mECm9c04NCrUAJQGsEf_bQ4?usp=sharing)

各模型与数据集对应关系仍以 **`README.md`** 中表格及 **`local_configs/segman/`** 下 config 为准。

---

## 7. 可选：绘图与日志

若需训练结束自动保存曲线图等，请安装 **matplotlib**（部分环境默认未装）：

```bash
pip install matplotlib
```

---

## 8. 自检

```bash
conda activate segman
cd "$REPO/segmentation"
python -c "import mmseg; import mmcv; print('mmseg/mmcv OK')"
```

通过后再按 [`SegMAN.md`](SegMAN.md) 执行训练或测试。
