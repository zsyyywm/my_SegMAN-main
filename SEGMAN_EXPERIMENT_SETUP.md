# SegMAN 对比实验环境（DataA / DataB，checkpoints1 / checkpoints2）

本文档对应 **`SegMAN-main/SegMAN-main/segmentation`**（在 `segmentation` 目录下跑命令）。**不修改** `TransNeXt-main`。

## 与 TransNeXt（mask2former）如何一一对应

| TransNeXt 做法 | SegMAN 本仓库 |
|----------------|----------------|
| `configs/*_iou.py` + `work_dir` → `data/checkpoints1` | `local_configs/segman_binary/*_512_iou.py` + `segman_checkpoint_family='checkpoints1'` |
| 仅按 **val 前景 IoU** 存 best | `evaluation.save_best='IoU.foreground'`（mmcv `EvalHook` 写入主 `train_*` 目录） |
| `configs/*_val_loss.py` + `checkpoints2` | `*_512_val_loss.py` + `segman_checkpoint_family='checkpoints2'` |
| `mask2former_enable_val_loss_best=True` | **`segman_enable_val_loss_best=True`**，或在配置里写同名 **`mask2former_enable_val_loss_best=True`**（`tools/train.py` 会自动映射） |
| IoU best 仍在 `checkpoints2/train_*` | 仍在 **`data/checkpoints2/train_<时间>/`**（EvalHook） |
| loss best 在 `val_loss_best/<同一时间>` | **`data/checkpoints2/val_loss_best/train_<时间>/best_val_loss_iter_*.pth`**（SegmanEvalHook） |

**想先练「按 loss 最优也存一份权重」**：请直接跑 **`*_512_val_loss.py`**（不是 `*_iou.py`）。该配置下 **IoU best 与 loss best 两种都会有**——与 TransNeXt 双轨一致；若只要 IoU、不要 loss 目录，用 **`*_512_iou.py`**。

| 配置文件模式 | `segman_checkpoint_family` | 说明 |
|-------------|---------------------------|------|
| `*_512_iou.py` | `checkpoints1` | 仅 IoU：`data/checkpoints1/train_<时间>/` |
| `*_512_val_loss.py` | `checkpoints2` | IoU + loss：`train_<时间>/` + `val_loss_best/train_<时间>/` |

---

## 第一步：新建 Conda 环境（建议 Python 3.10）

```powershell
conda create -n segman python=3.10 -y
conda activate segman
```

## 第二步：安装 PyTorch（按你本机 CUDA 版本调整 cu121/cu118/cpu）

README 示例为 CUDA 12.1：

```powershell
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

## 第三步：安装 MMCV 与 MMSegmentation 0.30

按 SegMAN 根目录 [README.md](README.md) 的说明，整体顺序为：**先固定 NumPy 1.x，再装 mmcv，再进 `segmentation` 装 mmseg**。

### 3.1 安装前必做（Windows 尤其容易忽略）

1. **NumPy 必须为 1.x**（不要默认装到 NumPy 2.x）。PyTorch 2.1 及不少二进制扩展仍按 NumPy 1.x ABI 编译；若环境里是 NumPy 2.x，会出现：
   - `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x ...`
   - `Failed to initialize NumPy: _ARRAY_API not found`（往往在 `import torch` 时触发）  
   进而导致 **`mim install mmcv-full` 在内部 `import torch` 时就失败**。

   在已装好 PyTorch 后执行：

   ```powershell
   pip install -U pip setuptools wheel
   pip install "numpy>=1.23,<2"
   python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"
   ```

   上述一行应打印出版本号且无异常，再继续下面步骤。

2. **PyTorch 与 CUDA 主版本**须与后续 mmcv 轮子索引一致。本文档第二步示例为 **cu121 + torch 2.1.2**；若你使用 **cu118**，下文中所有 `cu121` 请改为 `cu118`，并到 [MMCV 安装说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 核对 **torch 次版本** 对应的 `torch2.1.0` 索引表是否仍适用。

### 3.2 推荐：openmim 安装 mmcv-full

```powershell
pip install -U openmim
mim install mmcv-full
```

成功后：

```powershell
cd segmentation
pip install -v -e .
```

若使用 **torch>=2.1**，README 要求修改 mmcv 的 `_functions.py` 一处（见原 README **Step 2**）。

### 3.3 Windows 常见问题与替代安装方式

#### 问题 A：NumPy 2.x 与 PyTorch / mim

**现象**：安装或运行 mim 时出现 NumPy 1.x/2.x 不兼容提示，或 `_ARRAY_API not found`。

**处理**：按 **3.1** 执行 `pip install "numpy>=1.23,<2"`，**不要**在 segman 环境里长期保留 NumPy 2.x 再装 mmcv。

#### 问题 B：`mim install mmcv-full` 或 pip 解析依赖时出现 `json.decoder.JSONDecodeError`（如 `Unterminated string`）

**现象**：日志里往往在收集 **`opencv-python`** 等依赖时，在 `parse_links` / `json.loads` 处崩溃。

**原因**：pip 从 PyPI 拉取的索引 JSON 不完整或异常（网络、代理、防火墙、偶发 PyPI 问题等），与 mmcv 本身无关。

**处理**（建议按顺序尝试）：

1. 再次确认：`python -m pip install -U pip`，并已执行 **3.1** 的 NumPy 1.x 固定。
2. **先把 mmcv 常用依赖装好**，减少 resolver 去 PyPI「大海捞针」：

   ```powershell
   pip install opencv-python pyyaml packaging addict yapf "Pillow>=6.2.0"
   ```

   若 **`pip install opencv-python` 仍触发同类 JSON 错误**，可改用 conda 安装 OpenCV（不经 PyPI 那份 JSON）：

   ```powershell
   conda install -c conda-forge opencv -y
   ```

3. **绕过 mim，直接用 pip + OpenMMLab 官方轮子页**（**CUDA 主版本 + PyTorch 2.1.x** 须与当前环境一致；与 mim 日志里出现的链接保持一致即可）。示例（**cu121 + torch 2.1.x**）：

   ```powershell
   pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
   ```

   若 pip 仍异常，可加 `--no-cache-dir` 重试。`mmcv-full` 的具体版本号以你机器上 **Python 版本、CUDA、Torch** 在 [MMCV 安装说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 中可查到的 **win 预编译包**为准；上式为与第二步 **torch 2.1 + cu121** 常见组合一致的写法。

4. mmcv-full 安装成功后，照旧执行：

   ```powershell
   cd segmentation
   pip install -v -e .
   ```

## 第四步：Natten 与 selective_scan（Mamba/SSM 核）

SegMAN 骨干代码会 **`import natten`**（见 `segmentation/mmseg/models/backbones/segman_encoder.py`），**Natten 必须先装好**；再编译安装 **`kernels/selective_scan`**。两步都与 **「含 `kernels/` 的 SegMAN 根目录」** 无关：`pip install natten` 可在任意目录执行；**只有** `pip install .`（selective_scan）前需要 `cd` 到 `kernels\selective_scan`。

### 4.1 确认根目录（仅 selective_scan 需要）

在资源管理器或终端中进入 **同时包含 `kernels` 与 `segmentation` 的那一层**，例如：

`...\SegMAN-main\SegMAN-main\`（以你本机实际路径为准）。

### 4.2 安装 Natten（与第二步 PyTorch / CUDA 一致）

**Torch 2.1.x + CUDA 12.1** 示例（与文档第二步一致）：

```powershell
pip install natten==0.17.3+torch210cu121 -f https://shi-labs.com/natten/wheels/
```

版本后缀须与 [Natten wheels](https://shi-labs.com/natten/wheels/) 上存在的组合一致（如 `torch210cu118` 对应 cu118）。

#### 卡住常见原因 A：`SSLCertVerificationError`（证书不在有效期内等）

1. **校对 Windows 系统时间**（自动设置时间打开），重开终端再装。  
2. 换 **手机热点** 或关闭公司代理 / 杀毒「HTTPS 扫描」后再试。  
3. **权宜之计**（降低校验严格度，自行承担风险）：  

   ```powershell
   pip install natten==0.17.3+torch210cu121 -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com --trusted-host files.pythonhosted.org
   ```

4. 用浏览器打开 wheels 页，若可下载 **与本机 Python 位数、torch/cu 匹配的 `.whl`**，再：  

   ```powershell
   pip install C:\你的下载路径\xxx.whl
   ```

#### 卡住常见原因 B：Windows 上「没有可用轮子」

Natten 官方页 **常为 Linux 预编译**；若 pip 提示 **找不到满足条件的发行版**，在 **原生 Windows** 上可能无法靠一条命令解决，需要：

- 使用 **WSL2（Ubuntu）+ GPU**，在同一套 torch/cu 环境下安装 Natten，并在 WSL 内完成 SegMAN 训练；或  
- 使用 **Linux 远程环境**。

### 4.3 安装 selective_scan（必须在 `kernels\selective_scan`）

在 **4.1 的根目录** 下执行：

```powershell
cd kernels\selective_scan
pip install .
cd ..\..
```

说明：`setup.py` 里标注为 **CUDA 扩展**，且分类为 **Unix**；在 Windows 上若 `pip install .` 失败，多为 **本机缺少可用的 MSVC + CUDA/nvcc 与 PyTorch 一致** 或 **官方未在 Win 上测试**。此时同样建议 **WSL2 / Linux** 完成编译与训练。

### 4.4 自检

```powershell
python -c "import natten; print('natten', natten.__version__)"
python -c "import selective_scan_cuda_core; print('selective_scan ok')"
```

若第二行导入名因编译模式不同而变化，以 `pip install .` 成功日志中的扩展名为准；训练能正常启动即可。

## 第五步：其余依赖

```powershell
pip install -r requirements.txt
```

## 第六步：预训练骨干

从 README 的 Google Drive 下载 **SegMAN Encoder-T**（或你选用的尺度），放到：

`segmentation/pretrained/SegMAN_Encoder_t.pth.tar`

并在配置里保持：

`model.backbone.pretrained='pretrained/SegMAN_Encoder_t.pth.tar'`

（相对 **`segmentation`** 当前工作目录。）

## 第七步：数据集路径

配置中 `data_root` 为相对 **`segmentation`** 的 `../../../../DataA-B/DataA`（或 DataB），与 TransNeXt `mask2former` 下约定一致：请保证 **`DataA-B` 与 `mamba-main` 同级**（都在「大二下学习」下）；若你数据盘不同，请改：

`segmentation/local_configs/_base_/datasets/dataa_binary_512.py`  
`segmentation/local_configs/_base_/datasets/datab_binary_512.py`

中的 `_data_root`。

## 第八步：训练命令（必须在 `segmentation` 目录）

```powershell
cd C:\Users\34977\Desktop\大二下学习\mamba-main\SegMAN-main\SegMAN-main\segmentation

# DataA — 仅 IoU best（checkpoints1）
python tools/train.py local_configs/segman_binary/segman_t_dataa_512_iou.py

# DataA — IoU + val loss 双 best（checkpoints2）
python tools/train.py local_configs/segman_binary/segman_t_dataa_512_val_loss.py

# DataB
python tools/train.py local_configs/segman_binary/segman_t_datab_512_iou.py
python tools/train.py local_configs/segman_binary/segman_t_datab_512_val_loss.py
```

可选：指定工作目录（将关闭自动 `train_*` 子目录逻辑的部分便利，一般不推荐）：

```powershell
python tools/train.py local_configs/segman_binary/segman_t_dataa_512_iou.py --work-dir D:\runs\segman_a1
```

---

## 与 TransNeXt 的差异（写论文时请一句带过）

- 此处为 **SegMAN 编码器 + SegMANDecoder**（CE），不是 Mask2Former；属于 **另一套分割头** 的对比，而非「仅换骨干」。
- 选优指标在 mmcv 下使用 **`IoU.foreground`**（由 `CustomDataset.evaluate` 的 per-class IoU 键名提供）。
