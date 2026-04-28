# SegMAN 二分类线材三方案（与 MambaVision `semantic_segmentation` 中 WIRE 约定一致）

> **日常可复制命令**（DataA/DataB、三方案 / val_loss / wire_dual / 测试）见 **工程根** [`SegMAN.md`](../SegMAN.md)；**环境**见 [`SETUP.md`](../SETUP.md)；**项目综述**见 [`README.md`](../README.md)。本文档只说明 **Hook 与实现索引**。

## 实现位置

- `mmseg/apis/wire_test.py`：单卡下从 `EncoderDecoder.inference` 的 **softmax** 上生成 argmax 与 **P(fg)&gt;阈值** 预测，再走 `pre_eval`。
- `mmseg/core/evaluation/segman_eval_hooks.py`：`SegmanWireDualEvalHook`（同训双路）；`SegmanWireScheme3EvalHook`（**方案二 512/th0.55** 与 **方案三 256/th0.5** 共用，由 `segman_wire_th` 区分）。
- `mmseg/apis/train.py`：根据 `segman_wire_dual_512` / `segman_wire_scheme2_512` / `segman_wire_scheme3_256` 三选一注册 Hook；**多卡**时暂回退为 `SegmanEvalHook` 并告警。

## 方案（与 BEVANet 编号对齐）

| 方案 | 训练配置 | 验证解码 | 存盘要点 |
|------|----------|----------|----------|
| **一** | `segman_t_*_512_iou.py` | **argmax** | `save_best='val/IoU'`（`SegmanEvalHook`） |
| **二** | `segman_t_*_512_scheme2.py` | **P(fg)&gt;0.55** | 同上 Hook 类，``th=0.55``，主目录 `save_best` |
| **三** | `segman_t_dataa_256_scheme3.py` | **P(fg)&gt;0.5** | 同上，``th=0.5`` |
| **可选：同训** | `segman_t_dataa_512_wire_dual.py` | 1=argmax；2=P(fg)&gt;0.55 | `save_best` 盯方案1；方案2 权重在 `scheme2_t055_512/` |

- **早停**（`segman_iou_early_stop_patience`）：`tools/train.py` 注入的 `SegmanIoUPatienceEarlyStopHook` 使用 `evaluation.save_best` 的键。**双路同训**下 `save_best` 仍为方案 1 的 `val/IoU`。

**产物目录（二分类 `segman_use_binary_ckpt_layout`）**：训练默认写入仓库根下 **`data/checkpoints1/train_<时间戳>/`**；`tools/test.py` 未指定 `--work-dir` 时写入 **`data/checkpoints2/test_<时间戳>/`**（与 BEVANet `1.md` 约定一致）。可用配置键 `segman_binary_ckpt_root` / `segman_binary_eval_root` 覆盖。

## 训练日志与双路指标

- **测试**：`tools/test.py` + 与训练一致的 config + `checkpoint`；双路训练时可在单卡日志中对照 **`val/IoU`** 与 **`val_t055/IoU`**（详见工程根 **`SegMAN.md`**）。
- **训练时**：每个验证周期写入 **`work_dir`** 下日志与 `*.log.json`（键名以实际 Hook 为准）。

## 可选项（config）

- `segman_wire_subdir1` / `segman_wire_subdir2`：子目录名（仅双路）。
- `segman_wire_fg`：前景类下标，默认 1（background=0, foreground=1）。
- `segman_wire_th`：方案二默认 **0.55**；方案三默认 **0.5**；双路同训里为第二路阈值。

阈值验证与同训均要求验证集 **`samples_per_gpu=1`**（`wire_test` 中已约束）。
