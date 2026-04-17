#!/usr/bin/env bash
# DataA 二分类，SegMAN-T，IoU best + IoU patience 早停（配置内 load_from=segman_t_ade.pth）
# 用法：在「任意目录」执行 bash segmentation/scripts/train_dataa_segman_t_iou.sh
# 或：cd segmentation && bash scripts/train_dataa_segman_t_iou.sh
set -euo pipefail
SEG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SEG_ROOT"
ADE_CKPT="pretrained/segman_t_ade.pth"
if [[ ! -f "$ADE_CKPT" ]]; then
  echo "错误：缺少 $SEG_ROOT/$ADE_CKPT" >&2
  echo "请从 README Google Drive 下载 segman_t_ade.pth 放到 segmentation/pretrained/。" >&2
  exit 1
fi
MIN_BYTES=$((65 * 1024 * 1024))
SZ=$(stat -c%s "$ADE_CKPT" 2>/dev/null || stat -f%z "$ADE_CKPT" 2>/dev/null)
if [[ "$SZ" -lt "$MIN_BYTES" ]]; then
  echo "错误：$SEG_ROOT/$ADE_CKPT 仅 ${SZ} 字节，完整文件约 74MiB。请删除后重新下载。" >&2
  exit 1
fi
echo "校验 ADE 整网预训练权重可读..."
python -c "import torch; torch.load('$ADE_CKPT', map_location='cpu'); print('OK: segman_t_ade.pth 可被 PyTorch 读取')" || {
  echo "错误：segman_t_ade.pth 无法解析（损坏或非 PyTorch 权重）。请重新下载。" >&2
  exit 1
}
exec python tools/train.py local_configs/segman_binary/segman_t_dataa_512_iou.py "$@"
