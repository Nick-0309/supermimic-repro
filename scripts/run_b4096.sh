#!/usr/bin/env bash
set -uo pipefail
cd "${SM_ROOT:-$HOME/supermimic}" || exit 1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

RUN=lafan1_b4096_lr
TOTAL=320000
EVAL_ITS="12000 20000 28000"
LOGD=logs
mkdir -p "$LOGD/$RUN"
banner(){ echo; echo "=============== $* ==============="; }

banner "0/4 预检"
FAIL=0
gpu_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "GPU 显存占用: ${gpu_mib:-?} MiB"
[ "${gpu_mib:-99999}" -lt 2000 ] || { echo "FAIL: GPU 被占用"; FAIL=1; }
avail_g=$(df -BG --output=avail "${DATA_ROOT:-$PWD}" | tail -1 | tr -dc '0-9')
echo "数据盘可用: ${avail_g:-0} G"
[ "${avail_g:-0}" -ge 8 ] || { echo "FAIL: 数据盘不足 8G"; FAIL=1; }
n_npz=$(ls data/npz/lafan1/*.npz 2>/dev/null | wc -l)
echo "LAFAN1 npz: ${n_npz} 条"
[ "$n_npz" -eq 40 ] || { echo "FAIL: npz 不是 40 条"; FAIL=1; }
pgrep -f "sm-train" >/dev/null && { echo "FAIL: 已有 sm-train 在跑"; FAIL=1; }
[ "$FAIL" -eq 0 ] || { echo "预检失败，终止"; exit 1; }
echo "OK: 预检通过"

banner "1/4 探针: batch=4096, 100 迭代"
rm -rf "$LOGD/${RUN}_probe"
uv run --no-sync sm-train --run-name "${RUN}_probe" --data.npz-dir data/npz/lafan1 \
  --env.num-envs 4096 --policy-trainer.batch-size 4096 \
  --schedule.total-iterations 100 --schedule.warmup-iterations 20 \
  --schedule.log-interval 50 --schedule.checkpoint-interval 100 \
  --schedule.max-hours 0.5 > "$LOGD/${RUN}_probe.log" 2>&1 &
PROBE_PID=$!
PEAK=0
while kill -0 $PROBE_PID 2>/dev/null; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "${m:-0}" -gt "$PEAK" ] && PEAK=$m
  sleep 2
done
wait $PROBE_PID; rc=$?
echo "探针退出码=$rc  峰值显存=${PEAK} MiB / 32607 MiB"
tail -3 "$LOGD/${RUN}_probe.log"
grep -qiE "out of memory|OutOfMemory|CUDA error" "$LOGD/${RUN}_probe.log" && { echo "FAIL: OOM"; exit 1; }
[ "$rc" -eq 0 ] || { echo "FAIL: 探针失败"; exit 1; }
[ "$PEAK" -lt 28000 ] || { echo "FAIL: 峰值显存过高"; exit 1; }
echo "OK: 探针通过"
rm -rf "$LOGD/${RUN}_probe"

banner "2/4 正式训练 28k"
uv run --no-sync sm-train --run-name "$RUN" --data.npz-dir data/npz/lafan1 \
  --env.num-envs 4096 --policy-trainer.batch-size 4096 \
  --schedule.total-iterations "$TOTAL" \
  --schedule.checkpoint-interval 4000 \
  --schedule.log-interval 100 \
  --schedule.max-hours 3.6 > "$LOGD/${RUN}.train.log" 2>&1
rc=$?
echo "训练退出码=$rc"
[ "$rc" -eq 0 ] || { echo "FAIL: 训练失败"; exit 1; }
tail -3 "$LOGD/${RUN}.train.log"

banner "3/4 自动评测 12k/20k/28k"
for it in $EVAL_ITS; do
  ck=$(printf "%s/%s/ckpt_%07d.pt" "$LOGD" "$RUN" "$it")
  [ -f "$ck" ] || { echo "跳过 $it"; continue; }
  uv run --no-sync python scripts/eval_benchmark.py --ckpt "$ck" \
    --npz-dir data/npz/lafan1 --num-envs 1024 --device cuda:0 \
    --out "$LOGD/$RUN/bench_${it}.json" > "$LOGD/$RUN/bench_${it}.log" 2>&1
  uv run --no-sync python scripts/diag_drift.py --ckpt "$ck" \
    --npz-dir data/npz/lafan1 --device cuda:0 \
    --out "$LOGD/$RUN/drift_${it}.json" > "$LOGD/$RUN/drift_${it}.log" 2>&1
  echo "完成 $it"
done

banner "4/4 汇总"
python - "$RUN" <<'PY'
import json, pathlib, sys
run = sys.argv[1]; root = pathlib.Path("logs")/run
print(f"{'iter':>6} | {'ret':>9} | {'failed':>9}")
for it in (12000,20000,28000):
    p = root/f"bench_{it}.json"
    if p.exists():
        s = json.loads(p.read_text())["summary"]
        print(f"{it:>6} | {s['mean_norm_return']:>9.4f} | {s['failed']:>9}")
    else:
        print(f"{it:>6} | {'--':>9} | {'--':>9}")
PY
echo "全部完成"
