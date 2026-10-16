#!/bin/bash
# Harness campaign: load each model, run 8-task set, unload, next.
# Usage: ./k2-harness-campaign.sh  (log: k2-harness-campaign.log)
export HF_ALLOW_CODE_EVAL=1
VENV=/mnt/Projects/lemonade-server/lm-eval-venv/bin
OUT=/mnt/Projects/lemonade-server/k2-eval-results
TASKS="humaneval,mbpp,ifeval,k2_toolcall,coqa,gsm8k,squad_completion"
SEG="bigbench_paragraph_segmentation_generate_until"
LIMITS="--limit humaneval:10,mbpp:10,bigbench_paragraph_segmentation_generate_until:5,ifeval:15,k2_toolcall:12,coqa:15,gsm8k:10,squad_completion:10"
# NOTE: lm-eval --limit takes a single int; per-task limits unsupported -> use 12 global.
# Override: run with --limit 12 (ifeval/coqa/gsm8k/squad get 12, others capped by size).

MODELS="${*:-K2-Horizon-7B-GGUF user.K2-MoVA-Q4_K_M user.K2-MoVA-Q8_0}"

wait_loaded() {
  local m="$1" disp="${1#user.}" i=0
  while [ $i -lt 360 ]; do
    if curl -s --max-time 10 http://localhost:13305/v1/models 2>/dev/null | grep -q "$disp"; then return 0; fi
    sleep 15; i=$((i+1))
  done
  return 1
}

run_one() {
  local m="$1" disp="${1#user.}"
  if ! /usr/bin/lemonade list --downloaded 2>/dev/null | grep -q "$disp"; then
    echo "===== SKIP $m (not registered)"
    return 0
  fi
  # Resume: skip models with complete sample sets from a prior night.
  local sdir="/mnt/Projects/lemonade-server/k2-eval-results/$m" complete=1
  for t in humaneval mbpp ifeval k2_toolcall coqa gsm8k squad_completion bigbench_paragraph_segmentation_generate_until; do
    ls "$sdir/samples_${t}_"* >/dev/null 2>&1 || complete=0
  done
  if [ "$complete" = "1" ]; then
    echo "===== SKIP $m (samples complete)"
    return 0
  fi
  echo "===== LOADING $m $(date -u +%FT%TZ)"
  curl -s http://localhost:13305/v1/load -H 'Content-Type: application/json' \
    -d "{\"model_name\":\"$m\",\"ctx_size\":${CTX:-4096}}" | head -c 200; echo
  if ! wait_loaded "$m"; then echo "LOAD TIMEOUT $m"; return 1; fi
  echo "===== EVAL $m"
  "$VENV/lm-eval" run --model local-completions \
    --model_args "model=$m,base_url=http://localhost:13305/v1/completions,timeout=1200,tokenizer_backend=huggingface,tokenizer=IFM/K2-Horizon-7B" \
    --tasks "$TASKS" --include_path /mnt/Projects/lemonade-server/k2-tasks \
    --limit 12 --gen_kwargs max_gen_toks=256 temperature=0 \
    --confirm_run_unsafe_code --log_samples --output_path "$OUT" 2>&1 | tail -n 16
  echo "===== EVAL-SEG $m"
  "$VENV/lm-eval" run --model local-completions \
    --model_args "model=$m,base_url=http://localhost:13305/v1/completions,timeout=1200,tokenizer_backend=huggingface,tokenizer=IFM/K2-Horizon-7B" \
    --tasks "$SEG" \
    --limit 5 --gen_kwargs max_gen_toks=256 temperature=0 \
    --log_samples --output_path "$OUT" 2>&1 | tail -n 6
  echo "===== UNLOAD $m"
  /usr/bin/lemonade unload "$m" 2>&1 | head -n 2 || true
}

for m in $MODELS; do run_one "$m"; done
echo "CAMPAIGN PART DONE: $MODELS"
