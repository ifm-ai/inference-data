#!/bin/bash
# TPS probe: load each model, run fixed prompt, record predicted_per_second.
# Usage: ./tps-probe.sh  (log: k2-tps.log, results: k2-tps-results.jsonl)
# Skips BF16 (have 0.09) and batchA dupes (byte-identical to Research files).
OUT=/mnt/Projects/lemonade-server/k2-tps-results.jsonl
PROMPT="Explain why the sky is blue in one short paragraph."
MODELS="K2-Horizon-7B-GGUF user.K2-MoVA-Q4_K_M user.K2-MoVA-Q8_0
K2-Horizon-MoVA-36B-A4B-IQ1_S K2-Horizon-MoVA-36B-A4B-IQ1_M
K2-Horizon-MoVA-36B-A4B-IQ2_XXS K2-Horizon-MoVA-36B-A4B-IQ2_XS
K2-Horizon-MoVA-36B-A4B-IQ2_S K2-Horizon-MoVA-36B-A4B-IQ2_M
K2-Horizon-MoVA-36B-A4B-IQ3_XXS K2-Horizon-MoVA-36B-A4B-IQ3_XS
K2-Horizon-MoVA-36B-A4B-IQ3_S K2-Horizon-MoVA-36B-A4B-IQ3_M
K2-Horizon-MoVA-36B-A4B-IQ4_XS K2-Horizon-MoVA-36B-A4B-IQ4_NL
K2-Horizon-MoVA-36B-A4B-MXFP4_MOE K2-Horizon-MoVA-36B-A4B-Q2_K
K2-Horizon-MoVA-36B-A4B-Q3_K_M K2-Horizon-MoVA-36B-A4B-Q4_K_M
K2-Horizon-MoVA-36B-A4B-Q5_K_M K2-Horizon-MoVA-36B-A4B-Q6_K
K2-Horizon-MoVA-36B-A4B-Q8_0 K2-Horizon-MoVA-36B-A4B-Q3_K_L
feffd71999eb06bfa2fbb8ad220059b694077457"
: > "$OUT"
for m in $MODELS; do
  disp="${m#user.}"
  echo "===== $m $(date -u +%FT%TZ)"
  curl -s http://localhost:13305/v1/load -H 'Content-Type: application/json' \
    -d "{\"model_name\":\"$m\",\"ctx_size\":2048}" >/dev/null
  ok=0
  for i in $(seq 1 200); do
    if curl -s --max-time 10 http://localhost:13305/v1/models 2>/dev/null | grep -q "$disp"; then ok=1; break; fi
    sleep 15
  done
  if [ "$ok" = "1" ]; then
    curl -s --max-time 900 http://localhost:13305/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0,\"max_tokens\":64}" \
      | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
    t = d.get('timings', {})
    print(json.dumps({'model': '$m',
        'predicted_per_second': round(t.get('predicted_per_second', 0) or 0, 2),
        'prompt_per_second': round(t.get('prompt_per_second', 0) or 0, 1),
        'predicted_n': t.get('predicted_n')}))
except Exception as e:
    print(json.dumps({'model': '$m', 'error': str(e)[:100]}))
" | tee -a "$OUT"
  else
    echo "{\"model\": \"$m\", \"error\": \"load timeout\"}" | tee -a "$OUT"
  fi
  /usr/bin/lemonade unload "$m" >/dev/null 2>&1 || true
done
echo "TPS PROBE DONE"
