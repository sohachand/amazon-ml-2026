#!/bin/bash
# Runs unattended on the Jarvis box:
#  1) waits for the small-encoder rerun to finish
#  2) exports the probability files for teammates (submissions/)
#  3) pushes them to GitHub if GH_REPO + GH_TOKEN are set
#  4) waits for the e5-base cross-encoder fine-tune (if running), then runs the upgrade
GH_REPO="${GH_REPO:-}"      # e.g. gauravxyz/amazon-ml-2026   (owner/repo)
GH_TOKEN="${GH_TOKEN:-}"    # GitHub token with write access to that repo
cd ~/er_gpu
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for small run"
while pgrep -f "work work --out output" >/dev/null; do sleep 60; done
if ! grep -q "wrote outputs" ~/run_small.txt; then
  log "small run did NOT finish cleanly - see ~/run_small.txt; stopping"; exit 1
fi
cp output/matching_results.tsv output/matching_results_small_rerun.tsv
log "small run done; exporting"
mkdir -p submissions
python3 src/export_probs.py work submissions 2>&1 | tee submissions/soha_export_stats.txt

if [ -n "$GH_REPO" ] && [ -n "$GH_TOKEN" ]; then
  log "pushing to github $GH_REPO"
  rm -rf ~/teamrepo
  git clone --depth 1 "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" ~/teamrepo \
    && mkdir -p ~/teamrepo/soha \
    && cp submissions/soha_* ~/teamrepo/soha/ \
    && mkdir -p ~/teamrepo/soha/src && cp src/*.py src/*.sh ~/teamrepo/soha/src/ \
    && cp requirements_gpu.txt unpack_data.py ~/teamrepo/soha/ && cp SOHA_README.md ~/teamrepo/soha/README.md \
    && { cp ~/run_local_heavy.sh ~/teamrepo/soha/ 2>/dev/null; cp -r utils ~/teamrepo/soha/ 2>/dev/null; true; } \
    && cd ~/teamrepo && git config user.name "soha" && git config user.email "soha@users.noreply.github.com" \
    && git add -f soha/ && git commit -m "soha: pipeline code + stage-3 pair probabilities (small encoder)" \
    && git pull --rebase -q && git push && log "PUSHED" || log "push FAILED - files are in ~/er_gpu/submissions"
  cd ~/er_gpu
else
  log "no GH_REPO/GH_TOKEN set - files are in ~/er_gpu/submissions"
fi

log "waiting for e5-base cross-encoder fine-tune (if running)"
while pgrep -f "stage ce_train" >/dev/null; do sleep 60; done
mkdir -p work_base/train work_base/test
for f in work/train/*; do case "$(basename $f)" in ce_job.npz|ce_tmp.npy) ;; *) [ -e "work_base/train/$(basename $f)" ] || ln -s "$(realpath $f)" work_base/train/;; esac; done
for f in work/test/*; do case "$(basename $f)" in scores.npy|scores3.npy|ceK.npy|ce_job.npz|ce_tmp.npy) ;; *) [ -e "work_base/test/$(basename $f)" ] || ln -s "$(realpath $f)" work_base/test/;; esac; done
[ -e work_base/e5_finetuned ] || ln -s "$(realpath work/e5_finetuned)" work_base/e5_finetuned
log "starting e5-base upgrade run -> ~/run2.txt"
CE_MODEL=intfloat/multilingual-e5-base N_PROC=30 TOKENIZERS_PARALLELISM=false \
  python3 src/pipeline.py --data dataset --work work_base --out output_base --gpu --ce --train-frac 0.08 > ~/run2.txt 2>&1
log "upgrade finished:"; tail -c 800 ~/run2.txt
