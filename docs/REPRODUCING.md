# Reproducing the chair experiments

## Fixed training protocol

- Seed132422,4744 objects×24 views,64×64 RGB,batch32,3558 updates/epoch.
-238 epochs,846804 updates/group. Seven independent paired initializations;
  branch RNG scopes do not shift common parameter initialization.
- Adam LR1e-4/common,2.5e-5/residual,betas(.9,.999),eps1e-8.
  LR×.2 after224 completed epochs; after-backward clipping1.0/.5.
- Inherited detach-based pose/3D alternation each batch.
- Progressive zero-based indices14/84/154; texture0/14/84/154.
- Shape/texture alpha maxima.005/.01. Without S: active branches fixed from
  start. With S:`alpha_max*clip((epoch_index-30)/194,0,1)`.
  Index30 is zero,31 first positive,224 reaches maximum.
- Fixed64 validation objects/view0 every4000 updates; monitoring only.
  Compare fixed final epoch238, never the minimum validation checkpoint.

`configs/chair` mirrors `ablation.protocol.config_for`. Preparation freezes
source/config/data hashes. `src/trainer.py` is retained legacy infrastructure;
it is not the entry point for these seven runs. Use`train.py` for preparation,
then the prepared stage's`run.py` for execution.

## Linux/Slurm

Prepare as shown in the root README. `--partition NAME` sets the training
Slurm partition, default`gpu`. Generated jobs use the current Python executable;
its environment and mounted inputs must be available on compute nodes.
Review cluster resource/time settings. Evaluation jobs default to`gpu`;
override via`sbatch --partition=NAME` if necessary.

Submit smoke, inspect`ALL_PASSED: True`, then submit training. The array maps
0..6 toU/G/T/GT/GS/TS/GTS, each1GPU, maximum2 concurrent. No distributed
training is used. Do not execute GPU work on a cluster login node.

## Local or allocated GPU without Slurm

Linux is required (`fcntl` locks). After preparation:

```bash
python /path/to/workspace/stages/chair_ablation_v1/run.py --smoke --workspace /path/to/workspace
python /path/to/workspace/stages/chair_ablation_v1/run.py --train --group U --workspace /path/to/workspace
```

Run other groups similarly. Training remains gated by a matching GPU smoke.
For local evaluation, set a numeric identity for report filenames:

```bash
SLURM_JOB_ID=1 python /path/to/workspace/stages/chair_metrics_v1/evaluate.pyz --smoke --workspace /path/to/workspace
SLURM_JOB_ID=2 python /path/to/workspace/stages/chair_metrics_v1/evaluate.pyz --full --group U --workspace /path/to/workspace
```

That variable on a local workstation identifies the run, not a real allocation.

## Checkpoint/resume

Preparation refuses to overwrite an existing stage. Reuse its frozen source.
Each group has`runs/GROUP/checkpoint.pkl`and eventually`final.pkl`.
State includes model/buffers,Adam,scheduler,Python/NumPy/Torch/CUDA RNG,
progressive epoch/iteration,phase,neighbor memory,data cursor and counters.
Save every2000 updates and epoch boundary. Load only trusted checkpoints.

Group kernel locks reject simultaneous duplicate runs. After93h, training
cooperatively saves and prints`CLEAN_STOP`ahead of the96h limit. Exit0 at a
clean stop is not final completion. Resume only unfinished array indices:

```bash
sbatch --array=4 /path/to/workspace/stages/chair_ablation_v1/train_array.sbatch
```

Final evidence:`TRAINING_COMPLETED`,final cursor238/0,
`training_completed=true`in`segment_end.json`,and`final.pkl`.
Completed groups refuse training again. Source/config/data hashes must match.
Evaluation cache similarly binds its exact stage/weights/protocol.

## Loss curves

Every update logs weighted component/total losses,phase,epoch/batch/counters,
alpha,LR,gradient norms and residual RMS. PNG/PDF plots preserve all rows,
restart segments and phases; no smoothing or fabricated missing values.

```bash
python /path/to/workspace/stages/chair_ablation_v1/run.py --plot --workspace /path/to/workspace
```

Released comparison plots use exact per-epoch/per-phase arithmetic means.
All epoch means and211 validation points/group are in`results/chair`.
The large original per-update CSVs are not included in Git.

## Verification

Numerical model/optimizer and metric math were exercised in original A100 runs:
seven238-epoch trainings and9492 final test entries. Means were independently
recomputed from identical JSON/CSV/cache records.

This public port replaces private report dependencies and paths with arguments.
Local checks cover immutable preparation,config/schedules,source identity,
result aggregation and image metrics. A complete seven-group GPU training
has not been repeated after the public launch-adapter port. Every new environment
must run the included smoke/pilot. Original private-stage fingerprints are not
silently rewritten for this repository.

CUDA backward reductions can be nondeterministic. Resume acceptance separately
checks exact state/forward loss,bounded gradient vector differences and exact
same-gradient Adam replay. It does not promise bitwise long CUDA trajectories
or constitute multi-seed statistical evidence.
