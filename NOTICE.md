# Attribution and provenance

Based on [monniert/unicorn](https://github.com/monniert/unicorn), reference commit
`0aaf84408b8d9360bea515d2738cc10525e1fd26`, MIT, copyright(c)2022 monniert.
The original license is retained verbatim in `licenses/UNICORN-MIT.txt`.

`src/` is the U-OPT model snapshot used by the audited experiments, derived from
UNICORN. Attribution links in original source comments remain. Template
deformation, rendering, progressive conditioning, cross-instance consistency
and detach-based pose/3D alternation are inherited mechanisms.

`ablation/engine.py` supplies controlled residual branches, paired initialization,
alpha buffers, optimizer/logging and complete state restoration.
`ablation/protocol.py` retains the audited configuration values. Public stage
names and preparation adapters differ from the original cluster installation;
stage hashes intentionally change with this port.

`evaluation/geometry_metrics.py` retains verified point/normal/triangle formulas;
`evaluation/image_metrics.py` defines the RGB protocol.
`results/chair/provenance.json` records source hashes and upstream identity.

Dependencies including PyTorch/PyTorch3D/SciPy have their own licenses. UNICORN
source comments identify adaptations from PyTorch3D and pytorch-ssim. Dataset
assets, VGG and task checkpoints are obtained separately, not bundled here.
