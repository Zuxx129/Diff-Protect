# SD3 Attack v3.1 实验运行说明

本文档给出 v3.1 实验的实际运行命令。所有命令默认在仓库根目录执行：

```powershell
cd D:\courses\content_sec\theory\final\exp\Diff-Protect
```

v3.1 当前方法口径：

```text
textual_only, E, O, A, B, C, D, FMP_single, FMP_multi
```

其中：

- `textual_only`：只优化 textual target-pull。
- `E`：从 v2.1 O 拆出的 semantic-only velocity norm。
- `O/A/B/C/D`：v2.1 joint objective 口径，主实验固定 `textual_weight=1`、`mmdit_weight=100000`。
- `FMP_single/FMP_multi`：v3.1 FMP 主方法。
- `FMP_single_plus_step`：可选单步转移消融，不进入主实验网格。

新实验不使用 `O_repo`、`O_fair`、`textual_semantic_joint` 作为方法名。

---

## 1. 静态检查

### 1.1 Python 环境检查

不要直接假设 PowerShell 前缀显示 `(base)` 就是可用环境。先确认当前 Python：

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip show hydra-core omegaconf diffusers transformers accelerate safetensors sentencepiece protobuf einops tqdm pillow
```

如果运行 `code/diff_mist_SD3_v3.py` 时出现：

```text
ModuleNotFoundError: No module named 'hydra'
```

说明当前 Python 环境没有 `hydra-core`。v3.1 SD3/SD3.5 入口至少需要：

```text
hydra-core
omegaconf
diffusers
transformers
accelerate
safetensors
sentencepiece
protobuf
einops
tqdm
pillow
torch
torchvision
```

推荐新建独立环境，不要污染 base：

```powershell
conda create -n sd3v31 python=3.10 -y
conda activate sd3v31
python -m pip install --upgrade pip
python -m pip install hydra-core omegaconf diffusers transformers accelerate safetensors sentencepiece protobuf einops tqdm pillow opencv-python
```

然后按本机 CUDA 版本安装 PyTorch。若不确定 CUDA 版本，先检查：

```powershell
nvidia-smi
```

CUDA 12.1 示例：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8 示例：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

只想临时修复当前 base 环境时，可最小安装缺失包：

```powershell
python -m pip install hydra-core omegaconf diffusers transformers accelerate safetensors sentencepiece protobuf einops tqdm pillow opencv-python
```

安装后验证：

```powershell
python -c "import hydra, omegaconf, diffusers, transformers, torch; print('env ok', torch.__version__)"
```

仓库根目录的 `env.yml` 是原始 MIST / SD 1.x 时代环境，包含 `hydra-core`，但 Python 和 PyTorch 版本偏旧。v3.1 面向 SD3/SD3.5，优先使用上面的独立 `sd3v31` 环境。

### 1.2 Hugging Face gated model 授权

默认模型：

```text
stabilityai/stable-diffusion-3.5-medium
```

这是 Hugging Face gated repo。如果报错：

```text
401 Unauthorized
Cannot access gated repo
Access to model stabilityai/stable-diffusion-3.5-medium is restricted
```

需要先完成三步：

1. 登录 Hugging Face，并在模型页面申请/接受访问条款：
   ```text
   https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
   ```
2. 创建 read token：
   ```text
   https://huggingface.co/settings/tokens
   ```
3. 在本机设置 `HF_TOKEN`，或把 CLI 登录 token 写到项目内 `hub/`。

v3.1 入口会把 Hugging Face 缓存固定到项目目录：

```text
Diff-Protect/hub/
```

因此最稳妥方式是在当前 PowerShell 会话直接设置 `HF_TOKEN`：

```powershell
$env:HF_TOKEN = "hf_xxx"
```

然后重新运行实验命令。不要把 token 写成 Hydra 参数，也不要写入 `config.json`、脚本或 markdown。

如果希望使用 CLI 登录而不是 `HF_TOKEN`，需要让 CLI 也写到项目内 `hub/`：

```powershell
python -m pip install -U huggingface_hub
$env:HF_HOME = (Join-Path (Get-Location) "hub")
hf auth login
hf auth whoami
```

如果 `hf` 命令不可用，可以尝试旧命令：

```powershell
$env:HF_HOME = (Join-Path (Get-Location) "hub")
huggingface-cli login
huggingface-cli whoami
```

### 1.3 代码静态检查

先确认 v3.1 攻击、收集、分析和绘图脚本没有语法错误：

```powershell
python -m py_compile `
  code/attacks_SD3_v3.py `
  code/diff_mist_SD3_v3.py `
  scripts/collect_sd3_v3_data.py `
  code/analysis/analyze_sd3_v3_results.py `
  code/metrics/compute_sd3_metrics.py `
  draw/figures/style.py `
  draw/figures/io.py `
  draw/figures/loss_curves.py `
  draw/figures/metric_comparison.py `
  draw/figures/diagnostics.py `
  draw/figures/ablations.py `
  draw/figures/qualitative.py `
  draw/figures/make_all_figures.py
```

---

## 2. Collector dry-run

dry-run 只生成命令和日志，不加载模型、不运行 GPU 实验：

```powershell
python scripts/collect_sd3_v3_data.py `
  --modes textual_only,E,O,A,B,C,D,FMP_single,FMP_multi `
  --epsilons 8 `
  --steps 50 `
  --seeds 0 `
  --random-start true `
  --textual-weight 1 `
  --mmdit-weight 100000 `
  --input-size 512 `
  --max-exp-num 1 `
  --run-sdedit false `
  --paired-sdedit false `
  --output-path out_sd3_v3/ `
  --log-root out_sd3_v3/dry_run_logs `
  --dry-run
```

检查输出：

```powershell
Get-Content out_sd3_v3\dry_run_logs\manifest.csv
```

---

## 3. 单方法 GPU smoke test

每个 smoke test 只跑 1 张图、2 个 PGD step，不做 paired SDEdit，用于验证模型加载和 loss 分支是否能跑通。

```powershell
python code/diff_mist_SD3_v3.py attack.mode=textual_only attack.epsilon=4 attack.steps=2 attack.seed=0 attack.max_exp_num=1 attack.paired_sdedit=false attack.run_sdedit=false
```

```powershell
python code/diff_mist_SD3_v3.py attack.mode=E attack.epsilon=4 attack.steps=2 attack.seed=0 attack.max_exp_num=1 attack.paired_sdedit=false attack.run_sdedit=false
```

```powershell
python code/diff_mist_SD3_v3.py attack.mode=O attack.epsilon=4 attack.steps=2 attack.seed=0 attack.max_exp_num=1 attack.paired_sdedit=false attack.run_sdedit=false
```

```powershell
python code/diff_mist_SD3_v3.py attack.mode=A attack.epsilon=4 attack.steps=2 attack.seed=0 attack.max_exp_num=1 attack.paired_sdedit=false attack.run_sdedit=false
```

```powershell
python code/diff_mist_SD3_v3.py attack.mode=FMP_single attack.epsilon=4 attack.steps=2 attack.seed=0 attack.max_exp_num=1 attack.paired_sdedit=false attack.run_sdedit=false
```

若 GPU 不是配置文件默认的 `cuda:2`，显式指定：

```powershell
python code/diff_mist_SD3_v3.py attack.mode=FMP_single attack.device=cuda:0 attack.epsilon=4 attack.steps=2 attack.seed=0 attack.max_exp_num=1 attack.paired_sdedit=false attack.run_sdedit=false
```

---

## 4. 最小实验

最小实验用于确认完整链路：攻击生成、paired SDEdit、metrics、analysis、figures。

### Bash / Git Bash / WSL

```bash
bash scripts/run_sd3_v3_minimal.sh
```

该脚本运行：

```text
methods = textual_only,E,O,FMP_single
epsilon = 8
steps = 50
seed = 0
random_start = true
paired_sdedit = true
sdedit_noise_levels = 0.1,0.3,0.5
```

### PowerShell 等价命令

```powershell
python scripts/collect_sd3_v3_data.py `
  --modes textual_only,E,O,FMP_single `
  --epsilons 8 `
  --steps 50 `
  --seeds 0 `
  --random-start true `
  --textual-weight 1 `
  --mmdit-weight 100000 `
  --input-size 512 `
  --max-exp-num 1 `
  --run-sdedit true `
  --paired-sdedit true `
  --output-path out_sd3_v3/ `
  --log-root out_sd3_v3/minimal_logs `
  --skip-existing
```

然后生成指标、分析表和图：

```powershell
python code/metrics/compute_sd3_metrics.py --root out_sd3_v3 --out out_sd3_v3/full_metrics.csv
python code/analysis/analyze_sd3_v3_results.py --metrics out_sd3_v3/full_metrics.csv --out-dir out_sd3_v3/analysis
python -m draw.figures.make_all_figures --root out_sd3_v3 --metrics out_sd3_v3/full_metrics.csv --analysis-dir out_sd3_v3/analysis --out-dir out_sd3_v3/figures --loss-image suzume
```

---

## 5. 主实验 full modes

主实验网格：

```text
methods = textual_only,E,O,A,B,C,D,FMP_single,FMP_multi
epsilon = 4,8,16
steps = 50,100
seeds = 0,1,2
random_start = true
textual_weight = 1
mmdit_weight = 100000
paired_sdedit = true
sdedit_noise_levels = 0.1,0.3,0.5
```

### Bash / Git Bash / WSL

```bash
bash scripts/run_sd3_v3_full_modes.sh
```

### PowerShell 等价命令

```powershell
python scripts/collect_sd3_v3_data.py `
  --modes textual_only,E,O,A,B,C,D,FMP_single,FMP_multi `
  --epsilons 4,8,16 `
  --steps 50,100 `
  --seeds 0,1,2 `
  --random-start true `
  --textual-weight 1 `
  --mmdit-weight 100000 `
  --input-size 512 `
  --max-exp-num 100 `
  --run-sdedit true `
  --paired-sdedit true `
  --output-path out_sd3_v3/ `
  --log-root out_sd3_v3/full_logs `
  --skip-existing
```

汇总和绘图：

```powershell
python code/metrics/compute_sd3_metrics.py --root out_sd3_v3 --out out_sd3_v3/full_metrics.csv
python code/analysis/analyze_sd3_v3_results.py --metrics out_sd3_v3/full_metrics.csv --out-dir out_sd3_v3/analysis
python -m draw.figures.make_all_figures --root out_sd3_v3 --metrics out_sd3_v3/full_metrics.csv --analysis-dir out_sd3_v3/analysis --out-dir out_sd3_v3/figures --loss-image suzume
```

---

## 6. 可选 step ablation

该消融只在需要检查 `FMP_single_plus_step` 时运行。

### Bash / Git Bash / WSL

```bash
bash scripts/run_sd3_v3_step_ablation.sh
```

### PowerShell 等价命令

```powershell
python scripts/collect_sd3_v3_data.py `
  --modes FMP_single,FMP_single_plus_step `
  --epsilons 8 `
  --steps 50 `
  --seeds 0,1,2 `
  --random-start true `
  --textual-weight 1 `
  --mmdit-weight 100000 `
  --lambda-step 1.0 `
  --input-size 512 `
  --max-exp-num 100 `
  --run-sdedit true `
  --paired-sdedit true `
  --output-path out_sd3_v3/ `
  --log-root out_sd3_v3/step_ablation_logs `
  --skip-existing
```

汇总和绘图同主实验：

```powershell
python code/metrics/compute_sd3_metrics.py --root out_sd3_v3 --out out_sd3_v3/full_metrics.csv
python code/analysis/analyze_sd3_v3_results.py --metrics out_sd3_v3/full_metrics.csv --out-dir out_sd3_v3/analysis
python -m draw.figures.make_all_figures --root out_sd3_v3 --metrics out_sd3_v3/full_metrics.csv --analysis-dir out_sd3_v3/analysis --out-dir out_sd3_v3/figures --loss-image suzume
```

---

## 7. 单独重算 metrics、analysis、figures

只重算基础图像指标：

```powershell
python code/metrics/compute_sd3_metrics.py --root out_sd3_v3 --out out_sd3_v3/full_metrics.csv
```

若需要 CLIP prompt drop / source similarity drop：

```powershell
python code/metrics/compute_sd3_metrics.py --root out_sd3_v3 --out out_sd3_v3/full_metrics.csv --clip
```

若需要 LPIPS：

```powershell
python code/metrics/compute_sd3_metrics.py --root out_sd3_v3 --out out_sd3_v3/full_metrics.csv --lpips
```

CLIP 和 LPIPS 都计算：

```powershell
python code/metrics/compute_sd3_metrics.py --root out_sd3_v3 --out out_sd3_v3/full_metrics.csv --clip --lpips
```

生成分析表：

```powershell
python code/analysis/analyze_sd3_v3_results.py --metrics out_sd3_v3/full_metrics.csv --out-dir out_sd3_v3/analysis
```

生成所有论文图：

```powershell
python -m draw.figures.make_all_figures `
  --root out_sd3_v3 `
  --metrics out_sd3_v3/full_metrics.csv `
  --analysis-dir out_sd3_v3/analysis `
  --out-dir out_sd3_v3/figures `
  --loss-image suzume `
  --formats png,svg `
  --dpi 300
```

Windows 一键绘图脚本：

```powershell
.\scripts\draw_sd3_v3_figures.ps1 -Root out_sd3_v3 -Formats png,svg -Dpi 300
```

带 CLIP / LPIPS 的 Windows 一键绘图：

```powershell
.\scripts\draw_sd3_v3_figures.ps1 -Root out_sd3_v3 -Formats png,svg -Dpi 300 -Clip -Lpips -ForceMetrics
```

Bash 一键绘图：

```bash
bash scripts/draw_sd3_v3_figures.sh
```

Bash 带 CLIP / LPIPS：

```bash
CLIP=1 LPIPS=1 FORCE_METRICS=1 bash scripts/draw_sd3_v3_figures.sh
```

---

## 8. 只画 loss 曲线

v3.1 loss 曲线：

```powershell
python -m draw.figures.loss_curves --root out_sd3_v3 --image suzume --components --output out_sd3_v3/figures/loss --formats png,svg --dpi 300
```

兼容旧入口：

```powershell
python code/plot_loss.py --root out_sd3_v3 --image suzume --components --output out_sd3_v3/figures/loss --formats png,svg --dpi 300
```

如果要重画历史 `out_sd3` 的 O/A/B/C/D 组件图：

```powershell
python -m draw.figures.loss_curves --root out_sd3 --image suzume --modes O A B C D --components --output out_sd3/figures --formats png,svg --dpi 300
```

---

## 9. 输出目录

单次攻击输出目录格式：

```text
out_sd3_v3/
  <mode>_eps<epsilon>_steps<steps>_optmaximize_rstrue_seed<seed>/
    to_protect/
      *_attacked.png
      *_loss.npz
      *_sdedit_clean_noise_<sigma>.png
      *_sdedit_adv_noise_<sigma>.png
```

日志目录：

```text
out_sd3_v3/minimal_logs/
out_sd3_v3/full_logs/
out_sd3_v3/step_ablation_logs/
```

聚合输出：

```text
out_sd3_v3/
  full_metrics.csv
  analysis/
    full_metrics.csv
    summary_by_method.csv
    paired_differences.csv
    ablation_summary.csv
  figures/
    main/
    ablation/
    diagnostics/
    loss/
    qualitative/
    figure_manifest.csv
```

---

## 10. 常用排查命令

查看某次收集脚本生成的命令和状态：

```powershell
Import-Csv out_sd3_v3\full_logs\manifest.csv | Select-Object index,mode,epsilon,steps,seed,status,returncode,log_path
```

查看失败日志：

```powershell
Get-Content <log_path> -Tail 120
```

确认每个 run 是否有 loss：

```powershell
Get-ChildItem out_sd3_v3 -Recurse -Filter "*_loss.npz" | Select-Object FullName
```

确认 paired SDEdit 是否存在：

```powershell
Get-ChildItem out_sd3_v3 -Recurse -Filter "*_sdedit_adv_noise_*.png" | Measure-Object
Get-ChildItem out_sd3_v3 -Recurse -Filter "*_sdedit_clean_noise_*.png" | Measure-Object
```

检查 metrics 行数：

```powershell
(Import-Csv out_sd3_v3\full_metrics.csv).Count
```

---

## 11. 运行顺序建议

推荐顺序：

```text
1. py_compile 静态检查
2. collector dry-run
3. 单方法 GPU smoke
4. minimal 实验
5. metrics / analysis / figures
6. full modes 主实验
7. 可选 step ablation
8. 重新计算带 CLIP / LPIPS 的 metrics
9. 生成最终论文图
```

若 GPU 时间有限，先跑：

```text
textual_only, E, O, FMP_single
epsilon=8
steps=50
seed=0
```

确认 FMP 有外部编辑指标优势后，再扩展到：

```text
textual_only, E, O, A, B, C, D, FMP_single, FMP_multi
epsilon=4,8,16
steps=50,100
seeds=0,1,2
```
