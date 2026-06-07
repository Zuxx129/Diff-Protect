# SD3 工程实现说明

> 对应理论文档：`SD3_对抗扰动设计V2.1.md`  
> 对应分支：`fix/sd3-objective-sanity`  
> 目标：说明 V2.1 攻击建模如何落到代码中，哪些部分是严格实现，哪些部分是工程折中，如何运行、验证、收集数据和定位错误。

---

## 0. 当前工程边界

当前分支实现的是面向 SD3 / SD3.5 的白盒结构感知扰动实验框架。它不是一个通用图片攻击脚本，而是为了验证以下研究问题：

1. VAE latent textual target-pull 是否仍可作为 SD3 图像保护的基础项；
2. MMDiT Joint Attention、image-stream feature、flow velocity、image/text modality balance 是否可作为 SD3-specific attack surface；
3. 内部机制偏移是否能通过 paired SDEdit 转化为外部编辑结果偏移；
4. 结构化扰动是否显著优于同预算随机噪声与非结构 baseline。

当前实现已经包含攻击生成、paired SDEdit、row-level metrics、summary aggregation、随机 baseline、FID/KID 可选计算、最小/方向/完整实验 runner。仍需 GPU 实验确认显存、速度和具体效果。

---

## 1. 文件职责

| 文件 | 作用 |
|---|---|
| `configs/attack/base_sd3.yaml` | SD3 v2.1 的统一配置入口，包含 `opt_direction`、`textual_objective`、`seed`、`paired_sdedit` 等字段。 |
| `code/attacks_SD3.py` | 核心 PGD 与 O/A/B/C/D loss 实现。 |
| `code/diff_mist_SD3_v2.py` | v2.1 主入口，负责加载 SD3 pipeline、调用 PGD、执行 paired SDEdit、保存图像与 loss。 |
| `code/diff_mist_SD3.py` | 历史兼容入口。后续论文实验优先使用 v2 入口。 |
| `code/metrics/compute_sd3_metrics.py` | 从输出目录读取 attacked / paired SDEdit / loss.npz，生成 row-level metrics CSV。 |
| `code/metrics/aggregate_sd3_results.py` | 从 row-level metrics 聚合 mode / epsilon / steps / sigma 级 summary 与 success rate。 |
| `code/metrics/random_linf_baseline.py` | 生成同预算随机扰动 baseline。 |
| `code/metrics/compute_fid_kid.py` | 对 paired clean/adv SDEdit 集合计算 FID/KID，可选依赖 `torchmetrics`。 |
| `code/plot_loss.py` | 绘制 total/textual/mmdit 与机制诊断曲线。 |
| `scripts/validate_sd3_pipeline.py` | 静态优先验证脚本，检查文件、配置、语义关键字、入口与实验工具链。 |
| `scripts/collect_sd3_data.py` | 网格实验调度器，记录每条命令、日志路径、return code 与 manifest。 |
| `scripts/run_sd3_validation.sh` | 无 GPU / dry-run 验证入口。 |
| `scripts/run_sd3_direction_check.sh` | A/B/C/D 方向与梯度检查。 |
| `scripts/run_sd3_minimal_collect.sh` | 最小 paired 实验采集。 |
| `scripts/run_sd3_full_modes.sh` | 主表模式实验采集。 |
| `scripts/run_sd3_random_baseline.sh` | 随机扰动 baseline 生成与聚合。 |
| `scripts/run_sd3_ablation_weights.sh` | textual/mmdit 权重消融。 |

---

## 2. 配置层实现

`base_sd3.yaml` 中新增或规范化的字段：

```yaml
attack:
  g_mode: "+"
  opt_direction: maximize
  objective_convention: larger_is_more_attack
  textual_objective: toward_target
  debug_grad: False
  capture_blocks: [0, 6, 12, 17]
  seed: 0
  run_sdedit: True
  paired_sdedit: False
  sdedit_noise_levels: [0.1, 0.3, 0.5]
  sdedit_steps: 28
```

工程语义：

- `g_mode`：旧字段，仅用于兼容；
- `opt_direction`：真实优化方向；
- `objective_convention`：记录所有 loss 均按 larger-is-more-attack 设计；
- `textual_objective`：`toward_target` 使用负 MSE，`away_from_target` 使用正 MSE；
- `capture_blocks`：减少 hook 层数，降低显存；
- `paired_sdedit`：启用 clean/adv paired SDEdit；
- `sdedit_noise_levels`：SDEdit-style 加噪强度；
- `sdedit_steps`：denoising steps。

---

## 3. 攻击主链路实现

### 3.1 入口函数

论文实验优先使用：

```bash
python code/diff_mist_SD3_v2.py ...
```

该入口完成：

1. 设置 seed；
2. 根据路径选择 prompt；
3. 加载 `StableDiffusion3Pipeline`；
4. 构造 `SD3_target_model`；
5. 调用 `SD3_Linf_PGD`；
6. 保存 attacked image；
7. 按需执行 paired SDEdit；
8. 保存 `.npz` loss / diagnostics。

### 3.2 PGD 实现

`SD3_Linf_PGD.pgd_sd3()` 是核心循环：

```python
for i in range(steps):
    X_adv.requires_grad_(True)
    loss, components = self._compute_loss(...)
    loss.backward()
    X_adv = X_adv + g_dir * sign(grad) * step_size
    X_adv = project_to_linf_ball(X_adv)
```

工程注意：

- `g_dir` 由 `opt_direction` 决定；
- `X_adv` 每步 detach 后重新 requires_grad；
- 投影先限制到 `X ± eps`，再 clamp 到 `[-1,1]`；
- `loss_history` 自动保存所有非 `_` 开头的 component 和 diagnostic。

---

## 4. Loss 实现细节

### 4.1 Textual loss

位置：`SD3_Linf_PGD._textual_loss`

实现：

```python
raw_mse = MSE(z_adv, z_target)
if textual_objective == 'toward_target':
    return -raw_mse
if textual_objective == 'away_from_target':
    return raw_mse
```

工程折中：target image 仍使用 `test_images/target/MIST.png`。后续若做多 target ablation，应将 target path 纳入 config。

### 4.2 Mode A：text injection disruption

位置：

- `AttnMapCaptureProcessor`
- `cross_modal_disruption_loss`

实现路径：

1. 在 attention processor 内计算 image-query 到 text-key 的 attention：

```python
img_to_txt_attn = softmax(query @ text_key.T / sqrt(d))
```

2. 计算 text injection：

```python
txt_injection = img_to_txt_attn @ text_value
```

3. 对 clean 和 adv 的 `txt_injection` 做归一化 MSE：

```python
F.mse_loss(injection_adv, injection_clean) / norm(injection_clean)
```

4. 加入少量 entropy regularizer。

工程折中：没有保存完整 joint attention $[B,H,N,N]$，只保存 image-to-text block 和 injection output。这是为了控制显存。

### 4.3 Mode B：image-stream feature divergence

位置：

- `_compute_clean_features_at_timestep`
- `feature_divergence_loss`

实现路径：

1. 每个 PGD step 采样 timestep；
2. 用同一 timestep 计算 clean features；
3. 用同一 timestep 计算 adv features；
4. 比较 normalized cosine distance 与 Gram distance。

工程折中：clean features 每步重新计算，显存稳定但计算更慢。缓存多 timestep clean features 是未来优化方向。

### 4.4 Mode C：shared-noise trajectory divergence

位置：`_trajectory_loss_shared_noise`

实现路径：

```python
noise = randn_like(z_adv)
z_clean_t = (1 - sigma) * z_clean + sigma * noise
z_adv_t   = (1 - sigma) * z_adv   + sigma * noise
loss = 1 - cos(v_adv, v_clean)
```

工程折中：当前每步只采样一个 timestep。完整论文实验可扩展为 `num_timesteps_for_C > 1`。

### 4.5 Mode D：modality balance disruption

位置：`modality_imbalance_loss`

实现路径：

1. 计算 image/text stream energy ratio：

```python
ratio = log(E_img / E_txt)
```

2. 比较 clean/adv ratio deviation；
3. 计算 image/text channel covariance；
4. 用 covariance cosine 近似 CKA；
5. loss 为 ratio deviation + `0.1 * (1 - CKA)`。

工程折中：这里是 CKA proxy，不是完整 unbiased CKA estimator。它可作为机制指标和攻击目标，但论文中应称为 covariance-CKA proxy。

---

## 5. Paired SDEdit 实现

位置：`diff_mist_SD3_v2.py::_run_sdedit_pair`

流程：

1. 编码 `x_clean` 与 `x_adv`；
2. 对每个 noise level 生成同一个 `noise`；
3. 构造：

```python
z_clean_noisy = (1-sigma) * z_clean + sigma * noise
z_adv_noisy   = (1-sigma) * z_adv   + sigma * noise
```

4. 分别调用 `_denoise_from_noise_level`；
5. 保存：

```text
*_sdedit_clean_noise_<sigma>.png
*_sdedit_adv_noise_<sigma>.png
```

工程注意：paired SDEdit 对显存和时间开销明显高于 adv-only SDEdit。

---

## 6. Metrics 实现

### 6.1 Row-level metrics

`compute_sd3_metrics.py` 自动识别：

- attacked image；
- adv-only SDEdit image；
- paired clean/adv SDEdit image；
- `.npz` loss 与 diagnostics。

输出字段包括：

- `linf_perturb_0_1`；
- `linf_perturb_tensor_-1_1`；
- `psnr_perturb`；
- `ssim_perturb`；
- `l2_edit_rmse`；
- `psnr_edit`；
- `ssim_edit`；
- optional `lpips_perturb`、`lpips_edit`；
- optional `clip_prompt_clean/adv`、`delta_clip_prompt`；
- optional `clip_src_clean/adv`、`delta_clip_src`；
- all `loss_<key>_first/final/delta`。

### 6.2 Aggregation

`aggregate_sd3_results.py` 读取 metrics CSV，按 group keys 聚合。默认成功率定义使用：

- 最大 $L_\infty$；
- 最大 perturb LPIPS；
- 最小 edit LPIPS；
- 最小 CLIPScore drop。

若 optional 指标未计算，则不会因缺失字段直接失败；正式论文实验应启用 LPIPS/CLIP。

### 6.3 FID/KID

`compute_fid_kid.py` 搜索 paired SDEdit 文件：

```text
*_sdedit_clean_noise_*.png
*_sdedit_adv_noise_*.png
```

按 sigma 计算 FID/KID。该脚本依赖 `torchmetrics`，没有依赖时会清晰退出。

---

## 7. 实验脚本

### 7.1 静态验证

```bash
python scripts/validate_sd3_pipeline.py
```

检查：

- 文件存在；
- Python 语法；
- config keys；
- attack semantics；
- entrypoint；
- metrics / baseline / runner 是否存在。

### 7.2 方向检查

```bash
bash scripts/run_sd3_direction_check.sh cuda:0
```

目的：检查 A/B/C/D 的 `grad_mmdit_l2` 是否非零，以及机制指标是否随 step 变化。

### 7.3 最小实验

```bash
bash scripts/run_sd3_minimal_collect.sh cuda:0
```

默认启用 paired SDEdit，输出 `minimal_metrics.csv` 与 `minimal_summary.csv`。

### 7.4 完整模式实验

```bash
bash scripts/run_sd3_full_modes.sh cuda:0 512 5
```

覆盖：

```text
O_repo,O_fair,A,B,C,D
epsilon=4,8,16
steps=20,50
seed=0,1,2,3,4
```

### 7.5 随机 baseline

```bash
bash scripts/run_sd3_random_baseline.sh
```

生成同预算 uniform random $L_\infty$ 扰动，作为阈值和 sanity baseline。

### 7.6 权重消融

```bash
bash scripts/run_sd3_ablation_weights.sh cuda:0 C 512 5
```

通过 `collect_sd3_data.py --extra` 透传：

```text
attack.textual_weight=<TW>
attack.mmdit_weight=<MW>
```

---

## 8. 工程妥协与风险

### 8.1 显存风险

SD3.5 medium 在 24GB GPU 上可能 OOM。当前代码提供 `capture_blocks` 降低 hook 成本，但没有真正启用 CPU offload 或禁用 T5。`low_memory` 字段目前只是预留。

### 8.2 计算开销

- Mode B 每 step 计算 clean feature；
- Mode A 每 step 额外计算 clean attention injection；
- paired SDEdit 会使 denoising 成本约翻倍。

### 8.3 理论 proxy

- Mode A 已实现 injection divergence，但 entropy 仍作为 regularizer；
- Mode D 使用 covariance-CKA proxy，不是完整 CKA estimator；
- FID/KID 需要足够样本量，不能用于极小样本结论。

### 8.4 旧入口残留

`code/diff_mist_SD3.py` 仍保留历史逻辑。论文实验应使用 `code/diff_mist_SD3_v2.py`。后续可考虑清理旧入口，或在 README 中明确说明。

---

## 9. 验证与交付标准

进入正式实验前必须满足：

```bash
python scripts/validate_sd3_pipeline.py
```

通过后，再进行：

```bash
python scripts/validate_sd3_pipeline.py --run-smoke --device cuda:<id>
```

然后检查：

- `grad_mmdit_l2 > 0`；
- `attn_injection_l2` 对 Mode A 有值；
- `feature_cos` 对 Mode B 有值；
- `velocity_div` 对 Mode C / O_fair 有值；
- `modality_ratio_dev` 与 `cross_modal_cka` 对 Mode D 有值；
- paired clean/adv SDEdit 图片成对出现；
- metrics CSV 和 summary CSV 不为空。
