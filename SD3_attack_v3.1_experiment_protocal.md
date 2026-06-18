# SD3 攻击 v3.1 实验协议

> 文件名沿用用户指定的 `protocal` 拼写。  
> 目标：给 SD3 Attack v3.1 的实验实现、日志、指标、输出和诊断提供可执行协议。  
> 范围：本文档只定义实验协议，不要求当前立即修改代码或运行实验。
> 命名口径：O/A/B/C/D 均指 v2.1 joint objective，即 textual target-pull 与对应 MMDiT 项的联合目标；主实验统一设置 `textual_weight=1`、`mmdit_weight=100000`。v2.1 代码内部的 `O_repo` 只作为历史别名理解，本文和后续实验不再使用 `O_repo` 或 `O_fair` 作为方法名。新增 `E` 模式表示从 v2.1 O 中拆出的 semantic-only velocity-norm loss，只优化 \(L_E=\|v_\theta(z_p,t,c)\|_2\)，不加入 textual target-pull。纯 textual target-pull baseline 单独命名为 `textual_only`。

---

## 1. 实验目标

v3.1 的实验目标是验证 SD3 流匹配防护（SD3 Flow-Matching Protection, FMP）是否能在不可感知扰动约束下破坏 SD3/SD3.5 图像编辑结果。

核心假设为：

\[
\max_{\|\delta\|_\infty\le\epsilon}L_{\mathrm{FMP}}(x+\delta)
\]

能够比 `textual_only`、E 以及 \(mmdit\_weight=100000\) 的 O/A/B/C/D 更稳定地扩大成对 SDEdit 输出差异：

\[
D_{\mathrm{edit}}
\left(
\operatorname{Edit}_\theta(x+\delta;\omega),
\operatorname{Edit}_\theta(x;\omega)
\right).
\]

实验必须同时回答四个问题：

1. 约束是否满足：扰动是否在 \(L_\infty\) 预算内，并且视觉上不可感知；
2. 编辑是否失败：成对 clean/adv SDEdit 输出是否显著不同；
3. 机制是否合理：FMP 损失和梯度是否真实影响 SD3 速度预测；
4. 方法是否优于基线：FMP 是否显著优于 `textual_only`、E 和 \(mmdit\_weight=100000\) 的 O/A/B/C/D。

---

## 2. 方法列表

所有方法必须在相同图像、prompt、epsilon、steps、seed、SDEdit 噪声强度、input size 下进行成对比较。

### 2.1 随机 L-infinity sanity baseline

随机扰动：

\[
\delta\sim \operatorname{Uniform}(-\epsilon,\epsilon),
\quad
x_p=\operatorname{clip}(x+\delta).
\]

作用：验证观察到的编辑差异是否只是 \(L_\infty\) 噪声预算自然带来的结果。该方法不进入 3.2 主实验网格，只作为小规模 sanity baseline 或附录结果。

### 2.2 Textual-only baseline：`textual_only`

该基线只保留 textual target-pull，不加入任何 MMDiT 结构项或 semantic velocity-norm 项：

\[
L_{\mathrm{textual}}
=
L_T^*
=
-\operatorname{MSE}(z_p,z_y),
\]

其中：

- \(z_p=E(x_p)\) 是受保护图像的 VAE latent；
- \(z_y=E(y)\) 是 target image 的 VAE latent；
- 负号表示在 `opt_direction=maximize` 约定下，最大化 \(L_T^*\) 等价于最小化 \(\operatorname{MSE}(z_p,z_y)\)，即把 protected latent 拉向 target latent。

作用：隔离 textual target-pull 本身的编辑破坏能力，作为 E、O/A/B/C/D 与 FMP 的基础对照。

### 2.3 v2.1 O：textual + velocity-norm joint objective

`O` 只指 v2.1 的 `mode=O`，其 MMDiT 项是 velocity-norm proxy：

\[
L_O
=
\|v_\theta(z_p,t,c)\|.
\]

其中 \(z_p=E(x_p)\)，\(t\) 是采样 timestep，\(c\) 是 prompt 条件。它不是正确的 SD3 流匹配语义损失，因为它没有使用目标速度 \(u_t\)，只优化预测速度范数。

作为主实验方法时，`O` 按 v2.1 joint 口径表示：

\[
L_{\mathrm{v2.1},O}
=
\lambda_T L_T^*+\lambda_M L_O .
\]

主实验固定 \(\lambda_T=1,\lambda_M=100000\)。因此 `O` 已经包含 textual target-pull 与 semantic velocity-norm 两部分。`textual_only` 只用于去掉 \(L_O\) 后的纯 textual 对照，`E` 用于去掉 textual target-pull 后的 semantic-only 对照。

### 2.4 E：semantic-only velocity-norm baseline

E 从 v2.1 O 中拆分出 semantic velocity-norm 分量，并只优化该项：

\[
L_E
=
\|v_\theta(z_p,t,c)\|.
\]

E 不包含 textual target-pull：

\[
L_E\neq \lambda_TL_T^*+\lambda_ML_O.
\]

作用：验证 O 的 semantic velocity-norm proxy 单独优化时是否能造成编辑防护效果，并与 O 的 joint objective 区分。

### 2.5 v2.1 A/B/C/D joint 机制目标

A/B/C/D 均以 v2.1 mode 定义为准。每个 mode 的 MMDiT 机制项分别为：

- A：文本注入破坏；
- B：图像流特征差异；
- C：共享噪声速度 / 轨迹差异；
- D：模态平衡破坏。

作为主实验方法时，它们遵循 v2.1 joint objective：

\[
L_{\mathrm{v2.1},M}
=
\lambda_T L_T^*+\lambda_M L_M,
\quad M\in\{A,B,C,D\}.
\]

主实验固定 \(\lambda_T=1,\lambda_M=100000\)，用于检验被放大的 MMDiT 结构项是否能在 paired SDEdit 外部指标上产生可见影响。该设置是实验放大策略，不表示 v2.1 默认权重为 100000。

### 2.6 v3.1 FMP 单 sigma

每个 PGD step 采样或固定一个 \(\sigma\)，优化：

\[
\ell_{\mathrm{FMP}}
=
\frac{
\|v_\theta(z^p_\sigma,t_\sigma,c)-\operatorname{sg}(u_\sigma)\|_2^2
}{
\operatorname{sg}(\|u_\sigma\|_2^2)+\eta
}.
\]

作用：验证最低计算成本的 FMP 是否有效。

### 2.7 v3.1 FMP 多 sigma 期望

每个 PGD step 采样 \(m\) 个 \(\sigma\)：

\[
\widehat L_{\mathrm{FMP}}
=
\frac{1}{m}
\sum_{j=1}^m
\ell_{\mathrm{FMP}}(x_p;\sigma_j,\xi_j,c_j).
\]

作用：降低单 sigma 估计的方差，检验多编辑强度下的泛化保护能力。

### 2.8 v3.1 FMP + 可选单步转移分离

定义单步转移：

\[
T_\theta(z,t,c)=z+h_t v_\theta(z,t,c).
\]

辅助项：

\[
L_{\mathrm{step}}
=
\frac{
\|T_\theta(z^p_\sigma,t_\sigma,c)
-\operatorname{sg}(T_\theta(z_\sigma,t_\sigma,c))\|_2^2
}{
\operatorname{sg}(\|T_\theta(z_\sigma,t_\sigma,c)\|_2^2)+\eta
}.
\]

总目标可写为：

\[
L=L_{\mathrm{FMP}}+\lambda_{\mathrm{step}}L_{\mathrm{step}}.
\]

该方法只作为后续消融，不作为 v3.1 主方法。若未先证明 FMP 单独有效，不应优先投入该方案。

---

## 3. 默认实验网格

### 3.1 输入图像

默认图像：

```text
test_images/to_protect/suzume.png
test_images/to_protect/dinopark.png
test_images/to_protect/cat.jpg
```

表格中使用 image id：

```text
suzume
dinopark
cat
```

### 3.2 攻击参数

默认主实验网格：

| 参数 | 取值 |
|---|---|
| methods | textual_only, E, O, A, B, C, D, FMP_single, FMP_multi |
| epsilon | 4, 8, 16 |
| steps | 50, 100 |
| seeds | 0, 1, 2 |
| input_size | 512 |
| opt_direction | maximize |
| random_start | true |
| textual_weight | 1 |
| mmdit_weight | 100000 for O/A/B/C/D |

epsilon 以 0-255 像素尺度记录，同时在日志中记录对应 tensor-space 预算：

\[
\epsilon_{\mathrm{tensor}}=\frac{2\epsilon}{255}.
\]

### 3.3 成对 SDEdit 参数

默认评估网格：

| 参数 | 取值 |
|---|---|
| sdedit_noise_levels | 0.1, 0.3, 0.5 |
| sdedit_steps | 28 |
| same_noise | true |
| same_prompt | true |
| same_seed | true |
| guidance_scale | 记录实际配置 |

成对 SDEdit 必须对 clean 和 protected 使用同一 \(\xi\)，否则无法把输出差异归因给扰动。

---

## 4. FMP PGD 伪代码

```text
输入:
    原始图像 x
    prompt 集合 C
    扰动预算 epsilon
    PGD 步数 K
    步长 alpha
    sigma 分布 p(sigma)
    latent 噪声分布 xi ~ N(0, I)
    冻结的 SD3/SD3.5 组件 E, v_theta, scheduler

输出:
    受保护图像 x_p
    loss 与诊断日志

初始化:
    if random_start:
        delta_0 ~ Uniform(-epsilon, epsilon)
        x_p = clip(x + delta_0)
    else:
        delta_0 = 0
        x_p = x

对 k = 1 ... K:
    1. 对 x_p 开启梯度。

    2. 编码受保护图像:
        z_p = E(x_p)

    3. 采样编辑条件:
        sigma ~ p(sigma)
        xi ~ N(0, I)
        c ~ prompt distribution C
        t_sigma = scheduler.map_sigma_to_timestep(sigma)

    4. 构造 SDEdit 加噪 latent:
        z_p_sigma = (1 - sigma) * z_p + sigma * xi

    5. 计算 SD3 速度预测:
        v_pred = v_theta(z_p_sigma, t_sigma, c)

    6. 计算流匹配目标速度:
        u_sigma = flow_target(z_p, xi, sigma, scheduler)
        对 u_sigma 使用 stop-gradient

    7. 计算归一化 FMP loss:
        loss_fmp = ||v_pred - sg(u_sigma)||_2^2 / (sg(||u_sigma||_2^2) + eta)

    8. 可选：只做诊断:
        在 FMP run 中计算 O/A/B/C/D 机制指标作为诊断，但不加入 FMP 主 loss

    9. 反向传播:
        grad = gradient(loss_fmp, x_p)

    10. Sign-PGD 更新:
        x_p = x_p + alpha * sign(grad)

    11. 投影:
        x_p = min(max(x_p, x - epsilon), x + epsilon)
        x_p = clip(x_p, -1, 1)

    12. 记录日志:
        loss_fmp, grad_fmp_l2, grad_total_l2, linf, sigma, timestep, seed, prompt

返回 x_p
```

实现注意：`flow_target(z_p, xi, sigma, scheduler)` 必须遵循实际 SD3 scheduler 与 diffusers 约定。正式实验前必须用工程实现确认 \(u_\sigma\) 的符号和尺度。

---

## 5. 攻击生成协议

### 5.1 固定条件

同一组成对比较中必须固定：

1. 输入图像；
2. 图像 resize/crop 策略；
3. prompt；
4. epsilon；
5. PGD 步数；
6. seed；
7. 模型 checkpoint；
8. VAE scale 约定；
9. scheduler；
10. 输出精度策略。

若任一条件不同，该结果只能作为独立实验，不应进入成对比较。

### 5.2 步长

默认 step size 可从当前工程沿用：

\[
\alpha=\frac{1}{255}
\]

或使用与 tensor range 匹配的配置。日志中必须同时记录原始 alpha 和 tensor-space alpha。

### 5.3 随机初始化

主实验默认使用 `random_start=true`。原因是 B/C 等 clean-adv 差异项在 \(\delta=0\) 附近可能出现弱梯度或零梯度；若主实验不使用 random start，可能只是在测试优化启动失败，而不是测试机制目标本身的防护能力。

\[
\delta_0\sim\operatorname{Uniform}(-\epsilon,\epsilon).
\]

当前阶段不进行随机初始化消融，也不要求保留该消融作为主实验前置项。所有主实验、minimal 实验和 full modes 实验均固定 `random_start=true`。若后续出现明显的优化启动异常，再单独设计启动性诊断实验；该诊断不进入当前主表计划。

### 5.4 Prompt 协议

每张图像必须有固定默认 prompt。若使用多个 prompt，则 manifest 中记录：

```text
prompt_id
prompt_text
negative_prompt
prompt_source
```

prompt 消融至少包含：

1. 保持主体的编辑；
2. 风格编辑；
3. 背景编辑；
4. 恶意语义编辑，前提是课程规范允许以安全方式处理。

---

## 6. 成对 SDEdit 评估协议

### 6.1 Clean / Protected 成对样本

对每次运行，必须生成：

\[
y_{\mathrm{clean}}
=
\operatorname{Edit}_\theta(x;\omega),
\]

\[
y_{\mathrm{prot}}
=
\operatorname{Edit}_\theta(x_p;\omega).
\]

其中 \(\omega\) 完全相同，包括 prompt、seed、noise、sigma、steps 和 guidance。

### 6.2 加噪 latent 成对样本

clean 侧：

\[
z_\sigma=(1-\sigma)E(x)+\sigma\xi.
\]

protected 侧：

\[
z^p_\sigma=(1-\sigma)E(x_p)+\sigma\xi.
\]

这里必须使用同一个 \(\xi\)。如果 clean 和 protected 分别随机采样噪声，则成对比较失效。

### 6.3 保存要求

每个 sigma 至少保存：

```text
<image_id>_sdedit_clean_noise_<sigma>.png
<image_id>_sdedit_adv_noise_<sigma>.png
```

也可以额外保存 latent 或中间预览，但论文指标只依赖成对 clean/adv 输出。

---

## 7. 消融实验设计

### 7.1 主方法对比

表格维度：

```text
method x epsilon x steps x sigma
```

方法：

```text
textual_only
E
O
A
B
C
D
FMP_single
FMP_multi
```

目标：证明 FMP 在编辑破坏指标上优于 `textual_only`、E 以及 \(mmdit\_weight=100000\) 的 O/A/B/C/D。random 不进入主方法对比，只作为 sanity baseline 或附录结果。

### 7.2 Sigma 消融

FMP_single 固定不同优化 sigma：

```text
sigma_train = 0.1
sigma_train = 0.3
sigma_train = 0.5
```

评估仍在：

```text
sigma_eval = 0.1, 0.3, 0.5
```

目标：判断训练 sigma 是否迁移到其他编辑强度。

### 7.3 多 sigma 样本数消融

比较：

```text
m = 1, 2, 3
```

其中 \(m\) 是每个 PGD step 采样的 sigma/noise 数量。报告效果与运行时间。

### 7.4 随机初始化消融（暂缓）

当前阶段暂不进行该消融。主实验统一使用：

```text
random_start = true
```

若后续日志显示某些方法 `grad_total_l2` 长期接近 0、loss 不变化或扰动无法离开初始状态，再单独设计小规模启动诊断。该诊断只用于定位优化启动问题，不作为当前论文主消融。

### 7.5 文本辅助项消融

当前阶段暂不进行该消融。FMP 主方法保持不加入 textual target-pull，以避免重新引入 v2.1 中 textual 梯度主导的问题。

后续若恢复该消融，再比较：

```text
FMP only
FMP + small textual
O
```

若加入文本项，必须记录梯度比例：

\[
\frac{\lambda_T\|\nabla L_T\|_2}{\|\nabla L_{\mathrm{FMP}}\|_2}.
\]

若文本项再次主导梯度，则不得声称结果来自 FMP。

### 7.6 单步转移消融

该消融选择性进行：代码、日志字段和运行脚本需要预留实现；是否实际运行取决于 FMP_single / FMP_multi 主结果和成本收益。若主方法已经清晰优于基线，且 GPU 预算有限，可不运行该消融；若 FMP loss 上升但 paired SDEdit 差异不足，优先运行该消融检查局部 trajectory separation 是否能补强。

比较：

```text
FMP
FMP + L_step
```

实现要求：

1. 攻击代码支持 `use_step_loss=true/false` 与 `lambda_step`；
2. 日志记录 `loss_step`, `lambda_step`, `transition_sep`, `grad_step_l2`；
3. 运行脚本提供单独入口，例如 `run_sd3_v3_step_ablation.sh`，不混入 full modes 主表；
4. 分析脚本在 `ablation_summary.csv` 中单独汇总 `FMP` 与 `FMP + L_step`。

预期效果：若 FMP 的 prediction error 已经能传播到编辑轨迹，`L_step` 只应带来小幅或不显著提升；若 FMP 只改变局部预测误差但外部编辑差异不足，`L_step` 可能提高 edit RMSE / LPIPS，但运行时间会增加。

### 7.7 鲁棒性消融（暂缓）

本阶段暂不执行鲁棒性消融。该部分只保留为后续扩展协议，不进入当前实验计划。

对受保护图像做轻量变换后再编辑：

```text
JPEG quality = 95, 85, 75
resize down/up
center crop
VAE round-trip
```

目标：评估扰动是否容易被常见图像处理破坏。

---

## 8. 日志协议

每个 PGD run 必须保存 loss 历史，推荐为 `.npz` 和 `.jsonl` 双格式。

### 8.1 必需字段

每个 step 记录：

| 字段 | 含义 |
|---|---|
| `step` | PGD step 索引 |
| `loss_total` | 实际用于反向传播的总 loss |
| `loss_fmp` | FMP modes 的 FMP loss；非 FMP 方法可为空或记为 NaN |
| `loss_o` | velocity-norm proxy；用于 `O` 的 semantic 分量，也用于 E 的主 loss 记录 |
| `loss_textual_mse` | textual target-pull 的 MSE 分量；用于 `textual_only` 和 O/A/B/C/D |
| `loss_mechanism` | A/B/C/D 的 MMDiT 机制项；其他方法可为空或记为 NaN |
| `textual_weight` | O/A/B/C/D joint objective 中的 textual 权重 |
| `mmdit_weight` | O/A/B/C/D joint objective 中的 MMDiT 权重，主实验为 100000 |
| `grad_fmp_l2` | FMP modes 的 \(\|\nabla_x L_{\mathrm{FMP}}\|_2\)；非 FMP 方法可为空或记为 NaN |
| `grad_total_l2` | \(\|\nabla_x L_{\mathrm{total}}\|_2\) |
| `linf` | 当前 \(\|x_p-x\|_\infty\) |
| `sampled_sigma` | 当前 step 使用的 \(\sigma\) |
| `sampled_timestep` | 与 \(\sigma\) 对应的 scheduler timestep |
| `seed` | run seed |
| `prompt_id` | prompt id |
| `method` | 方法名 |
| `epsilon` | 像素空间 epsilon |
| `steps` | PGD 总步数 |

### 8.2 推荐字段

| 字段 | 含义 |
|---|---|
| `u_norm` | \(\|u_\sigma\|_2\) |
| `v_pred_norm` | \(\|v_\theta(z^p_\sigma,t_\sigma,c)\|_2\) |
| `prediction_error` | \(\|v_\theta-u_\sigma\|_2^2\) |
| `loss_step` | 可选单步转移消融的 \(L_{\mathrm{step}}\) |
| `lambda_step` | 可选单步转移项权重 |
| `transition_sep` | \(T_\theta(z^p_\sigma,t_\sigma,c)\) 与 clean/reference transition 的距离 |
| `grad_step_l2` | 可选单步转移项对输入图像的梯度范数 |
| `alpha` | PGD step size |
| `epsilon_tensor` | tensor-space epsilon |
| `vae_scale` | VAE latent 缩放约定 |
| `scheduler_name` | scheduler 标识 |
| `model_name` | SD3/SD3.5 checkpoint |
| `precision` | fp16/bf16/fp32 |

### 8.3 E、O 与 A/B/C/D 诊断字段

若做机制诊断，记录：

| 模式 | 字段 |
|---|---|
| E | `loss_o`, `v_pred_norm` |
| O | `loss_textual_mse`, `loss_o`, `textual_weight`, `mmdit_weight`, `v_pred_norm` |
| A | `loss_textual_mse`, `loss_mechanism`, `textual_weight`, `mmdit_weight`, `attn_entropy`, `attn_injection_l2` |
| B | `loss_textual_mse`, `loss_mechanism`, `textual_weight`, `mmdit_weight`, `feature_cos`, `feature_gram_l1` |
| C | `loss_textual_mse`, `loss_mechanism`, `textual_weight`, `mmdit_weight`, `velocity_cos`, `velocity_div` |
| D | `loss_textual_mse`, `loss_mechanism`, `textual_weight`, `mmdit_weight`, `modality_ratio_dev`, `cross_modal_cka` |

这些字段只解释内部机制，不作为主优化成功证据。

### 8.4 Manifest 字段

每个 run 记录到 `manifest.csv`：

```text
run_id
method
image_id
image_path
prompt_id
prompt_text
epsilon
steps
seed
input_size
sdedit_noise_levels
sdedit_steps
model_name
scheduler_name
output_dir
loss_path
metrics_path
status
elapsed_sec
error_message
git_commit
```

---

## 9. 指标定义

### 9.1 扰动约束指标

#### L-infinity 范数

\[
L_\infty^{[0,1]}=\|x_p-x\|_\infty.
\]

\[
L_\infty^{[-1,1]}=2L_\infty^{[0,1]}.
\]

要求：

\[
L_\infty^{[-1,1]}\le \frac{2\epsilon}{255}.
\]

#### 扰动 RMSE

\[
\operatorname{RMSE}_{\mathrm{perturb}}
=
\sqrt{
\frac{1}{N}
\sum_i
(x_{p,i}-x_i)^2
}.
\]

#### 扰动 PSNR

若图像在 \([0,1]\) 空间：

\[
\operatorname{PSNR}_{\mathrm{perturb}}
=
20\log_{10}
\frac{1}{\operatorname{RMSE}_{\mathrm{perturb}}}.
\]

#### 扰动 SSIM / LPIPS

SSIM 越高表示越接近，LPIPS 越低表示感知差异越小。论文中应同时报告至少一个像素指标和一个感知指标。

### 9.2 编辑破坏指标

定义成对输出：

\[
y_{\mathrm{clean}}=\operatorname{Edit}_\theta(x;\omega),
\]

\[
y_{\mathrm{prot}}=\operatorname{Edit}_\theta(x_p;\omega).
\]

#### 编辑 RMSE

\[
\operatorname{RMSE}_{\mathrm{edit}}
=
\sqrt{
\frac{1}{N}
\sum_i
(y_{\mathrm{prot},i}-y_{\mathrm{clean},i})^2
}.
\]

越大表示 protected edit 越偏离 clean edit。

#### 编辑 PSNR

\[
\operatorname{PSNR}_{\mathrm{edit}}
=
20\log_{10}
\frac{1}{\operatorname{RMSE}_{\mathrm{edit}}}.
\]

越低表示差异越大。

#### 编辑 SSIM

\[
\operatorname{SSIM}_{\mathrm{edit}}
=
\operatorname{SSIM}(y_{\mathrm{prot}},y_{\mathrm{clean}}).
\]

越低表示结构越不同。

#### 编辑 LPIPS

\[
\operatorname{LPIPS}_{\mathrm{edit}}
=
\operatorname{LPIPS}(y_{\mathrm{prot}},y_{\mathrm{clean}}).
\]

越高表示感知差异越大。

#### CLIP Prompt 下降

设：

\[
S_{\mathrm{prompt}}(y,c)=\operatorname{CLIPScore}(y,c).
\]

定义：

\[
\Delta_{\mathrm{prompt}}
=
S_{\mathrm{prompt}}(y_{\mathrm{prot}},c)
-
S_{\mathrm{prompt}}(y_{\mathrm{clean}},c).
\]

若 \(\Delta_{\mathrm{prompt}}<0\)，表示 protected edit 与 prompt 的匹配度低于 clean edit。

#### 源图相似度下降

设：

\[
S_{\mathrm{src}}(y,x)=\operatorname{CLIPImageSim}(y,x).
\]

定义：

\[
\Delta_{\mathrm{src}}
=
S_{\mathrm{src}}(y_{\mathrm{prot}},x)
-
S_{\mathrm{src}}(y_{\mathrm{clean}},x).
\]

该指标用于判断 protected edit 是否丢失原图身份或主体结构。

### 9.3 统计指标

每个分组报告：

```text
mean
std
median
bootstrap 95% CI
n
```

成对方法比较使用同一 image、epsilon、steps、seed、sigma 下的差值：

\[
\Delta M
=
M_{\mathrm{FMP}}-M_{\mathrm{baseline}}.
\]

推荐报告：

```text
mean paired difference
95% bootstrap CI
paired sign test or paired t-test
effect size
```

若样本较少，以 bootstrap CI 和成对差值为主，不做过强显著性结论。

---

## 10. 输出目录协议

### 10.1 单次运行目录

推荐格式：

```text
out_sd3_v3/
  <method>_eps<epsilon>_steps<steps>_seed<seed>/
    <image_id>/
      attacked.png
      original.png
      loss.npz
      metrics.json
      config.json
      <image_id>_sdedit_clean_noise_0.1.png
      <image_id>_sdedit_adv_noise_0.1.png
      <image_id>_sdedit_clean_noise_0.3.png
      <image_id>_sdedit_adv_noise_0.3.png
      <image_id>_sdedit_clean_noise_0.5.png
      <image_id>_sdedit_adv_noise_0.5.png
```

### 10.2 聚合输出

```text
out_sd3_v3/
  manifest.csv
  full_metrics.csv
  summary_by_method.csv
  ablation_summary.csv
  paired_differences.csv
  figures/
    method_vs_epsilon_edit_rmse.png
    method_vs_sigma_edit_lpips.png
    perturbation_quality_tradeoff.png
    loss_curves_fmp.png
    grad_norm_curves.png
    paired_examples_grid.png
```

### 10.3 文件含义

| 文件 | 含义 |
|---|---|
| `attacked.png` | 受保护图像 \(x_p\) |
| `original.png` | 原始图像 \(x\)，用于校验 |
| `loss.npz` | step-level loss 和诊断指标 |
| `metrics.json` | run-level 指标 |
| `config.json` | 完整配置快照 |
| `manifest.csv` | 所有 runs 的索引和状态 |
| `full_metrics.csv` | 每个 image/sigma/run 的 row-level 指标 |
| `summary_by_method.csv` | 方法级聚合指标 |
| `ablation_summary.csv` | 消融实验汇总 |
| `paired_differences.csv` | FMP 与基线的成对差值 |

---

## 11. 失败诊断协议

### 11.1 `grad_fmp_l2` 近似为 0

若：

\[
\|\nabla_x L_{\mathrm{FMP}}\|_2\approx 0,
\]

检查：

1. \(u_\sigma\) 是否被错误计算为与 \(v_\theta\) 完全相同；
2. scheduler 符号约定是否反了；
3. \(t_\sigma\) 是否与 \(\sigma\) 不匹配；
4. VAE latent 缩放是否错误；
5. `z_p` 或 `v_pred` 是否被意外 detach；
6. autocast / fp16 是否导致数值下溢；
7. loss 是否在 `.item()` 后重新包装，导致梯度断路。

### 11.2 FMP loss 上升但外部编辑不变

若 \(L_{\mathrm{FMP}}\) 上升，但成对 SDEdit 指标没有变化，检查：

1. 优化使用的 \(\sigma\) 是否不在评估 sigma 范围；
2. 优化状态是否与实际 SDEdit 加噪 latent 构造一致；
3. 预测误差是否只增加在高噪声但被后续 denoising 修复的区域；
4. prompt 是否过弱或编辑任务过简单；
5. 编辑指标是否被 image_id 或 sigma 混杂因素主导；
6. 是否需要 multi-\(\sigma\) 或 prompt sampling。

### 11.3 扰动不可见性差

若 \(x_p\) 视觉质量差或 LPIPS 过高，检查：

1. epsilon 是否过大；
2. step size 是否过大；
3. 是否开启 random start；
4. PGD 投影是否在正确 tensor range；
5. attacked image 保存时是否存在重复归一化或反归一化错误。

### 11.4 方法差异被 epsilon 主导

若结果主要由 epsilon 决定，方法效应很小，必须使用：

1. 成对比较；
2. 控制 epsilon/sigma/image_id 后的残差分析；
3. 按 image 和 sigma 分组的细分结果；
4. perturbation cosine / sign agreement analysis。

不能只看总体均值。

### 11.5 O/A/B/C/D/E 指标变化但编辑不变

若 O/A/B/C/D/E 机制指标变化，但成对 SDEdit 不变，结论应为：

```text
该机制项可被扰动改变，但当前证据不足以证明其能造成编辑防护效果。
```

不得写成：

```text
该方法成功防护。
```

---

## 12. 论文表图协议

### 12.1 主表

主表按 method 汇总：

```text
method
epsilon
perturb_Linf
perturb_LPIPS
edit_RMSE
edit_LPIPS
CLIP_prompt_drop
source_similarity_drop
success_rate
```

每个指标报告：

```text
mean ± std
95% CI
n
```

预期效果：

1. 所有方法的 `perturb_Linf` 均不超过预算，`perturb_LPIPS` 维持在可接受范围；
2. FMP_single / FMP_multi 的 `edit_RMSE`、`edit_LPIPS`、`source_similarity_drop` 应高于 `textual_only`、E 和 \(mmdit\_weight=100000\) 的 O/A/B/C/D；
3. FMP_multi 预期比 FMP_single 更稳定，表现为跨 seed / sigma 的 std 更小或 bootstrap CI 更窄；
4. 若 O/A/B/C/D/E 指标变化但编辑破坏指标不显著，应解释为机制项或 semantic proxy 可被扰动改变，但当前证据不足以证明其能造成有效防护。

### 12.2 消融表

当前阶段消融表包含：

1. FMP single 与 FMP multi 对比；
2. sigma_train 消融；
3. 多 sigma 样本数消融；
4. 可选单步转移消融，仅在选择性运行时进入消融表。

当前阶段不包含 random_start 消融和文本辅助项消融。

预期效果：

1. FMP_multi 相比 FMP_single 应降低对单一 \(\sigma\) 的过拟合，尤其在非训练 sigma 上保持更高编辑破坏；
2. sigma_train 消融应显示中等或覆盖评估区间的 \(\sigma\) 更稳健；若某个单一 \(\sigma\) 只在对应 eval sigma 有效，说明目标迁移性不足；
3. 多 sigma 样本数 \(m\) 增大时，预期效果更稳定但运行成本上升；若 \(m=2/3\) 无明显提升，应优先采用低成本配置；
4. 可选 `FMP + L_step` 若运行，预期只在 FMP 外部编辑差异不足时提供补强；若提升不显著，应保持 FMP 作为主方法。

### 12.3 图像网格

每组定性图包含：

```text
original x
protected x_p
clean edit y_clean
protected edit y_prot
absolute difference heatmap
```

同一图中必须标注 method、epsilon、sigma、seed。

预期效果：

1. `protected x_p` 与 `original x` 视觉差异应很小，扰动不可成为肉眼明显噪声；
2. FMP 的 `protected edit y_prot` 应相对 `clean edit y_clean` 出现更明显的结构、语义或局部编辑失败；
3. `absolute difference heatmap` 应主要反映编辑结果偏移，而不是原始扰动本身的大面积可见伪影；
4. 若 O/A/B/C/D/E 的 heatmap 很弱但内部机制指标变化明显，应在图注中避免宣称其防护成功。

### 12.4 曲线图

推荐曲线：

1. `loss_total` vs PGD step，按 method 分组；
2. `loss_fmp` vs PGD step，仅用于 FMP modes；
3. `grad_fmp_l2` vs PGD step，仅用于 FMP modes；
4. edit RMSE vs epsilon；
5. edit LPIPS vs sigma；
6. perturb LPIPS 与 edit LPIPS 的权衡；
7. FMP 与基线的成对差值分布。

预期效果：

1. `loss_total` vs PGD step：各方法的优化目标应整体可上升或稳定收敛；若 loss 不变且 `grad_total_l2≈0`，说明实现或启动存在问题；
2. `loss_fmp` vs PGD step：FMP modes 中应随 PGD step 上升；若 FMP loss 上升但编辑指标不变，应检查 \(\sigma\) 分布和 flow target；
3. `grad_fmp_l2` vs PGD step：应为有限非零值，不应长期 NaN/Inf 或接近 0；
4. edit RMSE vs epsilon：预期 epsilon 增大时编辑破坏增强，但扰动可见性也可能变差；
5. edit LPIPS vs sigma：预期 FMP_multi 在不同 eval sigma 上曲线更平滑，FMP_single 可能在训练 sigma 附近更强；
6. perturb LPIPS 与 edit LPIPS 权衡：理想方法应在较低 perturb LPIPS 下取得较高 edit LPIPS；
7. 成对差值分布：FMP 相对基线的差值应多数大于 0，且 bootstrap CI 不跨 0 或跨 0 较少。

### 12.5 负结果呈现

若 FMP 没有显著优于基线，应如实报告，并定位：

1. 目标函数是否实现错误；
2. \(\sigma\) 分布是否错误；
3. flow target \(u_\sigma\) 是否错误；
4. SD3 scheduler 约定是否错误；
5. 指标是否不能反映实际编辑失败；
6. 是否需要更强威胁模型或更合理 prompt 集合。

---

## 13. 验收标准

正式主实验开始前，必须满足以下 sanity checks：

1. `attacked.png` 存在且 \(L_\infty\) 不超过预算；
2. `loss.npz` 对所有方法包含 `loss_total`, `grad_total_l2`, `linf`, `sampled_sigma`, `sampled_timestep`；
3. `loss.npz` 对方法专属字段满足：`textual_only` 包含 `loss_textual_mse`；`E` 包含 `loss_o` 且不包含 `loss_textual_mse`；`O` 包含 `loss_textual_mse`, `loss_o`, `textual_weight`, `mmdit_weight`；A/B/C/D 包含 `loss_textual_mse`, `loss_mechanism`, `textual_weight`, `mmdit_weight` 和对应机制字段；FMP modes 包含 `loss_fmp` 和 `grad_fmp_l2`；
4. `grad_total_l2` 在首步和末步均不是 NaN/Inf；FMP modes 的 `grad_fmp_l2` 也必须不是 NaN/Inf；
5. paired clean/adv SDEdit 图像成对存在；
6. `full_metrics.csv` 每个 run 至少有三个 sigma rows；
7. method naming 能区分 textual_only、E、O、A/B/C/D、FMP_single、FMP_multi；
8. 统计脚本不会把不同 epsilon、sigma、seed 混成非成对比较；
9. 当 O/A/B/C/D/E 作为主实验 methods 时，按独立攻击方法评估；当 O/A/B/C/D/E 作为 FMP diagnostics 时，不进入 FMP 主目标的成功判断；
10. 任何历史目录中的 `O_repo` 只能作为 legacy alias 读取，不得进入 v3.1 主实验方法名；旧的 `textual_semantic_joint` 命名不得作为新实验方法名；
11. 若启用可选单步转移消融，`loss.npz` 额外包含 `loss_step`, `lambda_step`, `transition_sep`, `grad_step_l2`，且 run 目录或 manifest 明确标记 `use_step_loss=true`。

---

## 14. 最小可执行实验顺序

为了控制成本，建议按以下顺序执行：

1. 单图单 seed sanity：
   ```text
   image=suzume
   epsilon=8
   steps=50
   seed=0
   sigma_eval=0.3
   methods=textual_only, E, O, FMP_single
   random_start=true
   ```

2. 三图三 seed minimal：
   ```text
   images=suzume,dinopark,cat
   epsilon=8,16
   steps=50
   seeds=0,1,2
   sigma_eval=0.1,0.3,0.5
   methods=textual_only, E, O, FMP_single
   random_start=true
   ```

3. 加入 \(mmdit\_weight=100000\) 的 v2.1 joint 机制方法：
   ```text
   methods=A,B,C,D
   textual_weight=1
   mmdit_weight=100000
   random_start=true
   ```

4. 加入 FMP_multi：
   ```text
   m=2 or 3
   ```

5. 选择性加入单步转移消融：
   ```text
   methods=FMP_single,FMP_single_plus_step
   use_step_loss=true
   lambda_step=<configured value>
   random_start=true
   ```
   该步骤只在主方法结果显示需要 trajectory separation 补强时运行；否则跳过。

6. 完整主表：
   ```text
   epsilon=4,8,16
   steps=50,100
   seeds=0,1,2
   methods=all main methods
   random_start=true
   ```

该顺序避免在 flow target 或 scheduler convention 尚未确认前浪费大量 GPU 时间。
