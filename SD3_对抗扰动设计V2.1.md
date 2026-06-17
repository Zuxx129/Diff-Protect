# SD3 对抗扰动设计 V2.1

> 适用分支：`fix/sd3-objective-sanity`  
> 主要代码：`code/attacks_SD3.py`、`code/diff_mist_SD3_v2.py`、`configs/attack/base_sd3.yaml`  
> 目标：给出与当前工程实现对齐的 SD3 / SD3.5 MMDiT 结构感知图像扰动建模、数学推导、可证明性边界、伪代码与实验解释。

---

## 0. 版本定位

V2.1 不再把 A/B/C/D 写成松散的四个 loss，而是将它们统一为一个白盒、结构感知、可诊断的 $L_\infty$ 约束优化问题：

$$
\max_{\lVert_\delta \rVert_\infty\le \epsilon}
L_{\mathrm{attack}}(x+\delta,c;\theta),
\quad
x_{adv}=\operatorname{clip}(x+\delta,-1,1),
$$

其中 $x$ 是受保护图像，$c$ 是文本条件，$\theta$ 是冻结的 SD3 / SD3.5 pipeline 参数。当前实现采用 larger-is-more-attack 约定：所有进入 PGD 的子目标均被定义成数值越大代表扰动目标越强。

本版本与当前代码的对应关系如下。

| 理论模块 | 当前实现状态 | 主要代码 |
|---|---|---|
| $L_T$ target-pull textual loss | 已实现 `toward_target` / `away_from_target` | `SD3_Linf_PGD._textual_loss` |
| Mode A text injection disruption | 已实现 $A_{I\to T}V_T$ injection divergence + entropy regularizer | `AttnMapCaptureProcessor`、`cross_modal_disruption_loss` |
| Mode B image-stream feature divergence | 已实现同 timestep clean/adv normalized feature cosine + Gram divergence | `_compute_clean_features_at_timestep`、`feature_divergence_loss` |
| Mode C shared-noise flow trajectory divergence | 已实现 shared-noise $z_t$ 与 velocity cosine divergence | `_trajectory_loss_shared_noise` |
| Mode D modality balance disruption | 已实现 energy-ratio deviation + cross-modal covariance CKA proxy | `modality_imbalance_loss` |
| Paired SDEdit | 已在 v2 入口实现 clean/adv shared-noise paired output | `diff_mist_SD3_v2.py` |
| 指标与聚合 | 已实现 row-level metrics、diagnostics export、success aggregation、FID/KID optional | `code/metrics/*.py` |

---

## 1. 威胁模型

### 1.1 任务对象

输入为一张图像：

$$
x\in[-1,1]^{3\times H\times W}.
$$

攻击者生成不可感知扰动：

$$
\delta\in[-\epsilon,\epsilon]^{3\times H\times W},
\quad
x_{adv}=\Pi_{[-1,1]}(x+\delta).
$$

扰动预算在配置中以像素尺度给出，代码中映射到 $[-1,1]$ 张量空间：

$$
\epsilon_{tensor}=2\epsilon_{pixel}/255.
$$

### 1.2 攻击能力

当前建模为白盒图像保护场景，攻击者可访问冻结的：

- SD3 / SD3.5 VAE encoder；
- SD3Transformer / MMDiT blocks；
- text prompt embedding；
- flow-matching scheduler；
- 中间层 attention / feature hook。

攻击者不可修改模型参数，只优化输入图像扰动。

### 1.3 目标输出

对于同一 prompt、同一随机 seed、同一 SDEdit 噪声强度 $\sigma$，希望：

$$
\operatorname{Edit}_{\theta}(x_{adv},c,\sigma,s)
\not\approx
\operatorname{Edit}_{\theta}(x,c,\sigma,s),
$$

同时保持：

$$
\lVert x_{adv}-x\rVert_\infty\le \epsilon,
\quad
\operatorname{LPIPS}(x,x_{adv})\ \text{较低}.
$$

该目标不是通用图像破坏，而是授权图像保护中的编辑鲁棒性扰动。

---

## 2. SD3 / MMDiT 结构基础

SD v1.x 的主要 denoiser 是 UNet，经典 diffusion loss 往往围绕 $\epsilon$-prediction 或 noise residual 建模。SD3 / SD3.5 的核心变化是：图像 latent token 与文本 token 进入 MMDiT / joint transformer，并通过 flow matching / rectified flow 形式预测速度场。

抽象为：

$$
(z_t,c,t)\mapsto v_\theta(z_t,t,c),
$$

其中 $z_t$ 是 latent trajectory 中的状态，$v_\theta$ 是模型预测的速度。采样过程可写作 Euler 近似：

$$
z_{t-\Delta t}=z_t+\Delta t\cdot v_\theta(z_t,t,c).
$$

因此，面向 SD3 的结构感知扰动应作用在以下路径上：

1. VAE latent target relation；
2. Joint Attention 中 text token 到 image token 的注入；
3. MMDiT image-stream 表示；
4. flow trajectory velocity field；
5. image/text 双流能量与协方差平衡。

---

## 3. 统一优化问题与 PGD

### 3.1 Joint objective

当前实现统一为：

$$
L_{attack}=\lambda_T L_T+\lambda_M L_M,
$$

其中：

- $L_T$ 是 textual / VAE latent target loss；
- $L_M$ 是 MMDiT structural loss；
- $\lambda_T$ 对应 `textual_weight`；
- $\lambda_M$ 对应 `mmdit_weight`。

### 3.2 PGD 更新

设第 $k$ 步输入为 $x_k$：

$$
x_{k+1}=\Pi_{\mathcal B_\infty(x,\epsilon)\cap[-1,1]}
\left(x_k+s\alpha\operatorname{sign}(\nabla_x L_{attack}(x_k))\right),
$$

其中：

$$
s=
\begin{cases}
+1, & \texttt{opt\_direction=maximize},\\
-1, & \texttt{opt\_direction=minimize}.
\end{cases}
$$

V2.1 中默认使用：

```yaml
opt_direction: maximize
objective_convention: larger_is_more_attack
```

`g_mode` 只保留为向后兼容字段，不作为理论概念。

### 3.3 投影正确性证明

定义投影算子：

$$
\Pi(x')=\min\{\max\{x',x-\epsilon\},x+\epsilon\}
$$

随后再裁剪到 $[-1,1]$。对任意像素维度 $i$，有：

$$
x_i-\epsilon\le \Pi(x')_i\le x_i+\epsilon.
$$

因此：

$$
|\Pi(x')_i-x_i|\le \epsilon,
$$

从而：

$$
\lVert \Pi(x')-x\rVert_\infty\le \epsilon.
$$

这说明 PGD 每一步后均满足 $L_\infty$ 预算。

---

## 4. Textual loss

### 4.1 target-pull 建模

令 VAE encoder 为 $\mathcal E$：

$$
z_{adv}=\mathcal E(x+\delta),
\quad
z_y=\mathcal E(y),
$$

其中 $y$ 是错误 target image。Mist-compatible target-pull 的自然形式是：

$$
\min_\delta\lVert z_{adv}-z_y\rVert_2^2.
$$

为了统一最大化目标，代码实现为：

$$
L_T^{toward}=-\lVert z_{adv}-z_y\rVert_2^2.
$$

最大化 $L_T^{toward}$ 等价于最小化 target distance。

### 4.2 away-from-target 变体

若配置为：

```yaml
textual_objective: away_from_target
```

则：

$$
L_T^{away}=\lVert z_{adv}-z_y\rVert_2^2.
$$

它不再是 Mist target-pull，而是 target-repulsion 变体。

---

## 5. Mode A：Text Injection Disruption

### 5.1 数学定义

在 MMDiT joint attention 中，image-query 到 text-key 的 attention 记为：

$$
A_{I\to T}^{(l)}\in\mathbb R^{B\times H\times N_I\times N_T}.
$$

文本 value 表示为：

$$
V_T^{(l)}\in\mathbb R^{B\times H\times N_T\times d_h}.
$$

则 text token 注入到 image token 的分量为：

$$
O_{I\leftarrow T}^{(l)}=A_{I\to T}^{(l)}V_T^{(l)}.
$$

V2.1 使用：

$$
L_A=\frac{1}{|\mathcal B|}
\sum_{l\in\mathcal B}
\frac{\lVert O_{adv}^{(l)}-\operatorname{sg}(O_{clean}^{(l)})\rVert_2^2}
{\operatorname{sg}(\lVert O_{clean}^{(l)}\rVert_2^2)+\eta}
+
ho_A H(A_{adv}^{(l)}),
$$

其中 $\operatorname{sg}$ 为 stop-gradient，$H$ 是 attention entropy，当前实现中 $\rho_A=0.05$。

### 5.2 可证明性

attention 输出差异满足：

$$
O_{adv}-O_{clean}
=(A_{adv}-A_{clean})V_{clean}+A_{adv}(V_{adv}-V_{clean}).
$$

因此，只要扰动影响 attention 分布 $A$ 或 text value 表示 $V$，就会改变 text-to-image 注入通道。V2.1 的 $L_A$ 直接最大化该注入差异，因此该 loss 对 MMDiT joint attention 的跨模态注入具有结构针对性。

### 5.3 当前代码对应

- `AttnMapCaptureProcessor` 计算 `img_to_txt_attn`；
- 同时计算 `txt_injection = img_to_txt_attn @ encoder_hidden_states_value_proj`；
- `cross_modal_disruption_loss` 使用 clean/adv injection 的归一化 MSE，并记录 `attn_entropy` 与 `attn_injection_l2`。

---

## 6. Mode B：Image-stream Feature Divergence

### 6.1 数学定义

第 $l$ 层 image-stream feature：

$$
F_I^{(l)}(x,c)\in\mathbb R^{B\times N_I\times d}.
$$

归一化：

$$
\widehat F=\frac{F-\mu(F)}{\sigma(F)+\eta}.
$$

V2.1 loss：

$$
L_B=\frac{1}{|\mathcal B|}\sum_{l\in\mathcal B}
\left(1-\cos(\widehat F_{adv}^{(l)},\operatorname{sg}(\widehat F_{clean}^{(l)}))\right)
+\rho_B\lVert G(\widehat F_{adv}^{(l)})-\operatorname{sg}(G(\widehat F_{clean}^{(l)}))\rVert_1.
$$

其中：

$$
G(F)=F^\top F/(N_Id).
$$

### 6.2 为什么要同 timestep

MMDiT feature 依赖 timestep $t$。若比较 $F_{clean}(t_1)$ 与 $F_{adv}(t_2)$，则差异包含 timestep 变化，而不只来自扰动。V2.1 代码在同一个 PGD step 中使用同一个 `timestep` 计算 clean 与 adv feature，从而使：

$$
\Delta F=F_{adv}(t)-F_{clean}(t)
$$

更接近扰动导致的表示偏移。

---

## 7. Mode C：Shared-noise Flow Trajectory Divergence

### 7.1 数学定义

采样同一个噪声 $\epsilon_z$：

$$
z_t^{clean}=(1-\sigma_t)z_{clean}+\sigma_t\epsilon_z,
$$

$$
z_t^{adv}=(1-\sigma_t)z_{adv}+\sigma_t\epsilon_z.
$$

模型速度预测：

$$
v_{clean}^t=v_\theta(z_t^{clean},t,c),
\quad
v_{adv}^t=v_\theta(z_t^{adv},t,c).
$$

目标：

$$
L_C=1-\cos(v_{adv}^t,\operatorname{sg}(v_{clean}^t)).
$$

### 7.2 轨迹偏移推导

Euler 近似下：

$$
z_{t-\Delta t}^{adv}-z_{t-\Delta t}^{clean}
=
(z_t^{adv}-z_t^{clean})+
\Delta t(v_{adv}^t-v_{clean}^t).
$$

设：

$$
\lVert v_{adv}^t-v_{clean}^t\rVert_2\ge \gamma_t.
$$

若多个 timestep 上该差异累积，则终点 latent 偏移满足粗略下界：

$$
\lVert \Delta z_0\rVert_2
\gtrsim
\left\lVert\sum_t \Delta t(v_{adv}^t-v_{clean}^t)\right\rVert_2.
$$

该式不是充分保证，因为不同 timestep 的方向可能抵消；但它说明 velocity divergence 是编辑结果偏移的合理机制指标。

---

## 8. Mode D：Modality Balance Disruption

### 8.1 数学定义

定义 image/text stream 能量：

$$
E_I^{(l)}=\frac{1}{N_Id}\lVert F_I^{(l)}\rVert_F^2,
\quad
E_T^{(l)}=\frac{1}{N_Td}\lVert F_T^{(l)}\rVert_F^2.
$$

模态能量比：

$$
r^{(l)}=\log\frac{E_I^{(l)}+\eta}{E_T^{(l)}+\eta}.
$$

当前 Mode D：

$$
L_D=\frac{1}{|\mathcal B|}\sum_l
(r_{adv}^{(l)}-\operatorname{sg}(r_{clean}^{(l)}))^2
+\rho_D(1-\operatorname{CKA}(C_I^{(l)},C_T^{(l)})).
$$

其中 $C_I$、$C_T$ 是 image/text stream 的 channel covariance proxy。当前实现用 covariance cosine 近似 CKA：

$$
\operatorname{CKA}(C_I,C_T)=
\frac{\langle C_I,C_T\rangle_F}{\lVert C_I\rVert_F\lVert C_T\rVert_F+\eta}.
$$

### 8.2 可证明性边界

该 loss 可以证明会推动 image/text stream 的能量比例偏离 clean reference，并降低二者 covariance alignment。但它不能单独证明最终编辑失败；最终仍需 paired SDEdit 与外部指标验证。

---

## 9. 指标定义

### 9.1 约束指标

$$
L_\infty=\lVert x_{adv}-x\rVert_\infty.
$$

代码同时报告 $[0,1]$ 空间与 $[-1,1]$ 空间：

$$
L_\infty^{[-1,1]}=2L_\infty^{[0,1]}.
$$

### 9.2 外部效果指标

paired SDEdit 输出为：

$$
y_{clean}=\operatorname{Edit}(x,c,\sigma,s),
\quad
y_{adv}=\operatorname{Edit}(x_{adv},c,\sigma,s).
$$

当前 metrics 支持：

$$
\operatorname{RMSE}(y_{clean},y_{adv}),
\quad
\operatorname{PSNR}(y_{clean},y_{adv}),
\quad
\operatorname{SSIM}(y_{clean},y_{adv}),
$$

以及可选：

$$
\operatorname{LPIPS}(y_{clean},y_{adv}),
$$

$$
\Delta\operatorname{CLIPScore}
=\operatorname{CLIP}(c,y_{adv})-\operatorname{CLIP}(c,y_{clean}),
$$

$$
\Delta\operatorname{SrcSim}
=\operatorname{CLIP}(x,y_{adv})-\operatorname{CLIP}(x,y_{clean}).
$$

### 9.3 内部机制指标

当前 `.npz` 自动导出：

| Mode | 机制指标 |
|---|---|
| A | `attn_entropy`, `attn_injection_l2` |
| B | `feature_cos`, `feature_gram_l1` |
| C / O_fair | `velocity_cos`, `velocity_div` |
| D | `modality_ratio_dev`, `cross_modal_cka` |
| All debug | `grad_textual_l2`, `grad_mmdit_l2`, `grad_total_l2` |

### 9.4 成功率

聚合脚本采用阈值型成功率：

$$
\operatorname{Success}=\mathbf 1[\
L_\infty\le \tau_\infty
\land
\operatorname{LPIPS}(x,x_{adv})\le\tau_{imp}
\land
\operatorname{LPIPS}(y_{clean},y_{adv})\ge\tau_{edit}
\land
\Delta\operatorname{CLIPScore}\le-\tau_{clip}
].
$$

若某些 optional 指标未计算，则该项不作为失败条件；正式论文实验应安装 LPIPS / CLIP 并启用对应计算。

---

## 10. 伪代码

```text
Input: image x, target y, prompt c, budget epsilon, mode M
Output: protected image x_adv

Initialize x_adv = x
Encode prompt c -> prompt_embeds
for k = 1 ... K:
    z_adv = VAE.encode(x_adv)
    textual = textual_loss(z_adv, y)
    sample timestep t

    if M == A:
        clean_ref = MMDiT(clean, t, capture injection, detach)
        adv_ref   = MMDiT(adv,   t, capture injection, keep graph)
        mmdit = injection_divergence(adv_ref, clean_ref)
    if M == B:
        clean_feat = MMDiT(clean, t, capture image feature, detach)
        adv_feat   = MMDiT(adv,   t, capture image feature, keep graph)
        mmdit = feature_divergence(adv_feat, clean_feat)
    if M == C:
        noise = randn_like(z_adv)
        zt_clean = (1-sigma) z_clean + sigma noise
        zt_adv   = (1-sigma) z_adv   + sigma noise
        mmdit = 1 - cos(v(zt_adv), v(zt_clean))
    if M == D:
        clean_stream = MMDiT(clean, t, capture img/txt, detach)
        adv_stream   = MMDiT(adv,   t, capture img/txt, keep graph)
        mmdit = modality_balance_divergence(adv_stream, clean_stream)

    loss = textual_weight * textual + mmdit_weight * mmdit
    x_adv = project_Linf(x_adv + alpha * sign(grad_x(loss)))
return x_adv
```

---

## 11. 参考理论基础

- PGD / robust optimization: https://arxiv.org/abs/1706.06083
- SD3 / MMDiT / Rectified Flow Transformer: https://arxiv.org/abs/2403.03206
- Flow Matching: https://arxiv.org/abs/2210.02747
- SDEdit: https://arxiv.org/abs/2108.01073
- Mist: https://arxiv.org/abs/2305.12683
- AdvDM: https://arxiv.org/abs/2302.04578
- PhotoGuard: https://proceedings.mlr.press/v202/salman23a.html
- LPIPS: https://arxiv.org/abs/1801.03924
- FID / TTUR: https://arxiv.org/abs/1706.08500
- CLIPScore: https://arxiv.org/abs/2104.08718
- CKA: https://proceedings.mlr.press/v97/kornblith19a.html

---

## 12. 仍需实验证明的命题

V2.1 给出的是可导、结构对齐、可诊断的攻击建模。以下命题不能仅由数学定义推出，必须由实验验证：

1. $L_A/L_B/L_C/L_D$ 增大是否稳定导致 paired SDEdit 结果偏移；
2. 哪个 mode 在同一 $\epsilon$ 预算下最优；
3. O_fair 与 C 的差异是否显著；
4. 随机 $L_\infty$ baseline 是否显著弱于结构攻击；
5. 迁移到其他 SD3 / SD3.5 pipeline 是否仍有效；
6. JPEG / resize / crop 后扰动是否仍有效。

因此论文主结果必须同时报告外部编辑破坏指标和内部机制指标。
