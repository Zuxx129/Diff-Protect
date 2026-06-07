# SD3 对抗扰动设计 v2

> 适用项目：`Zuxx129/Diff-Protect`  
> 重点文件：`SD3对抗扰动设计.md`、`code/attacks_SD3.py`、`code/diff_mist_SD3.py`、`configs/attack/base_sd3.yaml`  
> 目标：将原始 SD3 对抗扰动设想整理为理论语义统一、可反传、可验证、可消融的 SD3 / SD3.5 MMDiT 结构感知对抗扰动框架。

---

## 0. 核心结论

本项目的研究问题成立：Stable Diffusion 3 / 3.5 将传统 `UNet + epsilon-prediction` 路线替换为 `MMDiT + rectified flow / flow matching` 路线，文本 token 与图像 latent token 在 Joint Attention 中双向交互。因此，仅沿用 SD v1.4 时代面向 UNet denoising error 的 semantic loss，不足以解释和攻击 SD3 的关键结构路径。

但 v1 设计中有三类必须修正的问题：

1. **优化方向不统一**：`g_mode='+'` 只是对当前标量 loss 做梯度上升，`g_mode='-'` 只是梯度下降；它们本身不等价于“攻击方向”或“防御方向”。所有 loss 应先被定义成“越大越攻击成功”，再统一使用梯度上升。
2. **部分目标缺少可导性约束**：Mode A/B/D 如果继续使用 `detach()` 或 `torch.no_grad()` 保存 attention / feature，再把这些张量用于 loss，则对输入扰动的梯度为零。
3. **部分结构目标容易退化为 trivial solution**：例如单纯最大化 feature norm、单纯增大 attention entropy、单纯增大 image-stream variance，都可能只制造内部数值异常，而不一定破坏编辑语义。因此 v2 需要引入归一化、clean reference、paired trajectory 和机制诊断。

v2 的总体原则是：

$$
\max_{\lVert \delta \rVert_\infty \le \epsilon}
L_{\mathrm{attack}}(x+\delta,c;\theta),
$$

其中 $x$ 是原图，$\delta$ 是不可感知扰动，$c$ 是文本条件，$\theta$ 是冻结的 SD3 / SD3.5 pipeline 参数。

---

## 1. 威胁模型与研究定位

### 1.1 任务定义

给定一张受保护图像 $x \in [-1,1]^{3\times H\times W}$，目标是在 $L_\infty$ 约束下生成扰动 $\delta$：

$$
\lVert \delta \rVert_\infty \le \epsilon,
\quad
x_{adv}=\operatorname{clip}(x+\delta,-1,1),
$$

使得 $x_{adv}$ 对人眼保持接近 $x$，但在 SD3 / SD3.5 的图生图、SDEdit 或编辑流程中产生明显偏移：

$$
\operatorname{Edit}_{\theta}(x_{adv},c,\sigma,s)
\not\approx
\operatorname{Edit}_{\theta}(x,c,\sigma,s),
$$

其中 $\sigma$ 是 SDEdit 风格加噪强度，$s$ 是随机种子。正式评估必须使用 paired setting：同一输入、同一 prompt、同一 $\sigma$、同一 seed。

### 1.2 授权假设

本项目默认场景为授权内容安全研究：图像持有者为防止未经授权的扩散式编辑、模仿或二次生成，对自己的图像添加不可感知扰动。当前主要是白盒 SD3.5-medium 研究，transferability 与黑盒评估属于扩展实验。

### 1.3 攻击者能力

白盒阶段默认攻击者可访问：

- VAE encoder / decoder；
- SD3Transformer / MMDiT transformer；
- scheduler；
- prompt encoder 输出；
- intermediate attention / feature hooks。

攻击者不可修改模型权重，只优化输入扰动 $\delta$。

### 1.4 防护目标与不可声称内容

可以声称：

- 在给定 $\epsilon$ 预算和白盒模型下，扰动能破坏 SD3 编辑稳定性；
- 某些 MMDiT 内部机制指标与外部编辑破坏指标存在统计相关；
- SD3-specific loss 相比旧 UNet-era baseline 在特定协议下更有效。

不应过早声称：

- 理论上保证任何 SD3 / SD3.5 模型均失效；
- 任意 prompt / 任意图像均有效；
- A/B/C/D 任一模式已经被证明优于 baseline，除非 `O_fair`、随机噪声、SD1.4 baseline、paired SDEdit 和统计显著性均完成。

---

## 2. SD3 / MMDiT 的结构基础

### 2.1 与 SD v1.4 的差异

SD v1.4 典型结构是：

$$
\text{VAE latent} \rightarrow \text{UNet denoiser} \rightarrow \epsilon\text{-prediction}.
$$

SD3 / SD3.5 的关键变化是：

$$
\text{VAE latent tokens} + \text{text tokens}
\rightarrow \text{MMDiT Joint Transformer}
\rightarrow \text{velocity / flow prediction}.
$$

这带来三点研究后果：

1. **Cross-Attention 不应被简单等同于 Joint Attention**。SD3 的文本-图像交互是在联合 token 序列中通过 attention 发生，且具有双向信息流。
2. **epsilon-prediction 不应与 flow / velocity prediction 混用**。Mode C 的理论目标应围绕 velocity field，而不是旧 UNet 的 noise residual。
3. **中间表示更适合作为攻击面**。MMDiT 的 image stream、text stream、joint attention 和 block-wise representation 都可以成为结构感知扰动的路径。

### 2.2 Flow Matching / Rectified Flow 视角

Flow matching 可以抽象为学习一个条件向量场：

$$
v_\theta(z_t,t,c) \approx u_t(z_t \mid z_0,z_1,c),
$$

其中 $z_t$ 是从数据 latent 到噪声 latent 的路径点，$v_\theta$ 是模型预测的速度场。采样或编辑过程可近似理解为沿向量场做数值积分：

$$
z_{t-\Delta t}=z_t+\Delta t\cdot v_\theta(z_t,t,c).
$$

因此，如果扰动使 clean 与 adv 的速度方向系统性分离：

$$
\cos\left(v_\theta(z_t^{adv},t,c),v_\theta(z_t^{clean},t,c)\right) \downarrow,
$$

则多步积分后的编辑结果有可能累积偏移。这是 Mode C 的核心数学基础。

---

## 3. 统一优化形式

### 3.1 基础 PGD 问题

设 $x_0$ 为原图，$x_k=x_0+\delta_k$。采用 $L_\infty$ PGD：

$$
x_{k+1}
=
\Pi_{\lVert x-x_0\rVert_\infty\le \epsilon,\ x\in[-1,1]}
\left(
  x_k+\alpha\cdot \operatorname{sign}\left(\nabla_x L_{attack}(x_k)\right)
\right).
$$

注意：v2 推荐不再用 `g_mode` 作为理论概念。建议配置中改为：

```yaml
attack:
  opt_direction: maximize
  objective_convention: larger_is_more_attack
```

如果为了兼容旧代码保留 `g_mode`，则：

```text
g_mode='+'  等价于 maximize 当前实现的 scalar loss
g_mode='-'  等价于 minimize 当前实现的 scalar loss
```

`g_mode` 不是研究贡献，也不是方法类别。

### 3.2 Joint Objective

v2 推荐统一写为：

$$
L_{attack}
=
\lambda_T L_T
+
\lambda_M L_M,
$$

其中：

- $L_T$ 是 VAE latent textual / target loss；
- $L_M$ 是 SD3 MMDiT-specific structural loss；
- $\lambda_T,\lambda_M\ge 0$ 是权重。

对 A/B/C/D 单模式实验：

$$
L_M \in \{L_A,L_B,L_C,L_D\}.
$$

对后续组合实验：

$$
L_M = \lambda_A L_A+\lambda_B L_B+\lambda_C L_C+\lambda_D L_D.
$$

但论文主线不建议一开始就组合四项，因为组合 loss 难解释、消融成本高、容易被审稿人认为是工程堆叠。

---

## 4. Textual Loss 的 v2 定义

### 4.1 Mist-compatible target-pull

如果沿用 Mist 思路，textual loss 的语义应是：让 $x_{adv}$ 的 latent 接近一个错误 target image $y$ 的 latent。

设 VAE encoder 为 $\mathcal E$：

$$
z_x=\mathcal E(x),
\quad
z_{adv}=\mathcal E(x+\delta),
\quad
z_y=\mathcal E(y).
$$

target-pull 的自然形式是最小化距离：

$$
\min_\delta \lVert z_{adv}-z_y\rVert_2^2.
$$

为了统一 larger-is-more-attack 的梯度上升语义，v2 将其写成：

$$
L_T^{toward}
=
-\lVert z_{adv}-z_y\rVert_2^2.
$$

这样最大化 $L_T^{toward}$ 等价于把 adv latent 拉向 target latent。

### 4.2 target-repulsion 作为独立变体

如果实验上想让 latent 远离 target 或远离 source，应明确命名为：

```yaml
textual_objective: away_from_target
```

对应：

$$
L_T^{away}
=
\lVert z_{adv}-z_y\rVert_2^2.
$$

不要把 $L_T^{away}$ 继续称为 Mist textual loss。它是另一个目标。

### 4.3 推荐配置

```yaml
attack:
  textual_objective: toward_target
  textual_weight: 1.0
```

论文主结果建议使用 `toward_target`，因为它与 Mist target-pull 思路更一致。

---

## 5. MMDiT Loss v2

### 5.1 Mode A：Joint Attention Injection Disruption

#### 目标

破坏 text token 对 image token 的语义注入，而不是简单让 attention 变“大”或变“小”。

设某个 MMDiT block 的 image-query 到 text-key attention 为：

$$
A_{I\to T}^{(l)}(x,c) \in \mathbb R^{B\times H\times N_I\times N_T}.
$$

文本 value 为：

$$
V_T^{(l)}(x,c)\in\mathbb R^{B\times H\times N_T\times d_h}.
$$

attention 注入到 image token 的文本分量可近似为：

$$
O_{I\leftarrow T}^{(l)}(x,c)
=
A_{I\to T}^{(l)}(x,c) V_T^{(l)}(x,c).
$$

相比单纯最大化 entropy，v2 推荐直接最大化 clean 与 adv 的文本注入差异：

$$
L_A
=
\frac{1}{|\mathcal B|}
\sum_{l\in\mathcal B}
\frac{
\left\lVert
O_{I\leftarrow T}^{(l)}(x_{adv},c)
-
\operatorname{sg}\left(O_{I\leftarrow T}^{(l)}(x,c)\right)
\right\rVert_F^2
}{
\operatorname{sg}\left(\left\lVert O_{I\leftarrow T}^{(l)}(x,c)\right\rVert_F^2\right)+\eta
}.
$$

其中 $\operatorname{sg}(\cdot)$ 表示 stop-gradient，仅 clean reference detach，adv 分支保持可导。

#### 机制诊断

同时记录但不一定作为主优化项：

$$
H_A^{(l)}=-\sum_j A_{ij}^{(l)}\log A_{ij}^{(l)},
$$

以及：

$$
D_{JS}\left(A_{I\to T}^{adv}\Vert A_{I\to T}^{clean}\right).
$$

#### 可证明性边界

attention 输出满足：

$$
\Delta O
=(A_{adv}-A_{clean})V_T + A_{adv}(V_T^{adv}-V_T^{clean}).
$$

因此扰动 attention 分布或 value 表示都会改变文本注入项。这能证明该目标直接作用在 Joint Attention 的文本注入通道上，但不能单独保证最终图像语义必然偏移。最终结论必须通过 paired SDEdit 指标验证。

#### 工程建议

- 不保存完整所有层 `[B,H,N,N]` attention map。
- 仅选择 `capture_blocks=[0,6,12,17]` 或类似稀疏层。
- 在 attention processor 内在线累计 $L_A$ 标量。
- `clean` 分支 detach，`adv` 分支保留图。

### 5.2 Mode B：Image-stream Representation Divergence

#### 目标

让 adv image stream 表示偏离 clean image stream 表示，而不是单纯增大 feature norm。

设第 $l$ 层 image stream 表示为：

$$
F_I^{(l)}(x,c)\in \mathbb R^{B\times N_I\times d}.
$$

先做 token-wise 或 feature-wise normalization：

$$
\widehat F=\frac{F-\mu(F)}{\sigma(F)+\eta}.
$$

v2 推荐：

$$
L_B
=
\frac{1}{|\mathcal B|}\sum_{l\in\mathcal B}
\left[
1-\frac{\langle \widehat F_{adv}^{(l)},\operatorname{sg}(\widehat F_{clean}^{(l)})\rangle}
{\lVert \widehat F_{adv}^{(l)}\rVert_F\lVert \operatorname{sg}(\widehat F_{clean}^{(l)})\rVert_F+\eta}
\right]
+
\rho_B
\left\lVert
G(\widehat F_{adv}^{(l)})-
\operatorname{sg}(G(\widehat F_{clean}^{(l)}))
\right\rVert_1.
$$

其中 $G(F)=FF^\top/N_I$ 或 $F^\top F/N_I$，取决于要比较 token-style 还是 channel-style。

#### 为什么需要归一化

如果直接最大化 $\lVert F_{adv}-F_{clean}\rVert$，可能得到 trivial solution：只放大 feature norm，而不是改变语义方向。归一化 cosine / CKA / Gram 差异能降低这种风险。

#### 机制诊断

- block-wise cosine similarity；
- linear CKA；
- Gram distance；
- feature norm ratio。

### 5.3 Mode C：Flow Trajectory Divergence

#### 目标

破坏 SD3 flow matching 轨迹，使 clean 与 adv 在相同 prompt、相同 timestep、相同噪声条件下预测不同速度方向。

构造 paired latent：

$$
z_t^{clean}=(1-\sigma_t)z_x+\sigma_t\epsilon,
$$

$$
z_t^{adv}=(1-\sigma_t)z_{adv}+\sigma_t\epsilon,
$$

其中 $\epsilon$ 必须共享。

MMDiT 预测：

$$
v_{clean}^t=v_\theta(z_t^{clean},t,c),
\quad
v_{adv}^t=v_\theta(z_t^{adv},t,c).
$$

v2 推荐：

$$
L_C
=
\mathbb E_{t\sim \mathcal T}
\left[
1-\frac{\langle v_{adv}^t,\operatorname{sg}(v_{clean}^t)\rangle}
{\lVert v_{adv}^t\rVert_2\lVert\operatorname{sg}(v_{clean}^t)\rVert_2+\eta}
\right].
$$

#### 可证明性基础

Euler 形式下：

$$
z_{t-\Delta t}^{adv}-z_{t-\Delta t}^{clean}
=
(z_t^{adv}-z_t^{clean})+
\Delta t\left(v_{adv}^t-v_{clean}^t\right).
$$

若在多个 $t$ 上持续增大 velocity direction divergence，则终点 latent 偏移有累积机制。该推导支持 Mode C 的理论合理性，但不构成最终图像失真的充分必要条件。最终仍需 paired SDEdit 输出验证。

#### 工程建议

- 每个 PGD step 采样固定数量 timestep，例如 `num_t=2` 或 `num_t=4`。
- clean 与 adv 使用同一噪声 $\epsilon$。
- 记录 $1-\cos(v_{adv}^t,v_{clean}^t)$ 的 step 曲线和 sigma 曲线。

### 5.4 Mode D：Modality Balance Disruption

#### 目标

利用 MMDiT 双流结构，使 image stream 与 text stream 的能量比、相关性或表示耦合发生异常。v2 不建议只最大化 image variance，因为这容易退化为特征能量放大。

设：

$$
E_I^{(l)}=\frac{1}{N_I d}\lVert F_I^{(l)}\rVert_F^2,
\quad
E_T^{(l)}=\frac{1}{N_T d}\lVert F_T^{(l)}\rVert_F^2.
$$

定义 clean modality balance：

$$
r_{clean}^{(l)}=
\log\frac{E_I^{(l)}(x,c)+\eta}{E_T^{(l)}(x,c)+\eta}.
$$

adv balance：

$$
r_{adv}^{(l)}=
\log\frac{E_I^{(l)}(x_{adv},c)+\eta}{E_T^{(l)}(x_{adv},c)+\eta}.
$$

v2 推荐：

$$
L_D
=
\frac{1}{|\mathcal B|}\sum_{l\in\mathcal B}
\left(r_{adv}^{(l)}-\operatorname{sg}(r_{clean}^{(l)})\right)^2
+
\rho_D
\left(1-\operatorname{CKA}(F_I^{(l)}(x_{adv},c),F_T^{(l)}(x_{adv},c))\right).
$$

#### 解释

第一项使 adv 的 image/text 能量平衡偏离 clean；第二项降低跨模态表示耦合。与 v1 中单纯 `-Var + corr` 相比，v2 更清楚地说明了“模态失衡”是什么，并通过 clean reference 避免只追求绝对能量放大。

---

## 6. Mode O 与公平 baseline

### 6.1 当前 O 模式的问题

如果 O 模式只是：

$$
L_O=\lVert v_{pred}\rVert_2,
$$

它既不是 Mist / AdvDM 的 denoising error，也不是 flow matching 的标准误差目标。因此它只能作为 `O_repo`，即仓库原始 baseline，而不是公平理论 baseline。

### 6.2 v2 baseline 分层

| 名称 | 含义 | 用途 |
|---|---|---|
| `SD14_Mist` | SD v1.4 原始 Mist 风格联合 loss | 与旧 UNet-era 方法对比 |
| `SD14_SDS` | SD v1.4 SDS / AdvDM 路线 | 旧路线补充对照 |
| `SD3_O_repo` | 当前仓库 O 模式 | 复现工程现状 |
| `SD3_O_fair` | flow-matching 语境下修正后的非结构 baseline | 公平比较 |
| `Random_Linf` | 同 $\epsilon$ 随机噪声 | 排除“任意噪声均有效” |

### 6.3 O_fair 建议

O_fair 不一定必须完全复刻 SD3 训练目标，但应至少满足：

1. 与 flow / velocity prediction 同语境；
2. 与 A/B/C/D 使用相同 $\epsilon$、steps、prompt、seed；
3. 没有明显弱化 baseline 的工程缺陷。

可选形式：

$$
L_{O\_fair}
=
1-\frac{\langle v_\theta(z_t^{adv},t,c),\operatorname{sg}(v_\theta(z_t^{clean},t,c))\rangle}
{\lVert v_\theta(z_t^{adv},t,c)\rVert_2\lVert\operatorname{sg}(v_\theta(z_t^{clean},t,c))\rVert_2+\eta}.
$$

但这会与 Mode C 接近。因此也可以把 O_fair 定义为“无结构 hook 的 velocity-prediction baseline”，而 Mode C 是“多 timestep paired trajectory + diagnostics + trajectory-oriented schedule”的结构版本。二者关系必须在文档中说明清楚，避免重复。

---

## 7. 可导性、公理化检查与可证明性边界

### 7.1 必要条件

每个进入主实验的 loss 必须满足：

$$
\left\lVert \nabla_x L_M(x) \right\rVert_2 > 0.
$$

实际代码中，应在第一个 PGD step 打印：

$$
\lVert \nabla_x L_T\rVert_2,
\quad
\lVert \nabla_x L_M\rVert_2,
\quad
\lVert \nabla_x L_{attack}\rVert_2.
$$

如果某模式的 $\nabla_x L_M=0$，该模式不能进入主表。

### 7.2 符号一致性

每个 mode 必须满足：

$$
\frac{d}{dk}\operatorname{Diag}_M(x_k) > 0,
$$

其中 $\operatorname{Diag}_M$ 是该 mode 的机制诊断量。

| mode | 诊断量 |
|---|---|
| A | attention injection difference / JS divergence 上升 |
| B | feature divergence 上升，cosine / CKA 下降 |
| C | $1-\cos(v_{adv},v_{clean})$ 上升 |
| D | modality balance deviation 上升，cross-modal CKA 下降 |

### 7.3 不可证明的部分

v2 不能也不应声称：

$$
L_M \uparrow \Rightarrow \text{编辑必然失败}.
$$

更合理的表述是：

$$
L_M \uparrow
\Rightarrow
\text{对应内部机制发生可测偏移};
$$

再通过实验验证：

$$
\text{内部机制偏移}
\Longleftrightarrow
\text{paired edit disruption 指标显著变化}.
$$

这就是本项目的理论-实验闭环。

---

## 8. 与已有研究的交叉验证基础

### 8.1 PGD / 一阶对抗优化

PGD 是标准一阶对抗优化框架。本项目使用 $L_\infty$ PGD 合理，但必须严格报告 $\epsilon$、$\alpha$、steps、投影空间和像素空间映射。

参考：Madry et al., “Towards Deep Learning Models Resistant to Adversarial Attacks”, 2017.  
https://arxiv.org/abs/1706.06083

### 8.2 Mist / AdvDM / PhotoGuard

Mist 与 AdvDM 提供了“对扩散模型生成/模仿过程添加不可感知扰动”的路线。PhotoGuard 提供了“对图像编辑进行 immunization”的安全场景和评估先例。

参考：

- Mist: https://arxiv.org/abs/2305.12683
- AdvDM: https://arxiv.org/abs/2302.04578
- PhotoGuard: https://proceedings.mlr.press/v202/salman23a.html

### 8.3 SD3 / MMDiT

SD3 原论文提出 rectified flow transformer，并强调图像 token 与文本 token 之间的双向信息流。这是 Mode A/D 的结构基础。

参考：Patrick Esser et al., “Scaling Rectified Flow Transformers for High-Resolution Image Synthesis”, 2024.  
https://arxiv.org/abs/2403.03206

### 8.4 Flow Matching

Flow Matching 将生成建模表述为对向量场的回归。Mode C 的 velocity trajectory divergence 直接建立在这一点上。

参考：Lipman et al., “Flow Matching for Generative Modeling”, 2022.  
https://arxiv.org/abs/2210.02747

### 8.5 SDEdit

SDEdit 的核心是对输入加噪，再通过生成先验去噪，以平衡 fidelity 与 realism。本项目的 SD3 评估应称为 SDEdit-style / FlowMatch-style proxy，而非严格复刻原始 SDEdit。

参考：Meng et al., “SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations”, 2021.  
https://arxiv.org/abs/2108.01073

### 8.6 指标基础

- LPIPS：学习型感知相似度。https://arxiv.org/abs/1801.03924
- FID：集合级生成分布距离，不适合单图。https://arxiv.org/abs/1706.08500
- CLIPScore：基于 CLIP 的 reference-free text-image compatibility。https://arxiv.org/abs/2104.08718
- CKA：神经网络表示相似性比较。https://proceedings.mlr.press/v97/kornblith19a.html

---

## 9. v2 推荐代码接口

### 9.1 配置

```yaml
attack:
  epsilon: 16
  alpha: 1
  steps: 50
  input_size: 512
  mode: C
  opt_direction: maximize
  objective_convention: larger_is_more_attack
  textual_objective: toward_target
  textual_weight: 1.0
  mmdit_weight: 1.0
  seed: 0

  mmdit:
    capture_blocks: [0, 6, 12, 17]
    detach_clean: true
    detach_adv: false
    online_attention_scalar: true
    normalize_features: true
    num_timesteps_for_C: 2

  eval:
    paired_sdedit: true
    noise_levels: [0.1, 0.3, 0.5]
    save_clean_edit: true
    save_adv_edit: true
    save_metrics_json: true
```

### 9.2 Loss 返回格式

```python
def compute_loss(...):
    return {
        "loss_opt": loss_opt,
        "components": {
            "textual": textual_loss.detach(),
            "mmdit": mmdit_loss.detach(),
            "total": loss_opt.detach(),
        },
        "diagnostics": {
            "attn_injection_diff": ...,
            "attn_entropy": ...,
            "feature_cos": ...,
            "feature_cka": ...,
            "velocity_cos": ...,
            "velocity_div": ...,
            "modality_ratio_dev": ...,
            "cross_modal_cka": ...,
        }
    }
```

### 9.3 必须保留的 debug 输出

```text
step, loss_total, loss_textual, loss_mmdit,
grad_textual_l2, grad_mmdit_l2, grad_total_l2,
max_delta, attn_entropy, feature_cos, velocity_div, modality_ratio_dev
```

---

## 10. v2 主张写法

推荐贡献表述：

> 本研究提出一种面向 SD3 / SD3.5 MMDiT 的结构感知图像扰动框架。与传统 UNet-era 扩散对抗保护不同，该框架显式建模 Joint Attention 跨模态注入、image-stream 表示偏移、flow-matching velocity trajectory divergence 与 image/text 双流模态平衡，并将所有结构目标统一为 larger-is-more-attack objective，在 $L_\infty$ 约束下通过 PGD 优化。实验上，通过 paired SDEdit、机制诊断曲线与多 baseline 消融，验证内部结构偏移是否能转化为实际编辑破坏。

不推荐贡献表述：

> 我们实现了四个 loss，并发现比 baseline 好。

---

## 11. 最小验收标准

v2 方法进入主实验前，必须满足：

- [ ] A/B/C/D 的 $\nabla_x L_M$ 均非零；
- [ ] A/B/C/D 的机制诊断量沿 PGD step 朝设计方向变化；
- [ ] `textual_objective` 语义明确；
- [ ] `g_mode` 不再作为理论概念，而是优化方向兼容字段；
- [ ] Mode O 拆成 `O_repo` 与 `O_fair`；
- [ ] SDEdit 输出为 clean/adv paired；
- [ ] 指标脚本可输出 per-image JSON 与 aggregate CSV；
- [ ] 主表不使用未修复的 A/B/D 原始实现。
