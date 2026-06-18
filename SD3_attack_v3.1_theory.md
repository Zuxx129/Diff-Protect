# SD3 攻击 v3.1 理论文档：面向 SD3/SD3.5 图像编辑的流匹配防护

> 版本定位：v3.1 是对 v2.1 的理论重构。O/A/B/C/D 均沿用 v2.1 的 textual target-pull + MMDiT 机制项 joint objective；新增 E 模式，用于从 v2.1 O 中单独拆分 semantic velocity-norm loss 并只优化该 semantic loss。  
> 主线方法：SD3 流匹配防护（SD3 Flow-Matching Protection, FMP）。  
> 研究场景：面向 SD3/SD3.5 图像编辑模型的对抗防护，即在图像中加入不可感知扰动或水印，使未经授权的 SD3/SD3.5 img2img/SDEdit 编辑失效或显著偏离。
> 命名口径：除非特别说明，后续所说 O/A/B/C/D 均指 `fix/sd3-objective-sanity` 分支中的 v2.1 joint objective。`O` 统一指 v2.1 `mode=O`；v2.1 代码内部的 `O_repo` 只作为历史别名理解，本文和后续实验不再使用 `O_repo` 作为方法名。`E` 是 v3.1 新增 semantic-only 模式，只优化从 v2.1 O 中拆出的 \(L_{\mathrm{sem}}=\|v_\theta(z_p,t,c)\|_2\)，不加入 textual target-pull。
> origin/main 口径：`origin/main` 的 SD3 `mode=O` 不再简称为 O；如需讨论，必须显式写作“origin/main 的 O”或 `textual_semantic_joint`。后续也不再使用 `O_fair` 作为方法名。

---

## 摘要

本文档给出 SD3 Attack v3.1 的理论建模。目标不是攻击分类器，也不是攻击纯文本生成，而是保护一张待发布图像，使攻击者在使用 SD3/SD3.5 图像编辑模型进行恶意编辑时得到不稳定、失真或偏离预期的编辑结果。形式上，给定原始图像 \(x\)，防护者寻找一个满足 \(L_\infty\) 预算的扰动 \(\delta\)，发布受保护图像 \(x_p=\operatorname{clip}(x+\delta)\)。攻击者随后使用 SD3/SD3.5 的 img2img 或 SDEdit 风格编辑链路生成结果。防护目标是让 \(\operatorname{Edit}_\theta(x_p)\) 与 \(\operatorname{Edit}_\theta(x)\) 在同一 prompt、同一 seed、同一编辑强度下显著不同，同时保持 \(x_p\) 对人眼接近 \(x\)。

v2.1 曾尝试把 MMDiT 内部结构项与 textual target-pull 组成 O/A/B/C/D joint objective：

\[
L_{\mathrm{attack}}=\lambda_T L_T+\lambda_M L_M.
\]

已有实验显示，该建模没有形成显著的模式差异：v2.1 A/B/C/D 与 origin/main 的 O baseline，即本文历史分析中记作 `textual_semantic_joint` 的 baseline，外部 paired SDEdit 指标几乎一致，扰动方向高度相似。核心原因不是绘图错误，而是目标层级和代理目标选择错误：文本项损失的梯度主导了 PGD 更新，MMDiT 结构项在实际优化方向中几乎被淹没；同时 v2.1 A/B/D 的输入状态并不严格位于 SDEdit 编辑轨迹的加噪 latent 分布上，v2.1 B/C 等 clean-adv 差异目标又在 \(\delta\approx 0\) 附近存在天然弱梯度或零梯度问题。

v3.1 因此转向更靠近 SD3 训练目标和编辑链路的代理目标：最大化受保护 latent 在 SDEdit 加噪状态上的流匹配预测误差。该目标简称 FMP：

\[
L_{\mathrm{FMP}}
=
\mathbb E_{\sigma,\xi,c}
\left[
\frac{\left\|v_\theta(z^p_\sigma,t_\sigma,c)-\operatorname{sg}(u_\sigma)\right\|_2^2}
{\operatorname{sg}(\left\|u_\sigma\right\|_2^2)+\eta}
\right].
\]

其中 \(z^p_\sigma=(1-\sigma)z_p+\sigma\xi\)。该目标直接作用于 SD3/SD3.5 去噪 / 流预测误差，而不是间接破坏某个内部 attention 或 feature 指标。O/A/B/C/D 保留为 v2.1 joint baseline；新增 E 作为 semantic-only velocity-norm baseline，用来检验“只优化 O 的 semantic 分量”是否足够，并与 FMP 的 flow-matching prediction error 形成对照。

---

## 1. 问题背景

### 1.1 图像编辑防护任务

传统对抗样本通常关注分类器误判。扩散模型防护任务不同：受害者不是模型输出的一个离散标签，而是一张图像在生成式编辑系统中的可编辑性。攻击者可能下载一张公开图片，并使用图像编辑模型进行换脸、换背景、属性修改、风格迁移、局部重绘或其他恶意编辑。防护者希望在发布图像前加入微小扰动，使这些编辑链路无法稳定地利用原图内容。

因此，本课题的核心对象是编辑映射：

\[
\operatorname{Edit}_\theta(\cdot;\omega),
\]

而不是分类函数。这里 \(\theta\) 表示冻结的 SD3/SD3.5 模型参数，\(\omega\) 表示一次编辑请求中的外部条件，例如 prompt、negative prompt、编辑强度、随机噪声、seed、guidance scale 和 denoising steps。

### 1.2 为什么 SD3/SD3.5 需要重新建模

SD v1.x 系列主要使用 UNet denoiser，并常见 \(\epsilon\)-prediction 或类似噪声预测目标。MIST 等方法中的 semantic loss 可写作最大化噪声预测误差：

\[
\mathbb E_{t,\epsilon}\left[
\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2
\right].
\]

SD3/SD3.5 的核心架构和训练目标发生了变化：

1. denoiser 是 MMDiT / joint transformer，而不是传统 UNet；
2. 图像 latent token 与文本 token 通过 joint attention 交互；
3. 采样目标来自 rectified flow / flow matching，模型预测速度场 \(v_\theta\)，而不是直接预测传统 DDPM 噪声残差；
4. img2img/SDEdit 编辑链路从一个由原图 latent 与噪声混合得到的加噪 latent 状态开始。

因此，直接把 SD v1.x 的 \(\epsilon\)-prediction 语义损失或某个 MMDiT 特征差异目标迁移到 SD3，并不能保证它与 SD3 编辑链路的真实失败机制一致。

---

## 2. 变量与符号定义

本节统一给出后续公式中的变量含义。除特别说明外，所有模型参数在攻击优化期间均冻结。

### 2.1 图像、扰动与约束

| 符号 | 含义 | 作用 |
|---|---|---|
| \(x\) | 原始待保护图像，通常表示为归一化张量 \(x\in[-1,1]^{3\times H\times W}\) | 防护算法的输入，也是人眼应看到的真实图像 |
| \(\delta\) | 对抗扰动或水印，形状与 \(x\) 相同 | 防护者优化的变量 |
| \(x_p\) | 受保护图像，定义为 \(x_p=\operatorname{clip}(x+\delta)\) | 最终发布给外界的图像 |
| \(\epsilon\) | \(L_\infty\) 扰动预算 | 限制每个像素可变化的最大幅度 |
| \(\mathcal B_\infty(x,\epsilon)\) | 以 \(x\) 为中心的 \(L_\infty\) ball | PGD 投影集合 |
| \(\Pi_{\mathcal B}\) | 投影算子 | 保证每一步优化后仍满足扰动约束 |

约束为：

\[
\|\delta\|_\infty\le \epsilon,
\quad
x_p\in[-1,1]^{3\times H\times W}.
\]

若 \(\epsilon\) 以 0-255 像素尺度给出，则映射到 \([-1,1]\) 张量空间通常为：

\[
\epsilon_{\mathrm{tensor}}=\frac{2\epsilon_{\mathrm{pixel}}}{255}.
\]

这个因子 2 来自图像从 \([0,1]\) 到 \([-1,1]\) 的线性变换。

### 2.2 VAE latent 变量

| 符号 | 含义 | 作用 |
|---|---|---|
| \(\mathcal E\) 或 \(E\) | SD3/SD3.5 的 VAE encoder | 将图像映射到 latent 空间 |
| \(\mathcal D\) 或 \(D\) | SD3/SD3.5 的 VAE decoder | 将 latent 解码回图像空间 |
| \(z\) | 原图 latent，\(z=E(x)\) | clean SDEdit 的起点信息 |
| \(z_p\) | 受保护图像 latent，\(z_p=E(x_p)\) | protected SDEdit 的起点信息 |
| \(\xi\) | latent 空间噪声，通常 \(\xi\sim\mathcal N(0,I)\) | SDEdit 加噪状态中的随机成分 |
| \(\sigma\) | SDEdit 噪声强度，\(\sigma\in[0,1]\) 或 scheduler 对应的噪声尺度 | 控制保留原图内容与注入噪声的比例 |
| \(z^p_\sigma\) | 受保护图像的加噪 latent | FMP 直接优化的输入状态 |
| \(z_\sigma\) | 原图的加噪 latent | paired SDEdit 的 clean 参考 |

本文采用与当前工程中 SDEdit 评估一致的混合形式：

\[
z_\sigma=(1-\sigma)z+\sigma\xi,
\]

\[
z^p_\sigma=(1-\sigma)z_p+\sigma\xi.
\]

其中 \(z_\sigma\) 表示 clean 图像在噪声强度 \(\sigma\) 下的编辑初始状态，\(z^p_\sigma\) 表示 protected 图像在同一噪声 \(\xi\) 和同一 \(\sigma\) 下的编辑初始状态。成对评估必须使用相同的 \(\xi\)，否则输出差异会混入 seed/noise 差异，而不只是扰动效果。

### 2.3 文本条件与 SD3 速度场

| 符号 | 含义 | 作用 |
|---|---|---|
| \(c\) | 文本条件，包括 prompt embedding、pooled embedding 等 | 控制编辑语义 |
| \(\theta\) | 冻结的 SD3/SD3.5 参数 | 被攻击但不被训练的模型 |
| \(t_\sigma\) | 与 \(\sigma\) 对应的 scheduler timestep | 标识 denoising/flow trajectory 的位置 |
| \(v_\theta(z_t,t,c)\) | SD3 MMDiT transformer 预测的速度场 | 采样器用它更新 latent |
| \(u_\sigma\) | rectified-flow / flow-matching 的目标速度 | FMP 中被故意预测错误的目标 |
| \(\operatorname{sg}(\cdot)\) | stop-gradient 算子 | 将参考目标视为常量，防止优化器改变目标本身 |
| \(\eta\) | 数值稳定常数，\(\eta>0\) | 防止除零并规范化 loss 尺度 |

SD3 的 denoising step 可抽象为一个由速度场驱动的离散流：

\[
z_{k+1}=z_k+h_k v_\theta(z_k,t_k,c),
\]

其中 \(h_k\) 是第 \(k\) 个离散 step 的步长。不同 scheduler 可能使用不同符号和步长约定。因此本文所有关于 \(u_\sigma\) 的实现细节必须以 diffusers 中实际 SD3 scheduler 和 transformer 输出约定为准，理论文档不强行写死符号方向。

---

## 3. 威胁模型

### 3.1 防护者能力

防护者拥有原始图像 \(x\)，并能在发布前生成受保护图像：

\[
x_p=\operatorname{clip}(x+\delta).
\]

防护者可以白盒访问 SD3/SD3.5 模型组件，包括：

1. VAE encoder \(E\)；
2. VAE decoder \(D\)；
3. text encoder 与 prompt embedding；
4. MMDiT transformer \(v_\theta\)；
5. scheduler 的 timestep 和 sigma 映射；
6. 必要时的中间 attention、feature、modality diagnostics。

防护者不修改模型参数，只优化输入图像扰动 \(\delta\)。

### 3.2 攻击者能力

攻击者获得 \(x_p\)，并使用 SD3/SD3.5 图像编辑系统进行编辑。一次编辑请求记为：

\[
\omega=(c,\sigma,\xi,g,N,s,\ldots),
\]

其中：

- \(c\) 是 prompt 或编辑文本；
- \(\sigma\) 是 img2img/SDEdit 噪声强度；
- \(\xi\) 是 latent noise；
- \(g\) 是 guidance scale；
- \(N\) 是 denoising steps；
- \(s\) 是 random seed；
- 省略号表示 negative prompt、scheduler variant、分辨率等其他设置。

攻击者不主动去除扰动，也不使用专门的 adaptive purification。若后续研究 adaptive attacker，则需要单独加入 JPEG、resize、crop、denoise、VAE round-trip、diffusion purification 等鲁棒性测试。

### 3.3 防护成功标准

理想成功条件同时包含不可感知性和编辑失败：

\[
\|x_p-x\|_\infty\le \epsilon,
\]

\[
d_{\mathrm{vis}}(x_p,x)\le \tau_{\mathrm{imperceptible}},
\]

\[
D_{\mathrm{edit}}(\operatorname{Edit}_\theta(x_p;\omega),\operatorname{Edit}_\theta(x;\omega))
\ge \tau_{\mathrm{edit}}.
\]

其中 \(d_{\mathrm{vis}}\) 可以由 LPIPS、PSNR、SSIM 或人工视觉检查近似，\(D_{\mathrm{edit}}\) 可以由 paired SDEdit 输出之间的 RMSE、LPIPS、CLIP prompt drop、source similarity drop 等指标近似。

---

## 4. 外层问题定义

最直接的防护目标是：

\[
\max_{\|\delta\|_\infty\le \epsilon}
\mathbb E_{\omega}
\left[
D_{\mathrm{edit}}
\left(
\operatorname{Edit}_\theta(x+\delta;\omega),
\operatorname{Edit}_\theta(x;\omega)
\right)
\right].
\]

该式中每个部分含义如下：

- \(\delta\) 是唯一被优化的变量；
- \(\|\delta\|_\infty\le\epsilon\) 是不可感知扰动约束；
- \(\omega\) 表示攻击者可能采用的编辑条件；
- \(\operatorname{Edit}_\theta(x+\delta;\omega)\) 是受保护图像的编辑输出；
- \(\operatorname{Edit}_\theta(x;\omega)\) 是原图的 clean paired 编辑输出；
- \(D_{\mathrm{edit}}\) 衡量两者差异；
- 期望 \(\mathbb E_\omega\) 表示防护不应只针对一个固定 seed 或一个固定噪声强度。

这个外层目标是语义上最准确的目标，但直接优化成本极高。若每个 PGD step 都完整展开 \(N\) 步 denoising，并对所有 \(\omega\) 求期望，则计算量近似为：

\[
O(K\cdot S\cdot N\cdot C_{\mathrm{MMDiT}}),
\]

其中 \(K\) 是 PGD steps，\(S\) 是每步采样的条件数，\(N\) 是 denoising steps，\(C_{\mathrm{MMDiT}}\) 是一次 transformer forward/backward 的成本。对 SD3.5 medium 这类模型，完整展开会带来显著显存和训练时间负担。

因此 v3.1 使用局部但更贴近 SD3 训练目标的代理目标：流匹配预测误差。

---

## 5. SD3 / 修正流 / SDEdit 基础

### 5.1 流匹配的抽象形式

Flow matching 将生成过程视为从噪声分布到数据分布的连续流。设某一时刻的 latent 状态为 \(z_t\)，模型学习速度场：

\[
v_\theta(z_t,t,c)\approx u_t,
\]

其中 \(u_t\) 是训练目标速度。对 rectified flow，常见构造会在数据样本 \(z_0\) 与噪声样本 \(z_1\) 之间定义插值路径，并学习从当前点指向目标方向的速度。不同实现会把时间方向、sigma、scheduler timestep 写成不同约定，因此工程中必须验证：

1. transformer 输出是 velocity 还是 scheduler 所需的其他参数；
2. \(t\) 与 \(\sigma\) 的映射方向；
3. 目标速度 \(u_\sigma\) 的符号；
4. scheduler step 中 \(h_k\) 的正负和尺度。

理论上，FMP 只要求 \(u_\sigma\) 与当前实现中的 SD3 流匹配目标保持一致。

### 5.2 SDEdit 加噪 latent 与编辑起点

SDEdit 的基本思想是：先将输入图像加噪到某个中间状态，再从该状态开始 denoise。噪声强度越小，结果越保留原图；噪声强度越大，编辑自由度越高。

在当前工程形式中：

\[
z_\sigma=(1-\sigma)z+\sigma\xi.
\]

对受保护图像：

\[
z^p_\sigma=(1-\sigma)z_p+\sigma\xi.
\]

两者的初始差异为：

\[
\Delta z_\sigma
=z^p_\sigma-z_\sigma
=(1-\sigma)(z_p-z).
\]

该式说明，在较大 \(\sigma\) 下，原图扰动对初始加噪 latent 的直接影响会被 \((1-\sigma)\) 缩小；但速度场 \(v_\theta\) 的非线性响应仍可能放大这种差异。因此 FMP 不只最大化 \(z^p_\sigma-z_\sigma\)，而是最大化模型在 \(z^p_\sigma\) 上对流匹配目标的预测错误。

---

## 6. v2.1 回顾与放弃原因

### 6.1 v2.1 的目标结构

v2.1 把攻击目标写成：

\[
L_{\mathrm{attack}}=\lambda_T L_T+\lambda_M L_M.
\]

其中 \(L_T\) 是 textual / VAE latent target-pull，\(L_M\) 是 MMDiT 结构项。v2.1 A/B/C/D 的机制项分别表示：

1. A：text injection disruption，破坏 image token 接收 text token 信息的 attention injection；
2. B：image-stream feature divergence，使 protected feature 偏离 clean feature；
3. C：shared-noise trajectory / velocity divergence；
4. D：modality balance disruption，破坏 image/text stream 能量比例和 covariance alignment。

这些项作为结构诊断有价值，但已有实验显示：当它们在 v2.1 中以默认尺度与 textual target-pull 加权相加时，并没有形成有效的模式差异。因此 v3.1 不再把 A/B/C/D 改写为新的默认含义，而是保留 v2.1 joint 口径作为 baseline，并在主实验中显式提高 MMDiT 项权重，以检验结构项在足够梯度尺度下是否能产生外部编辑影响。

### 6.2 实验现象：模式信号基本消失

已有 paired SDEdit 数据表明，v2.1 A/B/C/D 与 origin/main 的 O baseline 的外部结果几乎相同。本文将该 origin/main baseline 记为 `textual_semantic_joint`，因为它不是 v2.1 O，而是 \(L_T\) 与 \(\|v_\theta\|\) 的联合目标。以主实验为例，v2.1 A/B/C 与该 baseline 的 mean \(l2\_edit\_rmse\) 差异约为 \(10^{-5}\) 到 \(10^{-4}\) 量级，远小于 epsilon 和 sigma 带来的变化。扰动之间的 pairwise cosine 约为 0.972 到 0.975，sign agreement 约为 0.959 到 0.961，说明不同 mode 实际生成的扰动方向高度相似。

这意味着问题不是绘图造成的，也不是某一个指标文件的小错误导致的，而是优化目标没有让模式特定的结构项真正支配 PGD 更新。

### 6.3 原因一：文本项梯度主导

当前代码的 joint objective 为：

\[
L_{\mathrm{joint}}=\lambda_T L_T+\lambda_M L_M.
\]

梯度为：

\[
\nabla_x L_{\mathrm{joint}}
=
\lambda_T\nabla_x L_T+\lambda_M\nabla_x L_M.
\]

PGD sign update 使用：

\[
\operatorname{sign}(\nabla_x L_{\mathrm{joint}}).
\]

因此真正决定扰动方向的是梯度，而不是 loss 数值本身。loss 数值大不必然代表梯度大，loss 数值小也不必然代表梯度小。但在已有数据中，MMDiT 结构项的梯度确实远小于 textual 梯度。最终 step 的典型比例为：

\[
\frac{\|\nabla L_M\|_2}{\|\nabla L_T\|_2}
\approx
10^{-5}\sim 10^{-4}
\quad
\text{for v2.1 A/B/C/D},
\]

而 `textual_semantic_joint` 中的 semantic velocity-norm 分量也仅约 \(10^{-2}\)。因此在 \(\lambda_T=\lambda_M=1\) 或轻微权重消融下，PGD 更新方向几乎完全由 \(L_T\) 决定。

这解释了为什么 v2.1 A/B/C/D 的 loss 有值、有梯度、有曲线，但最终扰动与 `textual_semantic_joint` 或 textual-like baseline 高度相似。

### 6.4 原因二：clean-adv 差异目标在 \(\delta\approx0\) 附近弱梯度

B/C 类 clean-adv divergence 目标通常可抽象为：

\[
L_{\mathrm{div}}(x_p)
=
d(F(x_p),\operatorname{sg}(F(x))).
\]

若 \(x_p=x\)，则：

\[
F(x_p)=F(x),
\quad
L_{\mathrm{div}}=0.
\]

以 squared distance 为例：

\[
L_{\mathrm{div}}=\|F(x_p)-\operatorname{sg}(F(x))\|_2^2.
\]

对 \(x_p\) 求梯度：

\[
\nabla_{x_p}L_{\mathrm{div}}
=
2J_F(x_p)^\top(F(x_p)-F(x)).
\]

当 \(x_p=x\) 时：

\[
\nabla_{x_p}L_{\mathrm{div}}=0.
\]

这说明 clean-adv divergence 目标在起点附近天然存在零梯度或极弱梯度。random start 或 warm start 可以缓解，但这只是优化启动技巧，并没有解决目标本身与编辑失败之间的间接性问题。

相比之下，FMP 是预测误差目标：

\[
L_{\mathrm{FMP}}(x_p)
=
\|v_\theta(z^p_\sigma,t_\sigma,c)-u_\sigma\|_2^2.
\]

即使 \(x_p=x\)，只要 \(v_\theta(z_\sigma,t_\sigma,c)\neq u_\sigma\)，其梯度为：

\[
\nabla_{x_p}L_{\mathrm{FMP}}
=
2J_{v,E}(x_p)^\top
\left(
v_\theta(z^p_\sigma,t_\sigma,c)-u_\sigma
\right),
\]

一般不必为零。这里 \(J_{v,E}\) 是从图像到 VAE latent 再到 transformer 速度预测的复合 Jacobian。这个性质使 FMP 比 clean-adv 特征差异目标更不依赖随机初始化。

### 6.5 原因三：A/B/D 状态没有严格对齐 SDEdit 轨迹

v2.1 的 A/B/D 在实现上主要对 \(z_p=E(x_p)\) 或随机 timestep 上的 MMDiT 内部状态做比较，但 paired SDEdit 的真实编辑起点是：

\[
z^p_\sigma=(1-\sigma)z_p+\sigma\xi.
\]

如果优化时使用的状态分布与评估时的状态分布不一致，则内部结构项即使变化，也未必能在实际去噪轨迹上产生可见影响。该问题不是“是否必须完整对齐所有 SDEdit step”的二元问题，而是代理目标的采样分布必须覆盖攻击者编辑链路中的关键状态。FMP 至少将优化输入移动到 SDEdit 加噪 latent 分布上。

### 6.6 原因四：O 的 semantic 分量不是 SD3 flow-matching prediction error

无论是 origin/main 的 O，还是 v2.1 修正后的 O，其 semantic 分量都不是 standalone SD3 flow-matching prediction error，而是 velocity-norm proxy。以 v2.1 O 为准，它的 joint objective 为：

\[
L_{\mathrm{v2.1},O}
=
w_T L_T+w_M L_{\mathrm{sem}}.
\]

其中 \(L_T\) 是 VAE latent target-pull textual loss，\(L_{\mathrm{sem}}\) 是 semantic velocity-norm proxy：

\[
L_{\mathrm{sem}}=\|v_\theta(z_t,t,c)\|.
\]

其中 `O` 只指 v2.1 的 `mode=O`。origin/main 的 `mode=O` 必须显式写作“origin/main 的 O”或 `textual_semantic_joint`，不能再简称为 O。该 semantic proxy 最大化预测速度的范数，但 SD3 flow matching 的训练误差不是速度范数本身，而是预测速度与目标速度之间的误差：

\[
\|v_\theta(z_t,t,c)-u_t\|_2^2.
\]

因此 v2.1 O 不是 SD3 语义攻击的严格对应版本；`textual_semantic_joint` 也只是 origin/main 的 O baseline 的显式名称，不是 SD3-correct objective。FMP 正是将这一点修正为流匹配预测误差。

### 6.7 v2.1 在 v3.1 中的保留方式与 E 模式

v3.1 保留 O/A/B/C/D 的 v2.1 joint 口径：

\[
L_{\mathrm{joint},M}(x_p)
=
w_TL_T^*(x_p)+w_ML_M(x_p),
\quad M\in\{O,A,B,C,D\}.
\]

其中 \(L_O=L_{\mathrm{sem}}=\|v_\theta(z_p,t,c)\|_2\)，A/B/C/D 分别使用对应 MMDiT 机制项。为了避免再次被 textual 梯度完全主导，主实验将显式设置：

\[
w_T=1,\quad w_M=100000.
\]

这不是声称 \(100000\) 是理论最优权重，而是作为机制放大实验：如果 O/A/B/C/D 在足够 MMDiT 梯度尺度下仍不能显著改变 paired SDEdit 结果，则说明问题更可能来自代理目标本身，而不只是权重过小。

同时，v3.1 新增 E 模式，从 v2.1 O 中拆分出 semantic velocity-norm 分量，并只优化该项：

\[
L_E(x_p)=L_{\mathrm{sem}}(x_p)=\|v_\theta(z_p,t,c)\|_2.
\]

E 的 PGD 更新为：

\[
x_{p,k+1}
=
\Pi_{\mathcal B_\infty(x,\epsilon)}
\left(
x_{p,k}
+
\alpha\operatorname{sign}
\left(
\nabla_{x_p}L_E(x_{p,k})
\right)
\right).
\]

因此：

1. O/A/B/C/D 默认指 v2.1 joint objective；
2. 主实验中 O/A/B/C/D 使用 \(w_M=100000\) 的 MMDiT 放大设置；
3. E 是 semantic-only velocity-norm baseline，不加入 textual target-pull；
4. E 与 O 的差别是：O 同时含 \(L_T^*\) 与 \(L_{\mathrm{sem}}\)，E 只含 \(L_{\mathrm{sem}}\)；
5. 不将 O/A/B/C/D/E 的内部指标上升解释为外部编辑失败，除非 paired SDEdit 指标同步验证。

---

## 7. v3.1：流匹配防护目标

### 7.1 核心思想

SD3/SD3.5 的编辑链路依赖模型在加噪 latent 上预测正确速度场。若受保护图像 \(x_p\) 经过 VAE 编码和 SDEdit 加噪后得到 \(z^p_\sigma\)，而模型在该状态上的预测：

\[
v_\theta(z^p_\sigma,t_\sigma,c)
\]

显著偏离流匹配目标速度：

\[
u_\sigma,
\]

则后续去噪轨迹会沿错误方向更新，最终编辑输出更可能偏离 clean paired 编辑结果。

因此 v3.1 不再首先问“哪个 attention map 被破坏”，而是首先问“SD3 在真实编辑起点附近是否预测错了速度场”。

### 7.2 FMP 目标函数

定义：

\[
z_p=E(x_p),
\quad
x_p=\operatorname{clip}(x+\delta),
\quad
\|\delta\|_\infty\le\epsilon.
\]

采样 SDEdit 噪声强度 \(\sigma\)、latent 噪声 \(\xi\)、文本条件 \(c\)，构造：

\[
z^p_\sigma=(1-\sigma)z_p+\sigma\xi.
\]

令 \(t_\sigma\) 是 scheduler 中与 \(\sigma\) 对应的 timestep，令 \(u_\sigma\) 是当前 SD3 scheduler / 流匹配约定下的目标速度。FMP 定义为：

\[
L_{\mathrm{FMP}}
=
\mathbb E_{\sigma,\xi,c}
\left[
\ell_{\mathrm{FMP}}(x_p;\sigma,\xi,c)
\right],
\]

其中单样本 loss 为：

\[
\ell_{\mathrm{FMP}}(x_p;\sigma,\xi,c)
=
\frac{
\left\|v_\theta(z^p_\sigma,t_\sigma,c)-\operatorname{sg}(u_\sigma)\right\|_2^2
}{
\operatorname{sg}(\left\|u_\sigma\right\|_2^2)+\eta
}.
\]

每个部分的作用如下：

- 分子是流预测误差，越大表示模型在该加噪 protected state 上越不符合原训练目标；
- \(\operatorname{sg}(u_\sigma)\) 防止优化通过目标速度本身的计算路径改变 reference；
- 分母做尺度归一化，避免不同 \(\sigma\)、不同图像、不同 prompt 下目标速度范数差异过大；
- \(\eta\) 防止 \(\|u_\sigma\|_2^2\) 太小时数值不稳定；
- 期望 \(\mathbb E_{\sigma,\xi,c}\) 表示扰动要对不同编辑强度、噪声和文本条件有平均效果。

最终攻击问题为：

\[
\max_{\|\delta\|_\infty\le\epsilon}
L_{\mathrm{FMP}}(x+\delta).
\]

### 7.3 与外层编辑目标的关系

外层目标关注最终编辑输出差异：

\[
D_{\mathrm{edit}}
\left(
\operatorname{Edit}_\theta(x_p;\omega),
\operatorname{Edit}_\theta(x;\omega)
\right).
\]

FMP 关注的是编辑轨迹早期状态上的局部流预测误差。它不是外层目标的严格等价变换，而是一个计算可承受、与 SD3 训练目标一致、与 SDEdit 状态分布对齐的代理目标。

换言之，FMP 的理论保证是局部机制层面的：

1. 它确实增加受保护加噪 latent 上的流预测误差；
2. 该误差会通过 Euler-style denoising step 影响后续 latent；
3. 多 step 累积后可能造成最终编辑结果偏移；
4. 最终是否足够大必须由 paired SDEdit 实验验证。

这比 v2.1 直接把内部结构项当成主目标更严谨，因为 FMP 的目标量是 SD3 采样器实际使用的速度预测误差。

---

## 8. 数学推导与性质

### 8.1 从 SDEdit 状态到流预测误差

SDEdit 对输入图像 latent \(z_p\) 的 noised 状态为：

\[
z^p_\sigma=(1-\sigma)z_p+\sigma\xi.
\]

SD3 transformer 在该状态上输出：

\[
v_p=v_\theta(z^p_\sigma,t_\sigma,c).
\]

若当前 scheduler/flow matching 约定下的目标速度为 \(u_\sigma\)，则理想模型满足：

\[
v_\theta(z^p_\sigma,t_\sigma,c)\approx u_\sigma.
\]

FMP 最大化：

\[
\|v_p-u_\sigma\|_2^2.
\]

因此它把扰动优化直接放在模型训练目标的反方向：训练希望 prediction error 小，防护攻击希望 protected state 上 prediction error 大。

### 8.2 FMP 的梯度路径

FMP 的单样本 loss 为：

\[
\ell=
\frac{\|v_\theta(z^p_\sigma,t_\sigma,c)-\operatorname{sg}(u_\sigma)\|_2^2}
{\operatorname{sg}(\|u_\sigma\|_2^2)+\eta}.
\]

设：

\[
r=v_\theta(z^p_\sigma,t_\sigma,c)-\operatorname{sg}(u_\sigma),
\]

\[
a=\operatorname{sg}(\|u_\sigma\|_2^2)+\eta.
\]

则：

\[
\ell=\frac{r^\top r}{a}.
\]

因为 \(a\) 被 stop-gradient 视为常量，所以：

\[
\nabla_{x_p}\ell
=
\frac{2}{a}
J_{v,E}(x_p)^\top r.
\]

其中：

\[
J_{v,E}(x_p)
=
\frac{\partial v_\theta(z^p_\sigma,t_\sigma,c)}{\partial x_p}
\]

是 VAE encoder、SDEdit mixing 和 MMDiT transformer 的复合 Jacobian。由于：

\[
z^p_\sigma=(1-\sigma)E(x_p)+\sigma\xi,
\]

有：

\[
\frac{\partial z^p_\sigma}{\partial x_p}
=(1-\sigma)J_E(x_p).
\]

因此：

\[
J_{v,E}(x_p)
=
\frac{\partial v_\theta}{\partial z^p_\sigma}
(1-\sigma)J_E(x_p),
\]

更严格地写作链式乘积：

\[
J_{v,E}(x_p)
=
J_{v}(z^p_\sigma,t_\sigma,c)\,(1-\sigma)J_E(x_p).
\]

该式说明：

1. \(\sigma\) 越大，图像扰动经 VAE latent 传入 noised state 的直接系数 \((1-\sigma)\) 越小；
2. 但 \(J_v\) 的非线性敏感性仍可能在特定 \(\sigma\) 区间放大扰动影响；
3. 因此实验中需要多 \(\sigma\) 采样，而不应只使用一个固定 noise level。

### 8.3 为什么 FMP 不易起始零梯度

对于 clean-adv feature divergence：

\[
L_{\mathrm{div}}=\|F(x_p)-\operatorname{sg}(F(x))\|_2^2.
\]

当初始化 \(x_p=x\) 时：

\[
F(x_p)-F(x)=0,
\]

所以：

\[
\nabla_{x_p}L_{\mathrm{div}}
=
2J_F(x_p)^\top(F(x_p)-F(x))
=0.
\]

这就是 B/C 等 clean-adv divergence 目标在起点附近容易出现弱梯度的数学原因。

FMP 则是：

\[
L_{\mathrm{FMP}}=\|v_\theta(z^p_\sigma,t_\sigma,c)-u_\sigma\|_2^2.
\]

当 \(x_p=x\) 时：

\[
\nabla_{x_p}L_{\mathrm{FMP}}
=
2J_{v,E}(x)^\top
(v_\theta(z_\sigma,t_\sigma,c)-u_\sigma).
\]

除非模型在该状态上预测完全准确，即：

\[
v_\theta(z_\sigma,t_\sigma,c)=u_\sigma,
\]

或者 residual 恰好落在 Jacobian 的零空间中，否则梯度一般不为零。实际深度生成模型不会在所有样本、所有 \(\sigma\)、所有 prompt 上做到零 prediction error，因此 FMP 更自然地提供非零起始优化方向。

### 8.4 Euler 采样误差传播

设 clean trajectory 和 protected trajectory 的第 \(k\) 步 latent 分别为 \(z_k\) 与 \(z^p_k\)。采样更新抽象为：

\[
z_{k+1}=z_k+h_kv_\theta(z_k,t_k,c),
\]

\[
z^p_{k+1}=z^p_k+h_kv_\theta(z^p_k,t_k,c).
\]

定义状态差异：

\[
\Delta z_k=z^p_k-z_k,
\]

速度差异：

\[
\Delta v_k
=
v_\theta(z^p_k,t_k,c)-v_\theta(z_k,t_k,c).
\]

两式相减得到：

\[
\Delta z_{k+1}
=
\Delta z_k+h_k\Delta v_k.
\]

这就是 v3.1 计划中要求写入的误差传播公式。

进一步考察平方范数：

\[
\|\Delta z_{k+1}\|_2^2
=
\|\Delta z_k+h_k\Delta v_k\|_2^2.
\]

展开：

\[
\|\Delta z_{k+1}\|_2^2
=
\|\Delta z_k\|_2^2
+2h_k\langle\Delta z_k,\Delta v_k\rangle
+h_k^2\|\Delta v_k\|_2^2.
\]

该式说明：

1. 若 \(\Delta v_k\) 与 \(\Delta z_k\) 同向，则 \(2h_k\langle\Delta z_k,\Delta v_k\rangle\) 放大已有差异；
2. 即使二者近似正交，\(h_k^2\|\Delta v_k\|_2^2\) 仍会增加轨迹差异；
3. 若不同 step 的方向相互抵消，则最终差异可能不大，因此单步局部目标不能替代完整 paired SDEdit 实验。

FMP 最大化的是 protected state 上的 prediction error，而不是直接最大化 \(\Delta v_k\)。但当 prediction error 使 protected velocity 偏离正确 flow direction 时，\(\Delta v_k\) 往往增大，从而通过上述递推影响后续 trajectory。

### 8.5 PGD 更新与投影正确性证明

FMP 使用 \(L_\infty\)-bounded sign-PGD：

\[
x_{k+1}
=
\Pi_{\mathcal B_\infty(x,\epsilon)\cap[-1,1]}
\left(
x_k+\alpha\operatorname{sign}(\nabla_x L_{\mathrm{FMP}}(x_k))
\right).
\]

其中：

- \(x_k\) 是第 \(k\) 步的 protected image candidate；
- \(\alpha\) 是 PGD step size；
- \(\operatorname{sign}(\cdot)\) 将梯度转为每像素符号方向；
- \(\Pi\) 同时执行 \(L_\infty\) ball 投影和图像范围裁剪。

对任意像素维度 \(i\)，投影可写作：

\[
\Pi_{\mathcal B_\infty}(x')_i
=
\min\left\{
x_i+\epsilon,
\max\left\{
x_i-\epsilon,x'_i
\right\}
\right\}.
\]

因此：

\[
x_i-\epsilon
\le
\Pi_{\mathcal B_\infty}(x')_i
\le
x_i+\epsilon.
\]

两边减去 \(x_i\)：

\[
-\epsilon
\le
\Pi_{\mathcal B_\infty}(x')_i-x_i
\le
\epsilon.
\]

所以：

\[
|\Pi_{\mathcal B_\infty}(x')_i-x_i|\le\epsilon.
\]

对所有像素取最大值得：

\[
\|\Pi_{\mathcal B_\infty}(x')-x\|_\infty\le\epsilon.
\]

再与 \([-1,1]\) 裁剪相交后，图像范围也满足：

\[
x_{k+1}\in[-1,1]^{3\times H\times W}.
\]

因此 PGD 每一步都满足扰动预算和图像合法范围。

### 8.6 蒙特卡洛估计的合理性

FMP 是期望目标：

\[
L_{\mathrm{FMP}}
=
\mathbb E_{\sigma,\xi,c}
\left[
\ell_{\mathrm{FMP}}(x_p;\sigma,\xi,c)
\right].
\]

若每个 PGD step 独立采样：

\[
(\sigma_j,\xi_j,c_j)\sim p(\sigma,\xi,c),
\quad
j=1,\ldots,m,
\]

则多样本估计为：

\[
\widehat L_{\mathrm{FMP}}
=
\frac{1}{m}
\sum_{j=1}^m
\ell_{\mathrm{FMP}}(x_p;\sigma_j,\xi_j,c_j).
\]

其期望为：

\[
\mathbb E[\widehat L_{\mathrm{FMP}}]
=
\frac{1}{m}\sum_{j=1}^m
\mathbb E_{\sigma_j,\xi_j,c_j}
\left[
\ell_{\mathrm{FMP}}(x_p;\sigma_j,\xi_j,c_j)
\right]
=
L_{\mathrm{FMP}}.
\]

因此 \(\widehat L_{\mathrm{FMP}}\) 是无偏估计。单样本版本 \(m=1\) 也是无偏估计，但方差更大；multi-\(\sigma\) 或 multi-noise 版本会降低方差，代价是每个 PGD step 需要更多 transformer forward/backward。

这说明 FMP 的 single-step / single-\(\sigma\) 形式不是“随便取一步”，而是期望目标的随机近似。关键不是必须完整展开所有 denoising steps，而是采样分布 \(p(\sigma,\xi,c)\) 要与威胁模型中的编辑请求分布一致。

---

## 9. 目标层级

v3.1 使用以下层级，避免再次把所有项混成一个启发式 joint loss。

### 9.1 层级 0：不可感知约束

\[
\|\delta\|_\infty\le\epsilon,
\quad
d_{\mathrm{vis}}(x_p,x)\le\tau.
\]

这是硬约束或验收标准，不应作为任意权重混入主目标后被优化器牺牲。

### 9.2 层级 1：编辑失败外层目标

\[
\max_{\|\delta\|_\infty\le \epsilon}
\mathbb E_{\omega}
\left[
D_{\mathrm{edit}}
(\operatorname{Edit}_\theta(x_p;\omega),\operatorname{Edit}_\theta(x;\omega))
\right].
\]

这是研究问题的真实目标，但计算过重。

### 9.3 层级 2：FMP 主代理目标

\[
\max_{\|\delta\|_\infty\le\epsilon}
L_{\mathrm{FMP}}.
\]

这是 v3.1 的主优化目标。它对齐 SD3 的 flow matching 训练误差，也对齐 SDEdit 的 noised latent 起点。

### 9.4 层级 3：轨迹转移辅助项

可选辅助项是 one-step transition separation：

\[
T_\theta(z,t,c)=z+h_t v_\theta(z,t,c),
\]

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

该项更接近局部 trajectory separation，但需要 clean reference forward，计算成本高于纯 FMP。v3.1 中它只作为后续消融，不作为主方法。

### 9.5 层级 4：O/A/B/C/D/E baseline 与机制诊断

O/A/B/C/D 是 v2.1 joint baseline：

\[
L_{\mathrm{joint},M}=w_TL_T^*+w_ML_M,\quad M\in\{O,A,B,C,D\}.
\]

主实验设置 \(w_T=1,w_M=100000\)，用于检验被放大的 v2.1 机制项是否能产生外部编辑影响。E 是从 O 中拆出的 semantic-only baseline：

\[
L_E=L_{\mathrm{sem}}=\|v_\theta(z_p,t,c)\|_2.
\]

这些 baseline 可用于回答：

1. FMP 是否改变 text-to-image injection；
2. FMP 是否改变 image-stream feature；
3. FMP 是否引入 velocity divergence；
4. FMP 是否破坏 image/text modality balance。

无论作为独立方法还是诊断指标，O/A/B/C/D/E 都不是 FMP 的数学替代物。论文中不能只用内部指标证明防护成功，必须报告 paired SDEdit 外部效果。

---

## 10. 合理性分析

### 10.1 与 SD3 训练目标一致

FMP 最大化：

\[
\|v_\theta(z^p_\sigma,t_\sigma,c)-u_\sigma\|_2^2.
\]

这与 flow matching 训练中最小化 prediction error 的方向相反。因此它是 SD3 语义攻击的自然形式，比直接最大化 \(\|v_\theta\|\) 更符合模型训练目标。

### 10.2 与 SDEdit 编辑状态一致

FMP 使用：

\[
z^p_\sigma=(1-\sigma)E(x_p)+\sigma\xi.
\]

这正是 img2img/SDEdit 编辑链路中的初始 noised latent 类型。相比 v2.1 中某些直接在 \(E(x_p)\) 或随机 timestep feature 上做 divergence 的目标，FMP 的输入分布更贴近实际编辑过程。

### 10.3 计算负载可控

完整外层目标需要每个 PGD step 展开多个 denoising steps。FMP 每个 PGD step 只需要对 sampled noised latent 做一次或少量几次 transformer forward/backward。计算量约为：

\[
O(K\cdot m\cdot C_{\mathrm{MMDiT}}),
\]

其中 \(m\) 是每步采样的 \(\sigma/\xi/c\) 数量。若 \(m=1\)，成本与 v2.1 单步 MMDiT 目标接近；若 \(m>1\)，可用更高成本换取更低方差。

### 10.4 O/A/B/C/D 权重放大的意义与边界

v2.1 的一个深层问题是 textual loss 与 A/B/C/D 机制项没有自然共同尺度，导致：

\[
L_{\mathrm{v2.1}}=w_TL_T^*+w_ML_M
\]

容易被 \(w_T\nabla L_T^*\) 主导。v3.1 主实验不再默认使用 \(w_M=1\)，而是将 O/A/B/C/D 的 MMDiT 项统一放大：

\[
w_T=1,\quad w_M=100000.
\]

这样可以把“结构项无效”与“结构项梯度尺度过小”两个问题拆开：若放大后仍然无法造成 paired SDEdit 变化，则更有力地说明代理目标本身与编辑失败之间缺乏充分因果联系。

但这不意味着应该把多个机制项直接混合成新的主目标。若设置：

\[
L=\beta_A L_A+\beta_B L_B+\beta_C L_C+\beta_D L_D
\]

仍会引入大量调参和解释困难。FMP 将主目标统一为一个 flow prediction error，不需要在 A/B/C/D 之间分配权重。O/A/B/C/D 的 \(w_M=100000\) 应被解释为公平放大 baseline 的实验设置，而不是最终理论目标。E 则作为 semantic-only 对照，用来观察 velocity-norm 分量脱离 textual target-pull 后的效果。

---

## 11. 局限性与必须实验验证的命题

FMP 不是完整理论证明。以下命题必须由实验验证：

1. \(L_{\mathrm{FMP}}\) 增大会不会稳定导致 paired SDEdit 输出偏移；
2. single-\(\sigma\) 与 multi-\(\sigma\) 哪个在同等计算预算下更有效；
3. 哪些 \(\sigma\) 区间最关键；
4. FMP 是否显著优于 random \(L_\infty\)、`textual_only`、E，以及 \(w_M=100000\) 的 O/A/B/C/D；
5. FMP 生成的扰动是否保持足够不可感知；
6. FMP 是否迁移到不同 prompt、不同 seed、不同 guidance scale、不同 SD3/SD3.5 checkpoint；
7. JPEG、resize、crop、VAE round-trip 等处理是否削弱扰动。

理论文档只证明 FMP 与 SD3 流匹配目标和 SDEdit 加噪状态的局部一致性，不证明最终图像一定失败。最终结论必须由实验协议中的成对评估给出。

---

## 12. 与相关工作的关系

### 12.1 PhotoGuard

PhotoGuard 将图像扰动用于提高恶意图像编辑成本，强调在图像进入生成式编辑系统前加入保护信号。v3.1 与其任务动机一致，但目标模型换成 SD3/SD3.5，并把代理目标对齐到 rectified flow / flow matching。

### 12.2 AdvDM

AdvDM 研究 diffusion models 的对抗样本，说明扩散模型也存在可优化的脆弱输入方向。v3.1 继承其“攻击生成模型内部预测过程”的思想，但不直接复用 SD v1.x 噪声预测目标。

### 12.3 MIST

MIST 提出 textual loss 和 semantic loss 的 joint design。`origin/main` 的 SD3 `mode=O` baseline 可视为这种 joint design 在 SD3 上的迁移：textual target-pull 与 semantic velocity-norm proxy 共同进入目标函数。v2.1 又尝试把 textual target-pull 与 MMDiT 结构项组合为 O/A/B/C/D。v3.1 保留 O/A/B/C/D 作为放大后的 joint baseline，并新增 E 来单独测试 semantic velocity-norm 分量；同时保留“最大化生成模型预测误差”的核心思想，将主方法从速度范数或 SD v1.x 的 \(\epsilon\)-prediction 思路修正为 SD3 的 flow-matching prediction error。

### 12.4 SDEdit

SDEdit 给出从 noised image state 开始编辑的基本范式。v3.1 的关键改动就是把优化状态放到 SDEdit noised latent：

\[
z^p_\sigma=(1-\sigma)E(x_p)+\sigma\xi.
\]

因此 FMP 是面向图像编辑链路而不是纯文本生成链路的防护目标。

### 12.5 SD3 / 修正流 Transformer

SD3 使用 MMDiT 和 rectified flow transformer。v3.1 的 \(v_\theta\) 直接对应 SD3 transformer 输出，\(u_\sigma\) 对应 flow matching target。因此 v3.1 是面向 SD3/SD3.5 架构和训练范式的重构，而不是对旧 UNet diffusion attack 的表面迁移。

---

## 13. 附录：v2.1 O/A/B/C/D 与 E 模式

本附录记录 `fix/sd3-objective-sanity` 分支中的 O/A/B/C/D 口径，并定义 v3.1 新增的 E 模式。除非特别说明，O/A/B/C/D 均指 v2.1 joint objective；E 指从 v2.1 O 中拆出的 semantic-only velocity-norm objective。

### 13.1 v2.1 的共同优化约定

v2.1 沿用 SD3/MMDiT 攻击框架，但对 `origin/main` 的符号、梯度路径和诊断字段做了修正。设：

- \(x\)：原始图像；
- \(x_p=\operatorname{clip}(x+\delta)\)：受保护图像；
- \(\|\delta\|_\infty\le \epsilon\)：扰动预算；
- \(E\)：SD3 VAE encoder；
- \(z=E(x)\)，\(z_p=E(x_p)\)：clean/protected latent；
- \(y\)：target image；
- \(z_y=E(y)\)：target latent；
- \(c\)：文本条件；
- \(t\)：采样 timestep；
- \(v_\theta(\cdot,t,c)\)：SD3 MMDiT / transformer 输出的 velocity prediction；
- \(w_T=\texttt{textual\_weight}\)，\(w_M=\texttt{mmdit\_weight}\)：文本项和 MMDiT 项权重。

v2.1 代码默认没有 `mmdit_weight=100000`，默认配置为：

\[
w_T=1,\quad w_M=1.
\]

v2.1 将目标统一为“越大越像攻击成功”，并默认使用 `opt_direction=maximize`。在 `textual_objective=toward_target` 时，textual target-pull 被写成：

\[
L_T^*(x_p)
=
-\|z_p-z_y\|_2^2.
\]

因此 PGD 最大化 \(L_T^*\) 等价于最小化 \(\|z_p-z_y\|_2^2\)，即把 protected latent 拉向 target latent。对任意机制项 \(L_M\)，v2.1 的 joint objective 为：

\[
L_{\mathrm{v2.1}}(x_p)
=
w_TL_T^*(x_p)+w_ML_M(x_p).
\]

PGD 更新为：

\[
x_{p,k+1}
=
\Pi_{\mathcal B_\infty(x,\epsilon)}
\left(
x_{p,k}
+
\alpha\operatorname{sign}
\left(
\nabla_{x_p}L_{\mathrm{v2.1}}(x_{p,k})
\right)
\right).
\]

这里 \(\Pi_{\mathcal B_\infty(x,\epsilon)}\) 是 \(L_\infty\) ball 投影。该式说明，O/A/B/C/D 与 textual target-pull 组成 joint objective。v3.1 主实验为了避免 MMDiT 结构项被 textual 梯度淹没，将 O/A/B/C/D 统一设置为：

\[
w_T=1,\quad w_M=100000,
\quad M\in\{O,A,B,C,D\}.
\]

该设置是实验放大策略，而不是声称 v2.1 默认权重就是 100000。E 模式不使用该 joint 结构，只优化从 O 中拆出的 semantic loss。

### 13.2 v2.1 O：semantic velocity-norm proxy

v2.1 的 O 是本文后续 O 的唯一含义。v2.1 代码内部曾把同一语义项暴露为 `O_repo` 别名，但该别名不再作为本文或后续实验的方法名。O 的 MMDiT 项为：

\[
L_O(x_p)
=
\|v_\theta(z_p,t,c)\|_2.
\]

总目标为：

\[
L_{\mathrm{v2.1},O}
=
w_TL_T^*
+
w_ML_O.
\]

若 \(v_\theta(z_p,t,c)\neq 0\)，则：

\[
\nabla_{x_p}L_O
=
J_{v,E}(x_p)^\top
\frac{
v_\theta(z_p,t,c)
}{
\|v_\theta(z_p,t,c)\|_2
},
\]

其中 \(J_{v,E}(x_p)\) 是从图像 \(x_p\) 经 VAE encoder 到 MMDiT velocity prediction 的复合 Jacobian。该梯度通常非零，因此 O 可以提供优化方向。

但 O 的局限也很明确。它只增大 velocity prediction 的范数，不比较 prediction 与 flow target 是否一致。SD3 flow matching 的训练目标是：

\[
\|v_\theta(z_t,t,c)-u_t\|_2^2,
\]

而不是 \(\|v_\theta(z_t,t,c)\|_2\)。因此 O 是 semantic velocity-norm proxy，不是 SD3-correct prediction error。它适合作为 v2.1 机制基线，不应被解释为 FMP。

### 13.3 E：从 v2.1 O 拆分出的 semantic-only velocity-norm

E 模式只优化 v2.1 O 的 MMDiT semantic 分量：

\[
L_E(x_p)
=
L_{\mathrm{sem}}(x_p)
=
\|v_\theta(z_p,t,c)\|_2.
\]

它不包含 textual target-pull：

\[
L_E\neq w_TL_T^*+w_ML_{\mathrm{sem}}.
\]

因此 E 与 O 的区别是：

\[
L_O^{\mathrm{joint}}
=
w_TL_T^*+w_ML_{\mathrm{sem}},
\qquad
L_E=L_{\mathrm{sem}}.
\]

E 的作用是隔离 velocity-norm semantic proxy 自身的防护能力。如果 E 有效而 O/A/B/C/D 不稳定，说明 textual target-pull 可能干扰了语义扰动方向；如果 E 和 O 都无效，而 FMP 有效，则说明问题更可能来自 velocity-norm proxy 本身不等价于 SD3 flow-matching prediction error。

### 13.4 v2.1 A：cross-modal text injection disruption

MMDiT joint attention 中，设 image query、text key、text value 分别为 \(Q_I,K_T,V_T\)。image token 接收 text token 信息的 attention injection 可以写作：

\[
A_{I\to T}
=
\operatorname{softmax}
\left(
\frac{Q_IK_T^\top}{\sqrt d}
\right),
\quad
R_{I\leftarrow T}
=
A_{I\to T}V_T.
\]

其中 \(d\) 是每个 attention head 的维度，\(R_{I\leftarrow T}\) 表示 text information 注入 image stream 的结果。

v2.1 对 clean image 计算 detached reference：

\[
R^{clean}_{I\leftarrow T}
=
\operatorname{sg}
\left(
A^{clean}_{I\to T}V_T^{clean}
\right),
\]

对 protected image 保持可导：

\[
R^p_{I\leftarrow T}
=
A^p_{I\to T}V_T^p.
\]

主要损失为归一化 injection divergence：

\[
L_{A,\mathrm{inj}}
=
\frac{
\|R^p_{I\leftarrow T}-R^{clean}_{I\leftarrow T}\|_2^2
}{
\operatorname{sg}(\|R^{clean}_{I\leftarrow T}\|_2^2)+\eta
}.
\]

同时记录并可加入小权重 attention entropy：

\[
H(A^p_{I\to T})
=
-
\sum_j A^p_{I\to T,j}
\log(A^p_{I\to T,j}+\eta).
\]

v2.1 中实际机制项可概括为：

\[
L_A
=
L_{A,\mathrm{inj}}
+
0.05H(A^p_{I\to T}).
\]

相较 `origin/main`，v2.1 的关键修正是：adv forward 中的 attention / injection 不再 detach，clean reference 才 detach。因此：

\[
\nabla_{x_p}L_A
=
J_{R,E}(x_p)^\top
\nabla_{R^p}L_A
\]

可以通过 attention processor、MMDiT、VAE encoder 回传到输入图像。A 的作用是破坏文本条件向图像 token 的注入方式。其有效性依赖于这种 injection divergence 是否会传导到后续 velocity prediction 和最终 SDEdit 输出；因此它适合作为机制目标和诊断指标，但仍需 paired SDEdit 验证。

### 13.5 v2.1 B：image-stream feature divergence

设第 \(l\) 个 MMDiT block 的 image-stream feature 为：

\[
F_l^p=F_l(x_p,t,c),
\quad
F_l=F_l(x,t,c).
\]

v2.1 对 feature 做 token-wise normalization：

\[
\widehat F_l
=
\frac{F_l-\mu(F_l)}{\sigma(F_l)+\eta}.
\]

定义 Gram matrix：

\[
G(\widehat F_l)
=
\frac{1}{ND}\widehat F_l^\top\widehat F_l,
\]

其中 \(N\) 是 token 数，\(D\) 是 feature channel 数。v2.1 的 B 项为：

\[
L_B
=
\frac{1}{|\mathcal L|}
\sum_{l\in\mathcal L}
\left[
1-\cos(\widehat F_l^p,\operatorname{sg}(\widehat F_l))
+
\|G(\widehat F_l^p)-\operatorname{sg}(G(\widehat F_l))\|_1
\right].
\]

第一项降低 protected feature 与 clean feature 的方向相似度，第二项破坏 feature channel 的二阶相关结构。v2.1 相比 `origin/main` 的主要修正是：

1. clean features detach；
2. protected features 保持可导；
3. cosine 项写成 \(1-\cos\)，符合 maximize convention；
4. Gram 项写成正的 distance，而不是负号；
5. 可通过 `capture_blocks` 选择少量 block，降低计算成本。

在 \(x_p=x\) 且 clean/protected forward 完全相同时：

\[
F_l^p=F_l,
\quad
L_B=0,
\quad
\nabla_{x_p}L_B=0
\]

通常成立。因此 B 属于 clean-adv divergence 目标，在 \(\delta=0\) 附近可能存在弱梯度或零梯度。random start 可以缓解优化启动问题，但不能改变 B 的代理目标性质：它证明的是 image-stream representation 被拉开，不直接证明 flow prediction 变错。

### 13.6 v2.1 C：shared-noise trajectory / velocity divergence

v2.1 C 是最接近 SDEdit 轨迹的 v2.1 A/B/C/D 项。给定 shared noise \(\xi\) 和噪声强度 \(\sigma\)，构造：

\[
z_\sigma
=
(1-\sigma)z+\sigma\xi,
\]

\[
z^p_\sigma
=
(1-\sigma)z_p+\sigma\xi.
\]

clean 与 protected 的 velocity prediction 为：

\[
v_c
=
v_\theta(z_\sigma,t_\sigma,c),
\]

\[
v_p
=
v_\theta(z^p_\sigma,t_\sigma,c).
\]

v2.1 C 使用 detached clean prediction 作为 reference：

\[
L_C
=
1-
\cos
\left(
v_p,
\operatorname{sg}(v_c)
\right).
\]

最大化 \(L_C\) 等价于降低 \(v_p\) 与 \(v_c\) 的 cosine similarity，使 protected state 的局部 velocity direction 偏离 clean state。若使用 Euler-style 更新：

\[
z_{k+1}=z_k+h_kv_k,
\]

则 clean/protected 差异满足：

\[
\Delta z_{k+1}
=
\Delta z_k+h_k\Delta v_k,
\quad
\Delta v_k=v_p-v_c.
\]

因此 C 的机制解释是：若 protected velocity 与 clean velocity 在方向上分离，则后续 denoising trajectory 更可能分离。

但是 C 也有两个关键限制。第一，它的 reference 是模型对 clean state 的预测 \(v_c\)，不是 flow-matching 目标速度。第二，它主要约束方向，而不直接约束 velocity prediction error 的幅度。若 \(x_p=x\)，则 \(z^p_\sigma=z_\sigma\)，\(v_p=v_c\)，有：

\[
L_C=0.
\]

对 cosine loss 求导，令：

\[
\operatorname{cos}(a,b)
=
\frac{\langle a,b\rangle}{\|a\|_2\|b\|_2}.
\]

对 \(a\) 的梯度为：

\[
\nabla_a\operatorname{cos}(a,b)
=
\frac{b}{\|a\|_2\|b\|_2}
-
\frac{\langle a,b\rangle a}{\|a\|_2^3\|b\|_2}.
\]

当 \(a=b\neq 0\) 时：

\[
\nabla_a\operatorname{cos}(a,b)
=
\frac{a}{\|a\|_2^2}
-
\frac{\|a\|_2^2a}{\|a\|_2^4}
=0.
\]

因此若从 \(\delta=0\) 开始，C 的 clean-adv divergence 形式天然可能出现起始零梯度。random start 可以让 \(v_p\neq v_c\)，从而恢复梯度。

### 13.7 v2.1 D：modality balance disruption

MMDiT 同时维护 image stream 与 text stream。设第 \(l\) 个 block 的 image/text features 为：

\[
I_l^p,\quad T_l^p.
\]

定义 stream energy：

\[
E_I^p
=
\operatorname{mean}((I_l^p)^2),
\quad
E_T^p
=
\operatorname{mean}((T_l^p)^2).
\]

v2.1 使用 image/text energy ratio 的 clean-relative deviation：

\[
R_l^p
=
\log\frac{E_I^p}{E_T^p},
\quad
R_l
=
\operatorname{sg}
\left(
\log\frac{E_I}{E_T}
\right),
\]

\[
L_{D,\mathrm{ratio}}
=
(R_l^p-R_l)^2.
\]

同时定义 normalized feature 的 channel covariance：

\[
C_I^p
=
\frac{(\widehat I_l^p)^\top\widehat I_l^p}{N_I},
\quad
C_T^p
=
\frac{(\widehat T_l^p)^\top\widehat T_l^p}{N_T}.
\]

用 CKA 衡量 image/text covariance alignment：

\[
\operatorname{CKA}(C_I^p,C_T^p)
=
\frac{
\langle C_I^p,C_T^p\rangle_F
}{
\|C_I^p\|_F\|C_T^p\|_F+\eta
}.
\]

v2.1 D 可概括为：

\[
L_D
=
\frac{1}{|\mathcal L|}
\sum_{l\in\mathcal L}
\left[
L_{D,\mathrm{ratio}}
+
0.1(1-\operatorname{CKA}(C_I^p,C_T^p))
\right].
\]

该项试图破坏 image/text stream 的能量比例和 covariance alignment。相比 `origin/main`，v2.1 不再简单最大化 \(-\operatorname{Var}(I)\) 或相关性项，而是使用 clean-relative ratio deviation 和 \(1-\mathrm{CKA}\)。其有效性仍属于机制层解释：它可能改变 MMDiT 内部双流耦合，但是否造成外部编辑失败必须由 paired SDEdit 指标验证。

### 13.8 v2.1 C 与 FMP 的详细差别

v2.1 C 与 FMP 都作用在 SDEdit 加噪 latent 和 SD3 velocity prediction 层，因此它们相邻，但不重复。

v2.1 C 的目标是：

\[
L_C
=
1-\cos
\left(
v_\theta(z^p_\sigma,t_\sigma,c),
\operatorname{sg}(v_\theta(z_\sigma,t_\sigma,c))
\right).
\]

它的 reference 是 clean image 在同一噪声 \(\xi\) 下的模型预测：

\[
v_c=v_\theta(z_\sigma,t_\sigma,c).
\]

因此 v2.1 C 的机制项回答的问题是：

\[
\text{protected velocity 是否偏离 clean velocity?}
\]

FMP 的目标是：

\[
L_{\mathrm{FMP}}
=
\frac{
\|v_\theta(z^p_\sigma,t_\sigma,c)-\operatorname{sg}(u_\sigma)\|_2^2
}{
\operatorname{sg}(\|u_\sigma\|_2^2)+\eta
}.
\]

它的 reference 是 protected state 对应的 flow-matching target：

\[
u_\sigma=\xi-z_p
\]

在当前理论约定下成立；实际实现必须按 diffusers SD3 scheduler 的符号和尺度验证。因此 FMP 回答的问题是：

\[
\text{protected velocity 是否偏离正确 flow target?}
\]

v2.1 C 与 FMP 的数学对象不同：

\[
L_C \approx D(v_p,v_c),
\quad
L_{\mathrm{FMP}}\approx D(v_p,u_p).
\]

只有在以下强条件同时成立时，二者才可能局部相近：

1. clean prediction 几乎等于 clean target：
   \[
   v_\theta(z_\sigma,t_\sigma,c)\approx u_c=\xi-z;
   \]
2. clean target 与 protected target 方向几乎一致：
   \[
   u_c\parallel u_p=\xi-z_p;
   \]
3. 只关心方向，不关心幅度；
4. protected perturbation 足够小，使得 \(z_p\approx z\)。

这些条件在真实优化中一般不应假设成立。尤其是：

\[
u_p-u_c
=
(\xi-z_p)-(\xi-z)
=
-(z_p-z).
\]

只要扰动改变了 latent，protected target velocity 本身就与 clean target velocity 不同。v2.1 C 的机制项让 \(v_p\) 远离 \(v_c\)，但并不保证 \(v_p\) 远离 \(u_p\)。如果 clean prediction \(v_c\) 本身有误差，C 还可能继承 clean prediction error 的偏差。FMP 则直接把 reference 换成 flow target \(u_p\)，与 SD3 的训练目标一致。

从展开式也能看出差异。FMP 的未归一化分子为：

\[
\|v_p-u_p\|_2^2
=
\|v_p\|_2^2
-2\langle v_p,u_p\rangle
+\|u_p\|_2^2.
\]

其中交叉项 \(-2\langle v_p,u_p\rangle\) 直接惩罚预测速度与正确目标速度的对齐。C 的 cosine 形式为：

\[
1-
\frac{
\langle v_p,v_c\rangle
}{
\|v_p\|_2\|v_c\|_2
},
\]

它只惩罚 \(v_p\) 与 \(v_c\) 的方向一致性，不直接约束 \(v_p\) 与 \(u_p\) 的误差，也弱化了幅度信息。

从优化性质看，若 \(\delta=0\)，C 满足 \(v_p=v_c\)，容易出现：

\[
\nabla_{x_p}L_C=0.
\]

而 FMP 在 \(\delta=0\) 时为：

\[
L_{\mathrm{FMP}}(x)
=
\|v_\theta(z_\sigma,t_\sigma,c)-u_\sigma\|_2^2.
\]

只要模型在该样本、该 \(\sigma\)、该 prompt 上不是完全零误差，梯度通常不必为零。这是 FMP 比 C 更不依赖 random start 的重要原因。

从计算成本看：

- C 需要 clean forward 与 protected forward，成本约为两次 transformer forward；
- FMP 只需要 protected forward 和解析 target \(u_\sigma\)，成本约为一次 transformer forward；
- C 的 reference 是模型预测，FMP 的 reference 是训练目标。

因此 v3.1 C 应定位为继承 v2.1 C 机制项的 trajectory separation proxy；FMP 才是 flow-matching prediction error objective。二者可以同表比较，但不应视为重复方法。若讨论包含 textual target-pull 的历史版本，应写作 v2.1 C。

### 13.9 FMP 与直接预测结果类目标的差别

初版设计中曾记录：“ABCD 与 O 的结构不一样，直接用噪声预测的结果作为优化的依据不能很好地优化。”这里的“直接用噪声预测的结果”不是指所有使用 denoiser / transformer 输出的目标都无效，而是指把旧 SD v1.x / MIST 风格的 raw denoiser output proxy 直接迁移到 SD3/MMDiT 上，例如只优化预测结果的范数，或在未对齐 SD3 flow target 与 SDEdit state 的情况下使用预测输出作为语义损失。

为统一表述，令 raw prediction proxy 写成：

\[
L_{\mathrm{raw}}(x_p)
=
\phi(v_\theta(\tilde z,t,c)),
\]

其中 \(\tilde z\) 是某个 latent state，\(\phi\) 是只依赖预测结果本身的函数。v2.1 O 是其中一种典型形式：

\[
L_O(x_p)
=
\|v_\theta(z_p,t,c)\|_2.
\]

如果为了方便推导，把范数写成平方形式：

\[
L_{\mathrm{norm}}(x_p)
=
\|v_\theta(z_p,t,c)\|_2^2,
\]

则对 \(x_p\) 的梯度为：

\[
\nabla_{x_p}L_{\mathrm{norm}}
=
2J_v(x_p)^\top v_\theta(z_p,t,c),
\]

其中 \(J_v(x_p)=\partial v_\theta(E(x_p),t,c)/\partial x_p\)。该梯度只要求 prediction 本身变大，而不判断 prediction 是否朝正确 flow direction 对齐。

FMP 使用的不是 raw prediction，而是 prediction error：

\[
L_{\mathrm{FMP}}(x_p)
=
\frac{
\|v_\theta(z^p_\sigma,t_\sigma,c)-\operatorname{sg}(u_\sigma)\|_2^2
}{
\operatorname{sg}(\|u_\sigma\|_2^2)+\eta
}.
\]

忽略归一化分母并记 \(v_p=v_\theta(z^p_\sigma,t_\sigma,c)\)、\(u_p=u_\sigma\)，其主要梯度方向为：

\[
\nabla_{x_p}L_{\mathrm{FMP}}
\approx
2J_{v,\sigma}(x_p)^\top(v_p-\operatorname{sg}(u_p)).
\]

因此 raw norm 与 FMP 的梯度方向差别为：

\[
\nabla L_{\mathrm{FMP}}-\nabla L_{\mathrm{norm}}
\approx
-2J^\top \operatorname{sg}(u_p),
\]

在 \(u_p\neq 0\) 且 \(J^\top u_p\neq 0\) 时通常不为零。也就是说，FMP 并不是把 raw prediction norm 换个名字，而是在优化方向中显式加入了“远离正确目标速度”的项。

这个差别可以从一维投影模型中更直观看到。假设在某个局部方向上：

\[
v_p=k u_p,
\quad
u_p\neq 0.
\]

raw norm 得到：

\[
L_{\mathrm{norm}}
=
k^2\|u_p\|_2^2.
\]

只要 \(|k|\) 变大，该目标就认为攻击更强。可是当 \(k=1\) 时：

\[
v_p=u_p,
\]

模型预测完全正确，编辑轨迹并没有被该局部目标破坏；此时 raw norm 仍可能因为 \(\|u_p\|\) 很大而给出较大的 loss。

FMP 则为：

\[
L_{\mathrm{FMP}}
\propto
\|k u_p-u_p\|_2^2
=
(k-1)^2\|u_p\|_2^2.
\]

当 \(k=1\) 时，FMP 为零；当 \(k\) 偏离 1 时，FMP 增大。这说明 FMP 度量的是“prediction 是否错”，而 raw norm 度量的是“prediction 是否大”。二者的优化语义不同。

从展开式看，FMP 还包含 raw norm 缺失的方向对齐项：

\[
\|v_p-u_p\|_2^2
=
\|v_p\|_2^2
-2\langle v_p,u_p\rangle
+\|u_p\|_2^2.
\]

其中 \(-2\langle v_p,u_p\rangle\) 是关键。最大化 FMP 会降低 \(v_p\) 与 \(u_p\) 的正确对齐，或者扩大二者的幅度误差；最大化 raw norm 只会推动 \(\|v_p\|\) 变大。若 \(v_p\) 变大但方向仍与 \(u_p\) 对齐，raw norm 会增大，但 flow prediction 可能仍是可用的。

这也解释了初版设计中“结构不一样”的含义。SD v1.x / MIST 的 semantic loss 常围绕 \(\epsilon\)-prediction 或 denoiser residual 建模，而 SD3/SD3.5 使用 MMDiT 与 rectified flow / flow matching。若只把旧方法中的“噪声预测结果”替换成 SD3 的 \(v_\theta\)，并优化 \(\|v_\theta\|\) 或类似 raw output proxy，就没有真正对齐 SD3 的训练目标：

\[
\min_\theta
\mathbb E
\left[
\|v_\theta(z_t,t,c)-u_t\|_2^2
\right].
\]

FMP 的修正不是放弃 denoiser / transformer prediction，而是把优化对象从 raw prediction 改为 prediction error：

\[
\text{raw proxy: } \phi(v_\theta)
\quad\Longrightarrow\quad
\text{FMP: } \|v_\theta-u_t\|^2.
\]

同时，FMP 把输入状态从一般 latent \(z_p\) 或随机 feature state 改为 SDEdit 加噪 latent：

\[
z^p_\sigma=(1-\sigma)E(x_p)+\sigma\xi.
\]

因此，FMP 同时修正了两个问题：

1. **目标误差口径**：从 raw prediction / prediction norm 转为 flow-matching prediction error；
2. **编辑状态口径**：从未必对齐编辑链路的 latent 或 feature state，转为 SDEdit noised latent state。

如果某个方法已经使用了正确 target，例如 SD v1.x 中的：

\[
\|\epsilon_\theta(x_t,t)-\epsilon\|_2^2,
\]

那么它在形式上与 FMP 的思想相近，都是 prediction error objective。区别在于 SD3 中应使用 velocity / flow target \(u_t\)，而不是 \(\epsilon\)-prediction target；并且应在 SD3 img2img/SDEdit 的 noised latent 分布上采样。因此，FMP 不是退回旧的 \(L_T+L_S\)，而是把 \(L_S\) 从 raw 或错误 target 的 semantic proxy 重构为 SD3-correct flow-matching error。

最后，FMP 与 O/A/B/C/D/E 的关系也应明确。A/B/D 是 MMDiT 内部结构代理，C 是 clean/protected trajectory separation proxy，O/E 是 velocity-norm semantic proxy；FMP 是训练目标层的 prediction error proxy。O/A/B/C/D 可以作为 \(w_M=100000\) 的 v2.1 joint baseline 与 FMP 同表比较，E 可以作为 semantic-only baseline 与 O 对照；它们都不是 FMP 的数学替代物。

### 13.10 v2.1 机制项的作用与有效性边界

v2.1 的 O/A/B/C/D 与 E 有重要作用：

1. 它们修复了 `origin/main` 中多个符号和 detach 问题；
2. 它们提供了 MMDiT 内部机制维度的可解释指标；
3. 它们为 \(w_M=100000\) 的放大 baseline 提供机制项定义和工程实现基础；
4. E 隔离 O 的 semantic velocity-norm 分量，帮助判断 O 的效果来自 semantic proxy 还是 textual target-pull；
5. C 可以衡量 shared-noise velocity divergence，帮助分析 FMP 是否真的造成轨迹分离。

但 v2.1 joint 机制方法仍有边界：

1. O/A/B/C/D 与 textual target-pull 组成 joint objective，即使 \(w_M=100000\) 也需要检查梯度比例；
2. A/B/D 主要是内部结构代理，不直接等于编辑失败；
3. B/C 这类 clean-adv divergence 目标在 \(\delta=0\) 附近可能弱梯度或零梯度；
4. O 是 velocity-norm proxy，不是 flow-matching prediction error；
5. C 虽然对齐了 shared-noise SDEdit state，但 reference 仍是 clean prediction，不是 protected flow target。

因此 v3.1 不是否定 v2.1 的机制价值，而是重新分配层级：O/A/B/C/D 作为 \(w_M=100000\) 的放大 joint baseline，E 作为 semantic-only baseline，FMP 作为主攻击目标。

---

## 14. 参考文献

1. SD3 / Rectified Flow Transformer: [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)
2. Flow Matching: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
3. Rectified Flow: [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
4. SDEdit: [Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)
5. PhotoGuard: [Raising the Cost of Malicious AI-Powered Image Editing](https://arxiv.org/abs/2302.06588)
6. AdvDM: [Adversarial Examples for Diffusion Models](https://arxiv.org/abs/2302.04578)
7. MIST: [Towards Improved Adversarial Examples for Diffusion Models](https://arxiv.org/abs/2305.12683)
8. EditShield: [Protecting Unauthorized Image Editing by Instruction-guided Diffusion Models](https://arxiv.org/abs/2311.12066)
9. DiffusionGuard: [DiffusionGuard](https://arxiv.org/abs/2410.05694)
10. DIA: [DIA](https://arxiv.org/abs/2510.00778)
