# SD3 对抗扰动实验设计与实施路径

> 适用项目：`Zuxx129/Diff-Protect`
> 目标：给后续 ChatGPT / Codex / 其他 agent 一个可直接执行、可检查、可迭代的实验路线。
> 前置原则：先修梯度与方向，再跑实验；先做 paired evaluation，再做指标；先最小验真，再完整消融。

---

## 0. 总体路线

实验实施分为 8 个阶段：

| 阶段 | 名称 | 目标 | 是否阻塞后续 |
|---|---|---|---|
| S0 | 仓库冻结与分支管理 | 固定版本、创建修复分支 | 是 |
| S1 | 目标函数语义修复 | 统一 `larger-is-more-attack` 与 `opt_direction` | 是 |
| S2 | Hook 可导性修复 | 修复 A/B/D 的 `detach/no_grad` 梯度断路 | 是 |
| S3 | Debug 与方向验真 | 检查 gradient norm 与机制指标方向 | 是 |
| S4 | Paired SDEdit 链路 | 输出 clean/adv 成对编辑结果 | 是 |
| S5 | 指标与分析脚本 | 计算外部效果指标和内部机制指标 | 是 |
| S6 | 最小实验与迭代 | O/C 与 patched A/B/C/D 小规模验证 | 否 |
| S7 | 完整实验与论文级消融 | 模式对比、权重、预算、transfer、robustness | 否 |

任何 agent 执行时必须遵守：**S1-S5 没完成前，不允许生成主实验结论表。**

---

## 1. 仓库分支与文件规划

### 1.1 推荐分支

```bash
cd /path/to/Diff-Protect
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout -b fix/sd3-objective-sanity
git push -u origin fix/sd3-objective-sanity
```

### 1.2 需要修改的核心文件

| 文件 | 修改内容 |
|---|---|
| `configs/attack/base_sd3.yaml` | 添加 `opt_direction`、`textual_objective`、`debug_grad` 等字段 |
| `code/attacks_SD3.py` | 修复 A/B/D 可导性；重写 A/B/C/D 为 larger-is-more-attack；拆分 `O_repo/O_fair`；返回 diagnostics |
| `code/diff_mist_SD3.py` | 统一 loss 调用；增加 paired SDEdit；固定 seed；保存 clean_edit / adv_edit / metrics |
| `code/plot_loss.py` | 修复 O 标签；支持 diagnostics 曲线；支持多实验聚合 |
| `code/clip_similarity.py` | 改为 batchable 的 CLIP image-image 与 text-image 评价工具 |
| `README.md` | 更新运行命令、模式语义、实验阶段 |
| `SD3_对抗扰动设计v2.md` | 新设计文档 |

### 1.3 建议新增目录

```text
code/metrics/
code/analysis/
configs/attack/
scripts/
```

---

## 2. S1-S7 实施步骤

### S1: 目标函数语义修复
- opt_direction, objective_convention, textual_objective, seed, debug_grad
- g_mode 兼容字段
- 区分 loss_opt 与 diagnostics
- 修复 A/B/C/D 的 loss 符号方向
- 修复 O 模式标签与 O_repo/O_fair 命名

### S2: Hook 可导性修复
- clean detach, adv keep autograd
- Mode A 在线累计 attention scalar
- Mode B/D 正确保存 feature hook

### S3: Debug 与方向验真
- 打印梯度 norm
- 检查机制指标方向

### S4: Paired SDEdit
- 同 seed、sigma、prompt
- 输出 clean_edit 和 adv_edit
- 保存 metrics.json

### S5: 指标与分析脚本
- LPIPS, CLIPScore, L_inf, PSNR/SSIM, Source-image similarity, FID
- 机制指标 A/B/C/D
- 成功率计算
- aggregate CSV / JSON

### S6: 最小实验
- O_repo, C
- epsilon: 8,16
- steps: 10,20
- input_size: 512
- seeds: 0,1,2
- sigma: 0.3
- 输出 attacked.png, clean_edit.png, adv_edit.png, metrics.json

### S7: 完整实验
- modes: O_repo, O_fair, A, B, C, D
- epsilon: 4,8,16
- steps: 20,50
- input_size: 512
- seeds: 0-4
- sigma: 0.1,0.3,0.5
- 权重消融 / 预算消融 / 分辨率消融 / transfer / robustness

---

## 3. 可视化与统计
- 绘制 loss_total/textual/mmdit 曲线
- 绘制机制指标曲线
- epsilon-tradeoff曲线
- Paired comparison, bootstrap 95% CI

## 4. Agent 执行顺序
1. 确认分支与仓库
2. 提交 P0 文档
3. 实施 P0 修复
4. 梯度 sanity 检查
5. 方向 sanity 检查
6. Paired SDEdit
7. 指标脚本
8. 最小实验
9. 完整实验

## 5. 风险与回滚
- grad_mmdit_l2=0, loss NaN/Inf, L_inf 超过预算, paired SDEdit 不全, 1024 显存不稳
- 可接受但需记录的情况

## 6. 交付物
- configs/attack/debug_sd3.yaml, minimal_sd3.yaml, full_sd3.yaml
- code/metrics/*.py, code/analysis/*.py, scripts/*.sh
- out_sd3/<exp_id>/*
- SD3_对抗扰动设计v2.md
- SD3_实验设计与实施路径.md