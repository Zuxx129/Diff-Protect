# SD3 v3.1 Paper Figure Pipeline

本目录保存 v3.1 实验的论文图生成代码。旧入口仍可用，但新绘图逻辑统一维护在这里。

## 模块划分

- `style.py`：论文图共享样式、颜色、字体、图例和导出工具。
- `io.py`：CSV、loss `.npz`、实验目录和 paired SDEdit 图像读取。
- `loss_curves.py`：PGD loss 曲线、O/A/B/C/D loss components、FMP/step 诊断曲线。
- `metric_comparison.py`：主实验方法对比、epsilon/sigma 曲线、扰动-编辑权衡、paired difference。
- `ablations.py`：FMP single/multi、可选 step loss 消融图。
- `diagnostics.py`：loss、gradient、机制指标热力图和诊断曲线。
- `qualitative.py`：paired clean/adv SDEdit 定性图像网格。
- `make_all_figures.py`：一键生成所有可用图表，并写入 `figure_manifest.csv`。

## 默认目录

```text
out_sd3_v3/
  full_metrics.csv
  analysis/
    summary_by_method.csv
    ablation_summary.csv
    paired_differences.csv
  figures/
    main/
    ablation/
    diagnostics/
    loss/
    qualitative/
    figure_manifest.csv
```

## 一键运行

Windows PowerShell:

```powershell
.\scripts\draw_sd3_v3_figures.ps1
```

Linux/WSL/Git Bash:

```bash
bash scripts/draw_sd3_v3_figures.sh
```

也可以直接调用 Python 入口：

```bash
python -m draw.figures.make_all_figures --root out_sd3_v3 --loss-image suzume
```

若 `full_metrics.csv` 或 paired SDEdit 图像还不存在，对应图表会跳过；loss 曲线可从 `out_sd3_v3` 或 legacy `out_sd3` 中读取。

