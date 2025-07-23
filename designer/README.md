# 🧬 De Novo 多肽/糖肽设计器

一款用于多肽与糖肽从头设计 (`de novo design`) 的命令行工具，专为科研人员设计。该工具通过调用本地部署的 `Boltz-WebUI` 预测服务作为计算后端，在序列空间中进行探索，以发现具有高结合潜力的新分子。

## ✨ 核心特性

### 🔬 基础功能
- **生物学约束**: 设计糖肽时，工具会自动验证并确保糖基连接到化学上兼容的氨基酸残基上（例如，N-连锁聚糖连接到天冬酰胺'N'）。
- **pLDDT指导的突变**: 演化过程优先在结构预测置信度较低 (pLDDT分数低) 的区域引入突变，以高效探索构象空间。
- **并行评估**: 在每个演化代数中，通过多线程并行提交和评估多个候选序列，显著加快设计周期。

### 🚀 增强版功能
- **自适应突变策略**: 5种智能突变策略自动选择和学习
  - 保守突变：基于BLOSUM62矩阵的高分替换
  - 激进突变：大范围序列空间探索  
  - Motif导引：基于有益模式的定向突变
  - 能量导引：基于能量景观的温度控制
  - 多样性驱动：防止群体过度收敛
- **Pareto多目标优化**: 同时优化ipTM和pLDDT，避免单一目标偏向
- **智能收敛检测**: 自动早停机制，防止过度训练和资源浪费
- **序列模式学习**: 位置特异性偏好学习和motif自动发现
- **群体多样性维护**: 实时监测和调整探索策略


## 🔧 环境准备

1.  **Boltz-WebUI 服务**: 确保 `Boltz-WebUI` 平台正在本地运行且网络可访问。
2.  **Python 依赖**: 安装所需的 Python 库。
    ```bash
    pip install requests pyyaml numpy
    ```

## 🚀 使用指南

通过在终端运行 `run_design.py` 脚本来启动设计流程。

### 步骤 1: 准备模板YAML文件

创建一个YAML文件，用于描述系统中静态不变的部分（例如，受体蛋白）。待设计的链（binder）不应包含在此模板中，脚本会依据命令行参数动态生成。

#### **示例 1: 标准蛋白Binder设计**

模板应包含除待设计链之外的所有链。

**`template_protein.yaml`:**

```yaml
version: 1
sequences:
- protein:
    id: A
    # 此处填写靶蛋白的完整序列
    sequence: MTEYKLVVVGAGGVGKSALTVQFVQGIFVEYDPTHFESTEKT.... 
    msa: empty
# 注意: 待设计的链（例如 B 链）将由脚本自动添加。
```

#### **示例 2: 糖肽设计**

对于糖肽设计，模板只包含受体蛋白。

**`template_glycopeptide.yaml`:**

```yaml
version: 1
sequences:
- protein:
    id: A
    # 靶点受体蛋白的完整序列
    sequence: MTEYKLVVVGAGGVGKSALTVQFVQGIFVEYDPTHFESTEKT....
    msa: empty
# 脚本将根据命令行参数自动添加肽链、聚糖以及它们之间的共价键。
```

### 步骤 2: 设置API令牌

建议将API密钥设置为环境变量。

```bash
# Linux / macOS
export API_SECRET_TOKEN='your-super-secret-and-long-token'

# Windows (Command Prompt)
set API_SECRET_TOKEN="your-super-secret-and-long-token"
```

也可以通过 `--api_token` 参数直接提供。

### 步骤 3: 运行设计脚本

以下为运行脚本的命令示例。

#### **命令示例 1: 基础设计（自动使用增强功能）**

```bash
python run_design.py \
    --yaml_template /path/to/your/template_protein.yaml \
    --binder_chain "B" \
    --binder_length 100 \
    --iterations 50 \
    --population_size 16 \
    --num_elites 4 \
    --weight-iptm 0.6 \
    --weight-plddt 0.4 \
    --output_csv "binder_design_run_summary.csv"
```

#### **命令示例 2: 平稳优化模式（推荐）**

适用于需要稳定探索的场景，减少过度波动：

```bash
python run_design.py \
    --yaml_template /path/to/your/template_protein.yaml \
    --binder_chain "B" \
    --binder_length 12 \
    --initial_binder_sequence "GGLVYEWQRWAG" \
    --iterations 20 \
    --population_size 16 \
    --num_elites 4 \
    --convergence-window 5 \
    --convergence-threshold 0.001 \
    --max-stagnation 3 \
    --initial-temperature 1.0 \
    --output_csv "binder_optimization_stable.csv"
```

#### **命令示例 3: 激进探索模式**

适用于需要快速突破局部最优的场景：

```bash
python run_design.py \
    --yaml_template /path/to/your/template_protein.yaml \
    --binder_chain "B" \
    --binder_length 15 \
    --iterations 30 \
    --population_size 12 \
    --num_elites 2 \
    --convergence-window 3 \
    --max-stagnation 2 \
    --initial-temperature 2.0 \
    --min-temperature 0.2 \
    --output_csv "aggressive_design_run.csv"
```

#### **命令示例 4: 糖肽设计（增强版）**

此示例设计一个长度为15个残基的肽，并在其第3个位置连接一个甘露糖（'MAN'）：

```bash
python run_design.py \
    --yaml_template /path/to/your/template_glycopeptide.yaml \
    --binder_chain "B" \
    --binder_length 15 \
    --iterations 30 \
    --population_size 12 \
    --num_elites 3 \
    --glycan_ccd "MAN" \
    --glycosylation_site 3 \
    --glycan_chain "C" \
    --weight-iptm 0.7 \
    --weight-plddt 0.3 \
    --convergence-window 4 \
    --max-stagnation 3 \
    --output_csv "glycopeptide_enhanced_design.csv" \
    --keep_temp_files
```

#### **命令示例 5: 传统模式（兼容性）**

如需使用传统算法（不推荐，仅用于对比）：

```bash
python run_design.py \
    --yaml_template /path/to/your/template_protein.yaml \
    --binder_chain "B" \
    --binder_length 20 \
    --iterations 25 \
    --disable-enhanced \
    --output_csv "traditional_design.csv"
```

## ⚙️ 命令行参数详解

#### 输入与目标定义

  - `--yaml_template` **(必需)**: 模板YAML文件的路径。
  - `--binder_chain` **(必需)**: 待设计肽链的链ID (例如, "B")。
  - `--binder_length` **(必需)**: 待设计肽链的长度。
  - `--initial_binder_sequence`: (可选) 提供一个初始肽链序列作为优化的起点。如果未提供，将从随机序列开始。

#### 演化算法控制

  - `--iterations`: 优化循环（代数）的次数。 (默认: `20`)
  - `--population_size`: 每代并行评估的候选数量。 (默认: `8`)
  - `--num_elites`: 保留到下一代的顶级候选（精英）数量。必须小于`population_size`。 (默认: `2`)
  - `--weight-iptm` / `--weight-plddt`: 分别是复合评分函数中 `ipTM` 和 `pLDDT` 的权重。 (默认: `0.7` / `0.3`)

#### 增强功能选项 🚀

  - `--enable-enhanced` / `--disable-enhanced`: 启用/禁用增强版功能。(默认: 启用)
  - `--convergence-window`: 收敛检测的滑动窗口大小。较小值更敏感。 (默认: `5`)
  - `--convergence-threshold`: 收敛检测的分数方差阈值。较小值更严格。 (默认: `0.001`)
  - `--max-stagnation`: 触发早停的最大停滞周期数。较小值更激进。 (默认: `3`)
  - `--initial-temperature`: 自适应突变的初始温度。较高值更探索性。 (默认: `1.0`)
  - `--min-temperature`: 自适应突变的最小温度。较高值保持更多随机性。 (默认: `0.1`)

#### 🎯 优化模式推荐

| 模式 | 收敛窗口 | 停滞阈值 | 初始温度 | 适用场景 |
|------|----------|----------|----------|----------|
| **平稳优化** | 5 | 3 | 1.0 | 标准设计，稳定收敛 |
| **激进探索** | 3 | 2 | 2.0 | 突破局部最优，快速探索 |
| **精细调优** | 7 | 4 | 0.8 | 已有好序列，精细优化 |
| **保守设计** | 6 | 5 | 0.5 | 对稳定性要求高的场景 |

#### 糖肽设计 (可选)

  - `--glycan_ccd`: 激活糖肽设计模式。提供聚糖的3字母PDB CCD代码 (例如, `MAN`, `NAG`, `GAL`)。 (默认: `None`)
  - `--glycosylation_site`: 在binder序列上共价连接聚糖的位置 **(1-based索引)**。如果使用了`--glycan_ccd`，此项为**必需**。
  - `--glycan_chain`: 分配给聚糖配体的唯一链ID。 (默认: `C`)

#### 输出与日志

  - `--output_csv`: 保存所有已评估设计结果的CSV汇总文件的路径。 (默认: `design_summary_<timestamp>.csv`)
  - `--keep_temp_files`: 若设置，则在运行结束后不删除临时工作目录。

#### API 连接

  - `--server_url`: 运行中的Boltz-WebUI预测API的URL。 (默认: `http://127.0.0.1:5000`)
  - `--api_token`: API密钥。建议通过 `API_SECRET_TOKEN` 环境变量设置。

## 📊 输出解读

程序运行结束后，会生成以下输出：

  - **控制台日志**: 实时显示演化进程、每一代的最佳分数、警告和错误信息。
  - **CSV汇总文件 (`--output_csv`)**: 包含所有已评估序列的详细信息及其评分指标（`composite_score`, `ipTM`, `pLDDT` 等）。
    - 🆕 **新增字段**: `mutation_strategy` (使用的突变策略), `is_pareto_optimal` (是否为Pareto最优解)
    - 文件已按综合分数从高到低排序。
  - **临时文件 (`--keep_temp_files`)**: 如果选择保留，工作目录中将包含每次API调用的输入（YAML）、输出（PDB/CIF）和置信度文件，可用于后续分析或调试。

### 📈 性能监控

增强版提供更丰富的运行状态信息：

```
[INFO] Generation 5 complete. Best score: 0.8542 (ipTM: 0.8901, pLDDT: 78.43)
[INFO] Enhanced features enabled with custom parameters
[DEBUG] Strategy success rates: conservative=0.75, aggressive=0.32, motif_guided=0.68
[DEBUG] Population diversity - similarity: 0.423, entropy: 2.87
[INFO] Convergence detected: score variance 0.0008 < threshold 0.001
[INFO] Early stopping triggered after 3 stagnation cycles
```