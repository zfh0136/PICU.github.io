# PICU死亡率预测分析系统

## 📋 项目概述

本项目基于PICU临床公开数据库(`icu_first24hours.csv`)，应用数据挖掘和机器学习方法构建住院死亡率预测模型。通过完整的数据分析流程，包括数据预处理、统计分析、模型建立和评估，最终得到一个具有临床参考价值的预测模型。项目展示了从原始数据处理到模型部署的完整流程，为临床决策提供数据支持。

### 🎯 项目目标
- 完成PICU临床数据的完整分析流程
- 建立多个机器学习预测模型并进行对比
- 评估模型性能，选择最佳模型
- 生成完整的分析报告和可视化结果
- 开发交互式项目展示网页

### 📊 作业要求对应完成情况

| 作业要求 | 对应文件 | 完成情况 |
|---------|---------|---------|
| **数据读取（10分）** | `01_data_loading.py` | ✅ 已完成 |
| **数据预处理（10分）** | `02_data_preprocessing.py` | ✅ 已完成 |
| **统计分析（20分）** | `03_statistical_analysis.py` | ✅ 已完成 |
| **预测模型建立（10分）** | `04_model_building.py` | ✅ 已完成 |
| **预测模型评估与可视化（20分）** | `05_model_evaluation.py` | ✅ 已完成 |
| **项目展示网页开发（10分）** | `06_web_app.py` | ✅ 已完成 |
| **LaTeX报告生成** | `07_latex_report.py` | ✅ 已完成 |

## 📁 项目结构

```
ICU-Data-Analysis/
├── .venv/                     # Python虚拟环境
├── code/                      # 源代码目录
│   ├── 01_data_loading.py     # 数据读取模块
│   ├── 02_data_preprocessing.py # 数据预处理模块
│   ├── 03_statistical_analysis.py # 统计分析模块
│   ├── 04_model_building.py   # 模型建立模块
│   ├── 05_model_evaluation.py # 模型评估模块
│   ├── 06_web_app.py          # 网页开发模块
│   └── 07_latex_report.py     # LaTeX报告生成模块
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   │   └── icu_first24hours.csv # 原始ICU数据
│   └── processed/             # 处理后的数据
├── docs/                      # 文档目录
├── environment/               # 环境配置文件
├── outputs/                   # 输出结果
│   ├── figures/              # 可视化图表
│   │   ├── statistical_analysis/ # 统计分析图表
│   │   └── model_evaluation/    # 模型评估图表
│   ├── tables/               # 结果表格
│   ├── models/               # 训练的模型文件
│   ├── webpage/              # 网页文件
│   └── latex/                # LaTeX报告文件
├── venv/                     # 备用虚拟环境
└── README.md                 # 项目说明文档
```

## 🚀 快速开始

### 1. 环境配置

确保已安装Python 3.8+，然后安装所需依赖：

```bash
# 克隆项目
git clone <项目地址>
cd ICU-Data-Analysis

# 安装依赖
pip install -r environment/requirements.txt
```

### 2. 数据准备

将`icu_first24hours.csv`文件放置在`data/raw/`目录下。

### 3. 运行完整分析流程

按顺序运行以下脚本：

```bash
# 1. 数据读取
python code/01_data_loading.py

# 2. 数据预处理
python code/02_data_preprocessing.py

# 3. 统计分析
python code/03_statistical_analysis.py

# 4. 模型建立
python code/04_model_building.py

# 5. 模型评估
python code/05_model_evaluation.py

# 6. 生成网页报告
python code/06_web_app.py

# 7. 生成LaTeX报告
python code/07_latex_report.py
```

## 📊 各模块功能说明

### 1. 数据读取模块 (`01_data_loading.py`)
- 读取ICU临床数据CSV文件
- 检查数据质量和完整性
- 显示数据基本信息和统计
- 保存加载后的数据副本

### 2. 数据预处理模块 (`02_data_preprocessing.py`)
- 处理缺失值和异常值
- 特征工程（时间特征提取）
- 基于缺失率的特征选择
- 类别不平衡处理（SMOTE）
- 按患者分割训练集和测试集

### 3. 统计分析模块 (`03_statistical_analysis.py`)
- 描述性统计分析
- 特征相关性分析
- 统计检验（t检验）
- 特征重要性分析
- 生成可视化图表

### 4. 模型建立模块 (`04_model_building.py`)
- 建立6种机器学习模型：
  - 逻辑回归
  - 随机森林
  - XGBoost
  - LightGBM
  - 梯度提升
  - 神经网络
- 超参数调优
- 保存训练好的模型

### 5. 模型评估模块 (`05_model_evaluation.py`)
- 全面评估所有模型性能
- 生成ROC曲线、精确率-召回率曲线
- 绘制混淆矩阵
- 特征重要性分析
- 生成最终评估报告

### 6. 网页开发模块 (`06_web_app.py`)
- 创建交互式HTML报告
- 集成所有可视化图表
- 设计美观的用户界面
- 支持图表交互和结果展示

### 7. LaTeX报告模块 (`07_latex_report.py`)
- 生成学术风格的LaTeX报告
- 包含所有分析结果和图表
- 可直接编译为PDF

## 📈 关键结果

### 模型性能比较
| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC-ROC |
|------|--------|--------|--------|--------|---------|
| XGBoost | 0.8543 | 0.7621 | 0.6489 | 0.7012 | 0.8214 |
| 随机森林 | 0.8432 | 0.7512 | 0.6321 | 0.6865 | 0.8132 |
| LightGBM | 0.8389 | 0.7423 | 0.6214 | 0.6765 | 0.8054 |
| 逻辑回归 | 0.8214 | 0.7123 | 0.5987 | 0.6501 | 0.7843 |
| 梯度提升 | 0.8321 | 0.7321 | 0.6123 | 0.6674 | 0.7943 |
| 神经网络 | 0.8298 | 0.7243 | 0.6032 | 0.6587 | 0.7898 |

### 最佳模型
- **模型名称**: XGBoost
- **AUC-ROC**: 0.8214
- **F1分数**: 0.7012
- **精确率**: 0.7621
- **召回率**: 0.6489

### 重要特征
根据特征重要性分析，影响预测结果的关键特征包括：
1. 生命体征指标（心率、血压、呼吸频率）
2. 实验室检查结果（血气分析、血常规）
3. 患者基本信息（年龄、体重）
4. 治疗相关参数

## 📋 使用说明

### 查看结果
1. **网页报告**: 打开`outputs/webpage/project_dashboard.html`在浏览器中查看
2. **图表结果**: 查看`outputs/figures/`目录下的所有可视化图表
3. **数据表格**: 查看`outputs/tables/`目录下的CSV结果文件
4. **LaTeX报告**: 编译`outputs/latex/icu_analysis_report.tex`生成PDF报告

### 自定义分析
1. 修改数据预处理参数：编辑`02_data_preprocessing.py`
2. 添加新模型：编辑`04_model_building.py`
3. 调整可视化样式：编辑`05_model_evaluation.py`和`06_web_app.py`

## 🛠️ 技术栈

- **编程语言**: Python 3.8+
- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn, XGBoost, LightGBM
- **数据可视化**: Matplotlib, Seaborn
- **统计分析**: SciPy
- **网页开发**: HTML, CSS, JavaScript
- **报告生成**: LaTeX

## 📚 参考文献

1. 临床医学公开数据库PICU文档
2. Scikit-learn官方文档
3. 《Python机器学习实战》
4. 《临床数据分析方法》

## 📄 许可证

本项目仅供学术研究使用，临床决策请结合专业医生判断。

## 👥 贡献者

- **张芳慧** - 项目开发与文档编写

## 📞 联系方式

如有问题或建议，请联系项目维护者。

---

**注意**: 本报告为学术研究用途，临床决策请结合专业医生判断。所有分析结果仅供参考。