"""
07_latex_report.py - LaTeX报告生成模块（优化版）- 只生成.tex文件
"""

import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

def prepare_latex_environment():
    """准备LaTeX环境"""
    project_root = Path(__file__).parent.parent
    latex_dir = project_root / "outputs" / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建图片目录
    figures_dir = latex_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"LaTeX环境准备完成: {latex_dir}")
    return latex_dir

def copy_figures_to_latex(latex_dir):
    """复制图表到LaTeX目录"""
    project_root = Path(__file__).parent.parent
    source_figures = project_root / "outputs" / "figures"
    
    # 复制所有图表
    figure_count = 0
    for subdir in ['statistical_analysis', 'model_evaluation']:
        source_dir = source_figures / subdir
        if source_dir.exists():
            for fig_file in source_dir.glob('*.png'):
                try:
                    shutil.copy2(fig_file, latex_dir / "figures" / fig_file.name)
                    figure_count += 1
                except Exception as e:
                    print(f"  复制图表失败 {fig_file.name}: {e}")
    
    print(f"已复制 {figure_count} 个图表到LaTeX目录")
    return list((latex_dir / "figures").glob('*.png'))

def load_results_for_latex():
    """加载所有分析结果"""
    project_root = Path(__file__).parent.parent
    tables_dir = project_root / "outputs" / "tables"
    
    results = {}
    
    # 加载关键结果文件
    try:
        results['model_comparison'] = pd.read_csv(tables_dir / 'model_performance_comparison.csv')
        print(f"  加载 model_comparison: {len(results['model_comparison'])} 行")
    except Exception as e:
        print(f"  加载 model_comparison 失败: {e}")
        results['model_comparison'] = None
    
    try:
        results['final_evaluation'] = pd.read_csv(tables_dir / 'final_model_evaluation.csv')
        print(f"  加载 final_evaluation: {len(results['final_evaluation'])} 行")
    except Exception as e:
        print(f"  加载 final_evaluation 失败: {e}")
        results['final_evaluation'] = None
    
    try:
        results['feature_importance'] = pd.read_csv(tables_dir / 'feature_importance_best_model.csv')
        print(f"  加载 feature_importance: {len(results['feature_importance'])} 行")
    except Exception as e:
        print(f"  加载 feature_importance 失败: {e}")
        results['feature_importance'] = None
    
    try:
        results['statistical_tests'] = pd.read_csv(tables_dir / 'statistical_tests_results.csv')
        print(f"  加载 statistical_tests: {len(results['statistical_tests'])} 行")
    except Exception as e:
        print(f"  加载 statistical_tests 失败: {e}")
        results['statistical_tests'] = None
    
    # 加载文本报告
    try:
        results['statistical_report'] = (tables_dir / 'statistical_report_summary.txt').read_text(encoding='utf-8')
        print(f"  加载 statistical_report: {len(results['statistical_report'])} 字符")
    except Exception as e:
        print(f"  加载 statistical_report 失败: {e}")
        results['statistical_report'] = ""
    
    try:
        results['final_report'] = (tables_dir / 'final_evaluation_report.txt').read_text(encoding='utf-8')
        print(f"  加载 final_report: {len(results['final_report'])} 字符")
    except Exception as e:
        print(f"  加载 final_report 失败: {e}")
        results['final_report'] = ""
    
    return results

def generate_model_comparison_table(results):
    """生成模型性能比较表格"""
    if results['final_evaluation'] is None or results['final_evaluation'].empty:
        return r"""
\begin{table}[H]
\caption{模型性能比较}
\label{tab:model_comparison}
\centering
\begin{tabular}{lccccc}
\toprule
Model & Accuracy & Precision & Recall & F1\_Score & AUC\_ROC \\
\midrule
\textbf{数据为空，请先运行前面的分析步骤} & - & - & - & - & - \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    table_content = r"""
\begin{table}[H]
\caption{模型性能比较}
\label{tab:model_comparison}
\centering
\begin{tabular}{lccccc}
\toprule
Model & Accuracy & Precision & Recall & F1\_Score & AUC\_ROC \\
\midrule
"""
    
    for _, row in results['final_evaluation'].iterrows():
        table_content += f"{row['Model']} & {row['Accuracy']:.4f} & {row['Precision']:.4f} & {row['Recall']:.4f} & {row['F1_Score']:.4f} & {row['AUC_ROC']:.4f} \\\\\n"
    
    table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return table_content

def generate_feature_importance_table(results):
    """生成特征重要性表格"""
    if results['feature_importance'] is None or results['feature_importance'].empty:
        return r"""
\begin{table}[H]
\centering
\caption{Top 10特征重要性}
\label{tab:feature_importance}
\begin{tabular}{lr}
\toprule
Feature & Importance \\
\midrule
\textbf{数据为空，请先运行前面的分析步骤} & - \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    table_content = r"""
\begin{table}[H]
\centering
\caption{Top 10特征重要性}
\label{tab:feature_importance}
\begin{tabular}{lr}
\toprule
Feature & Importance \\
\midrule
"""
    
    top_features = results['feature_importance'].head(10)
    for _, row in top_features.iterrows():
        table_content += f"{row['Feature']} & {row['Importance']:.4f} \\\\\n"
    
    table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return table_content

def generate_latex_report(results, figures):
    """生成完整的LaTeX报告"""
    
    # 基本信息
    project_title = "基于机器学习的PICU患者死亡率预测分析"
    student_name = "张芳慧"
    student_id = "2511110136"
    course_name = "Python编程（2025秋）"
    date = "2026年01月07日"
    
    # 最佳模型信息
    if results['final_evaluation'] is not None and not results['final_evaluation'].empty:
        try:
            best_model_row = results['final_evaluation'].loc[results['final_evaluation']['AUC_ROC'].idxmax()]
            best_model_name = best_model_row['Model']
            best_auc = best_model_row['AUC_ROC']
        except Exception as e:
            print(f"  获取最佳模型信息失败: {e}")
            best_model_name = "未确定"
            best_auc = 0.0
    else:
        best_model_name = "未确定"
        best_auc = 0.0
    
    # 获取图表列表
    figure_names = [f.name for f in figures]
    
    # 生成表格内容
    model_comparison_table = generate_model_comparison_table(results)
    feature_importance_table = generate_feature_importance_table(results)
    
    # LaTeX文档内容 - 基于可成功编译的版本
    latex_content = r"""\documentclass[12pt,a4paper]{article}
\usepackage[UTF8]{ctex}
\usepackage{geometry}
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{multirow}
\usepackage{float}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{listings}

% 设置代码样式
\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{red},
    numbers=left,
    numberstyle=\tiny,
    frame=single,
    breaklines=true
}

% 设置目录深度
\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

\begin{document}

% 标题页
\begin{titlepage}
    \centering
    \vspace*{2cm}
    
    {\Huge\bfseries """ + project_title + r"""}\\[2cm]
    
    {\Large 数据分析报告}\\[2cm]
    
    \begin{minipage}{0.8\textwidth}
        \centering
        {\large
        \begin{tabular}{ll}
            \textbf{课程名称:} & """ + course_name + r""" \\
            \textbf{学生姓名:} & """ + student_name + r""" \\
            \textbf{学号:} & """ + student_id + r""" \\
            \textbf{提交日期:} & """ + date + r""" \\
        \end{tabular}
        }
    \end{minipage}
    
    \vfill
    
    {\large \textbf{摘要}}\\[0.5cm]
    
    \begin{minipage}{0.9\textwidth}
        \small
        本报告基于PICU临床公开数据库(icu\_first24hours.csv)，应用数据挖掘和机器学习方法构建住院死亡率预测模型。通过完整的数据分析流程，包括数据预处理、统计分析、模型建立和评估，最终得到了一个具有临床参考价值的预测模型。报告展示了从原始数据处理到模型部署的完整流程，为临床决策提供数据支持。
        
        \textbf{关键词：} PICU，死亡率预测，机器学习，临床数据分析，预测模型
    \end{minipage}
    
    \vspace{1cm}
\end{titlepage}

% 目录页
\newpage
\tableofcontents
\newpage

\section{引言}
\subsection{项目背景}
儿科重症监护室(PICU)是救治危重患儿的重要场所，早期识别高风险患者对于改善预后具有重要意义。通过分析患者入院24小时内的临床数据，建立死亡风险预测模型，可以辅助临床医生进行决策。

\subsection{研究目标}
\begin{itemize}
    \item 完成PICU临床数据的完整分析流程
    \item 建立多个机器学习预测模型
    \item 评估模型性能，选择最佳模型
    \item 生成完整的分析报告和可视化结果
\end{itemize}

\subsection{数据来源}
使用公开的PICU临床数据库(icu\_first24hours.csv)，包含患者入院24小时内的临床特征和住院结局。

\section{数据分析方法}
\subsection{分析流程}
本项目采用以下完整的数据分析流程：
\begin{enumerate}
    \item 数据读取与质量检查
    \item 数据预处理（清洗、特征工程、标准化）
    \item 统计分析（描述性统计、相关性分析）
    \item 预测模型建立（多种机器学习算法）
    \item 模型评估与可视化
    \item 结果总结与报告生成
\end{enumerate}

\subsection{技术线}
\begin{itemize}
    \item \textbf{编程语言:} Python 3.8+
    \item \textbf{数据处理:} Pandas, NumPy
    \item \textbf{机器学习:} Scikit-learn, XGBoost, LightGBM
    \item \textbf{数据可视化:} Matplotlib, Seaborn
    \item \textbf{统计分析:} SciPy
    \item \textbf{报告生成:} LaTeX
\end{itemize}

\section{数据读取与预处理}
\subsection{数据读取}
原始数据集包含1667个特征，通过数据质量检查，发现存在以下问题：
\begin{itemize}
    \item 缺失值：部分特征存在较高的缺失率
    \item 异常值：年龄、体重等特征存在不合理数值
    \item 类别不平衡：死亡病例比例较低
\end{itemize}

\subsection{数据预处理步骤}
\subsubsection{特征选择}
基于缺失率进行特征筛选，删除缺失率超过50\%的特征。

\subsubsection{缺失值处理}
数值特征使用中位数填充，分类特征使用众数填充。

\subsubsection{异常值处理}
\begin{itemize}
    \item 年龄异常：将负值年龄设为0（新生儿）
    \item 体重异常：保留0值，后续通过年龄估算
\end{itemize}

\subsubsection{类别不平衡处理}
使用SMOTE过采样方法，平衡训练集中的类别分布。

\section{统计分析}
\subsection{描述性统计分析}
对处理后的数据进行描述性统计分析，包括均值、标准差、分位数等。
"""
    
    # 添加统计分析图表
    if 'target_distribution.png' in figure_names:
        latex_content += r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/target_distribution.png}
    \caption{目标变量分布}
    \label{fig:target_distribution}
\end{figure}
"""
    
    if 'feature_target_correlation.png' in figure_names:
        latex_content += r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/feature_target_correlation.png}
    \caption{特征与目标变量相关性分析}
    \label{fig:correlation}
\end{figure}
"""
    
    latex_content += r"""
\subsection{相关性分析}
计算各特征与目标变量(HOSPITAL\_EXPIRE\_FLAG)的相关性，识别与死亡率显著相关的特征。

\subsection{统计检验}
使用t检验比较存活组和死亡组在各特征上的差异，识别具有统计学显著差异的特征。

\section{预测模型建立}
\subsection{模型选择}
本项目建立了6种不同的机器学习模型进行对比：
\begin{enumerate}
    \item 逻辑回归(Logistic Regression)
    \item 随机森林(Random Forest)
    \item XGBoost
    \item LightGBM
    \item 梯度提升(Gradient Boosting)
    \item 神经网络(Neural Network)
\end{enumerate}

\subsection{超参数调优}
对关键模型进行超参数调优，使用随机搜索(RandomizedSearchCV)方法。

\subsection{模型性能比较}
所有模型在测试集上的性能比较如下：
""" + model_comparison_table + r"""

\subsection{最佳模型选择}
根据AUC-ROC评分，选择""" + best_model_name + r"""作为最佳模型，其AUC-ROC得分为""" + f"{best_auc:.4f}" + r"""。

\section{模型评估与可视化}
\subsection{ROC曲线分析}
绘制所有模型的ROC曲线，评估模型的区分能力。
"""
    
    if 'roc_curves_all_models.png' in figure_names:
        latex_content += r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/roc_curves_all_models.png}
    \caption{所有模型的ROC曲线}
    \label{fig:roc_curves}
\end{figure}
"""
    
    latex_content += r"""
\subsection{混淆矩阵分析}
分析最佳模型的混淆矩阵，评估其在各类别上的预测性能。
"""
    
    if 'confusion_matrices.png' in figure_names:
        latex_content += r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/confusion_matrices.png}
    \caption{各模型的混淆矩阵}
    \label{fig:confusion_matrix}
\end{figure}
"""
    
    latex_content += r"""
\subsection{精确率-召回率曲线}
分析模型在不同阈值下的精确率和召回率平衡。
"""
    
    if 'precision_recall_curves.png' in figure_names:
        latex_content += r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/precision_recall_curves.png}
    \caption{精确率-召回率曲线}
    \label{fig:pr_curve}
\end{figure}
"""
    
    latex_content += r"""
\subsection{特征重要性分析}
分析最佳模型的特征重要性，识别对预测结果影响最大的特征。
"""
    
    if 'feature_importance.png' in figure_names:
        latex_content += r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/feature_importance.png}
    \caption{最佳模型的特征重要性}
    \label{fig:feature_importance}
\end{figure}
""" + feature_importance_table + r"""

\section{讨论与结论}
\subsection{主要发现}
\begin{itemize}
    \item \textbf{模型性能：} """ + best_model_name + r"""模型在AUC-ROC指标上表现最佳，达到""" + f"{best_auc:.4f}" + r"""。
    \item \textbf{临床意义：} 特征重要性分析揭示了影响PICU患者预后的关键临床指标。
    \item \textbf{局限性：} 由于数据不平衡问题，模型的精确率有待提高。
\end{itemize}

\subsection{临床意义}
本研究发现的关键预测特征可以为临床医生提供以下参考：
\begin{itemize}
    \item 重点关注高风险特征的监测
    \item 早期识别可能需要更多干预的患者
    \item 优化临床资源分配
\end{itemize}

\subsection{改进方向}
\begin{enumerate}
    \item 收集更完整、标准化的临床数据
    \item 尝试更复杂的特征工程方法
    \item 使用集成学习方法提升模型性能
    \item 进行前瞻性研究验证模型效果
\end{enumerate}

\subsection{结论}
本项目完成了从数据预处理到模型部署的完整分析流程，建立了具有临床参考价值的PICU死亡率预测模型。尽管存在数据不平衡等挑战，但研究结果为临床决策支持系统的开发提供了基础。

\section*{致谢}
感谢课程教师提供的指导，以及公开数据提供方PICU数据库。

\newpage
\appendix
\section{附录}
\subsection{代码实现}
本项目所有代码已开源，包含以下模块：
\begin{itemize}
    \item 01\_data\_loading.py - 数据读取模块
    \item 02\_data\_preprocessing.py - 数据预处理模块
    \item 03\_statistical\_analysis.py - 统计分析模块
    \item 04\_model\_building.py - 模型建立模块
    \item 05\_model\_evaluation.py - 模型评估模块
    \item 06\_webpage\_development.py - 网页开发模块
    \item 07\_latex\_report.py - LaTeX报告生成模块
\end{itemize}

\subsection{数据集信息}
原始数据集icu\_first24hours.csv包含以下关键信息：
\begin{itemize}
    \item 患者人口学特征：年龄、性别、体重等
    \item 生命体征：心率、血压、呼吸频率等
    \item 实验室检查：血气分析、血常规、生化指标等
    \item 治疗信息：机械通气参数、药物使用等
    \item 结局指标：HOSPITAL\_EXPIRE\_FLAG（院内死亡标志）
\end{itemize}

\end{document}
"""
    
    return latex_content

def main_latex_report():
    """主LaTeX报告生成流程"""
    print("="*60)
    print("LaTeX报告生成模块")
    print("="*60)
    
    # 1. 准备LaTeX环境
    print("\n步骤1: 准备LaTeX环境...")
    latex_dir = prepare_latex_environment()
    
    # 2. 复制图表
    print("\n步骤2: 复制图表到LaTeX目录...")
    figures = copy_figures_to_latex(latex_dir)
    
    # 3. 加载结果
    print("\n步骤3: 加载分析结果...")
    results = load_results_for_latex()
    
    # 检查是否有模型评估结果
    if results['final_evaluation'] is None or (hasattr(results['final_evaluation'], 'empty') and results['final_evaluation'].empty):
        print("⚠️ 警告：未找到模型评估结果或结果为空，将使用默认表格")
    
    # 4. 生成LaTeX报告
    print("\n步骤4: 生成LaTeX报告...")
    latex_content = generate_latex_report(results, figures)
    
    # 5. 保存.tex文件
    tex_file = "icu_analysis_report.tex"
    tex_path = latex_dir / tex_file
    
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"✅ LaTeX报告已生成: {tex_path}")
    print(f"\n📝 生成的LaTeX文件: {tex_path}")
    print(f"   图表目录: {latex_dir / 'figures'}")
    print(f"\n手动编译方法:")
    print(f"   1. 使用Overleaf在线编译（推荐）")
    print(f"   2. 使用本地LaTeX环境编译:")
    print(f"      cd {latex_dir}")
    print(f"      pdflatex {tex_file}")
    print(f"      pdflatex {tex_file}  # 编译两次以获得正确的目录")
    
    return tex_path

if __name__ == "__main__":
    main_latex_report()