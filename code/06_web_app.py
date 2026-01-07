"""
项目展示网页开发模块
对应作业要求：项目展示网页开发（10分）

主要任务：
1. 创建HTML报告网页
2. 嵌入所有可视化图表和结果表格
3. 设计美观的用户界面
4. 生成完整的项目报告
"""

import pandas as pd
import numpy as np
import base64
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_all_results():
    """加载所有分析结果"""
    project_root = Path(__file__).parent.parent
    tables_dir = project_root / "outputs" / "tables"
    figures_dir = project_root / "outputs" / "figures"
    
    results = {}
    
    # 1. 加载数据统计结果
    try:
        results['statistical_report'] = (tables_dir / 'statistical_report_summary.txt').read_text(encoding='utf-8')
    except:
        results['statistical_report'] = "统计分析报告未找到"
    
    # 2. 加载模型性能比较
    try:
        model_comparison = pd.read_csv(tables_dir / 'model_performance_comparison.csv')
        results['model_comparison'] = model_comparison
    except:
        results['model_comparison'] = None
    
    # 3. 加载最终模型评估结果
    try:
        final_evaluation = pd.read_csv(tables_dir / 'final_model_evaluation.csv')
        results['final_evaluation'] = final_evaluation
    except:
        results['final_evaluation'] = None
    
    # 4. 加载特征重要性
    try:
        feature_importance = pd.read_csv(tables_dir / 'feature_importance_best_model.csv')
        results['feature_importance'] = feature_importance
    except:
        results['feature_importance'] = None
    
    # 5. 读取最终评估报告
    try:
        results['final_report'] = (tables_dir / 'final_evaluation_report.txt').read_text(encoding='utf-8')
    except:
        results['final_report'] = "最终评估报告未找到"
    
    # 6. 检查可视化图表
    results['figures'] = {
        'statistical_analysis': list((figures_dir / 'statistical_analysis').glob('*.png')),
        'model_evaluation': list((figures_dir / 'model_evaluation').glob('*.png'))
    }
    
    # 7. 获取最佳模型信息
    if results['final_evaluation'] is not None:
        best_model = results['final_evaluation'].loc[results['final_evaluation']['AUC_ROC'].idxmax()]
        results['best_model'] = {
            'name': best_model['Model'],
            'accuracy': best_model['Accuracy'],
            'precision': best_model['Precision'],
            'recall': best_model['Recall'],
            'f1_score': best_model['F1_Score'],
            'auc_roc': best_model['AUC_ROC']
        }
    
    return results

def image_to_base64(image_path):
    """将图片转换为base64编码"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return None

def create_html_report(results):
    """创建HTML报告"""
    print("创建HTML报告...")
    
    # 基本信息
    project_name = "PICU死亡率预测分析系统"
    current_date = pd.Timestamp.now().strftime('%Y年%m月%d日')
    
    # HTML模板
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - 分析报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, #1a73e8, #0d47a1);
            color: white;
            padding: 40px 0;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 20px;
        }}
        
        .date {{
            background: rgba(255,255,255,0.1);
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
        }}
        
        .section {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
        }}
        
        .section:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        h2 {{
            color: #1a73e8;
            border-left: 5px solid #1a73e8;
            padding-left: 15px;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            background: linear-gradient(135deg, #e3e6ec, #b3bdd4);
            transform: scale(1.05);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #1a73e8;
            margin: 10px 0;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background-color: #f2f6ff;
            font-weight: 600;
            color: #1a73e8;
        }}
        
        tr:hover {{
            background-color: #f5f7ff;
        }}
        
        .visualization-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .chart-container {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            transition: transform 0.3s ease;
        }}
        
        .chart-container:hover img {{
            transform: scale(1.02);
        }}
        
        .chart-title {{
            text-align: center;
            margin: 10px 0;
            color: #555;
            font-weight: 500;
        }}
        
        .conclusion-box {{
            background: linear-gradient(135deg, #e8f4ff, #d4e7ff);
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #1a73e8;
        }}
        
        .conclusion-title {{
            color: #1a73e8;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }}
        
        footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
            border-top: 1px solid #eee;
        }}
        
        .highlight {{
            color: #1a73e8;
            font-weight: 600;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            background-color: #1a73e8;
            color: white;
            border-radius: 20px;
            font-size: 0.8rem;
            margin: 0 5px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .section {{
                padding: 20px;
            }}
            
            .visualization-grid {{
                grid-template-columns: 1fr;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 页眉 -->
        <header>
            <h1>🏥 {project_name}</h1>
            <div class="subtitle">基于机器学习的重症监护室死亡率预测分析</div>
            <div class="date">📅 报告生成日期：{current_date}</div>
        </header>
        
        <!-- 项目摘要 -->
        <section class="section">
            <h2>📊 项目摘要</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">分析模型数</div>
                    <div class="metric-value">{len(results['model_comparison']) if results['model_comparison'] is not None else 'N/A'}</div>
                    <div class="metric-desc">机器学习模型</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最佳模型</div>
                    <div class="metric-value">{results['best_model']['name'] if 'best_model' in results else 'N/A'}</div>
                    <div class="metric-desc">基于AUC-ROC评分</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最佳AUC-ROC</div>
                    <div class="metric-value">{results['best_model']['auc_roc']:.4f if 'best_model' in results else 'N/A'}</div>
                    <div class="metric-desc">模型区分能力</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">生成图表数</div>
                    <div class="metric-value">{len(results['figures']['model_evaluation']) + len(results['figures']['statistical_analysis'])}</div>
                    <div class="metric-desc">数据可视化</div>
                </div>
            </div>
            
            <div class="conclusion-box">
                <div class="conclusion-title">🎯 核心发现</div>
                <p>本项目通过对ICU患者入院前24小时数据的分析，建立了多个机器学习模型来预测院内死亡率。</p>
                <p>虽然模型在准确率方面表现良好，但在精确率和召回率的平衡上仍有改进空间，这反映了临床数据中常见的类别不平衡问题。</p>
            </div>
        </section>
        
        <!-- 数据统计分析 -->
        <section class="section">
            <h2>📈 数据统计分析</h2>
            <div class="table-container">
                <p>原始数据集经过清洗、特征工程和标准化处理后，进行了全面的统计分析。主要步骤包括：</p>
                <ul>
                    <li><span class="highlight">数据清洗</span>：处理缺失值、异常值，删除无效特征</li>
                    <li><span class="highlight">特征工程</span>：从时间数据提取特征，处理类别不平衡</li>
                    <li><span class="highlight">统计分析</span>：描述性统计、相关性分析、统计检验</li>
                    <li><span class="highlight">特征选择</span>：基于缺失率和重要性的特征筛选</li>
                </ul>
            </div>
            
            <!-- 统计分析图表 -->
            <div class="visualization-grid">
    """
    
    # 添加统计分析图表
    statistical_figures = results['figures']['statistical_analysis']
    for i, fig_path in enumerate(statistical_figures[:4]):  # 只显示前4个
        img_base64 = image_to_base64(fig_path)
        if img_base64:
            fig_name = fig_path.stem.replace('_', ' ').title()
            html_content += f"""
                <div class="chart-container">
                    <img src="data:image/png;base64,{img_base64}" alt="{fig_name}">
                    <div class="chart-title">{fig_name}</div>
                </div>
            """
    
    html_content += """
            </div>
        </section>
        
        <!-- 模型性能比较 -->
        <section class="section">
            <h2>🤖 模型性能比较</h2>
            <p>本项目训练了6种不同的机器学习模型，下表展示了它们在测试集上的性能表现：</p>
            
            <div class="table-container">
    """
    
    # 添加模型性能表格
    if results['final_evaluation'] is not None:
        df = results['final_evaluation'][['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']]
        html_content += df.to_html(classes='performance-table', index=False, float_format='{:.4f}'.format)
    
    html_content += """
            </div>
            
            <!-- 模型评估图表 -->
            <div class="visualization-grid">
    """
    
    # 添加模型评估图表
    model_figures = results['figures']['model_evaluation']
    important_figures = ['roc_curves_all_models', 'confusion_matrices', 'model_performance_comparison', 'feature_importance']
    
    for fig_name in important_figures:
        for fig_path in model_figures:
            if fig_name in str(fig_path):
                img_base64 = image_to_base64(fig_path)
                if img_base64:
                    display_name = fig_name.replace('_', ' ').title()
                    html_content += f"""
                        <div class="chart-container">
                            <img src="data:image/png;base64,{img_base64}" alt="{display_name}">
                            <div class="chart-title">{display_name}</div>
                        </div>
                    """
                    break
    
    html_content += """
            </div>
            
            <!-- 最佳模型性能指标 -->
            <div class="metrics-grid">
    """
    
    # 添加最佳模型性能指标
    if 'best_model' in results:
        best = results['best_model']
        metrics = [
            ('准确率', best['accuracy'], '模型整体正确预测的比例'),
            ('精确率', best['precision'], '阳性预测的可靠性'),
            ('召回率', best['recall'], '识别真实阳性病例的能力'),
            ('F1分数', best['f1_score'], '精确率和召回率的调和平均')
        ]
        
        for label, value, desc in metrics:
            html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-desc">{desc}</div>
                </div>
            """
    
    html_content += """
            </div>
        </section>
        
        <!-- 特征重要性分析 -->
        <section class="section">
            <h2>🔍 特征重要性分析</h2>
            <p>以下是预测模型认为最重要的特征，这些特征对于识别高风险患者具有重要意义：</p>
            
            <div class="table-container">
    """
    
    # 添加特征重要性表格
    if results['feature_importance'] is not None:
        df_top10 = results['feature_importance'].head(10)
        html_content += df_top10.to_html(classes='feature-table', index=False, float_format='{:.6f}'.format)
    
    html_content += """
            </div>
            
            <div class="conclusion-box">
                <div class="conclusion-title">📋 临床意义</div>
                <p>特征重要性分析揭示了影响ICU患者预后的关键因素。排名靠前的特征可能包括：</p>
                <ul>
                    <li><span class="highlight">生命体征</span>：血压、心率、呼吸频率等</li>
                    <li><span class="highlight">实验室指标</span>：血气分析、血常规、生化指标等</li>
                    <li><span class="highlight">患者特征</span>：年龄、体重、合并症等</li>
                    <li><span class="highlight">治疗参数</span>：机械通气参数、药物剂量等</li>
                </ul>
                <p>这些发现有助于临床医生重点关注高风险患者的监测和管理。</p>
            </div>
        </section>
        
        <!-- 项目结论与建议 -->
        <section class="section">
            <h2>🎯 项目结论与建议</h2>
            
            <div class="conclusion-box">
                <div class="conclusion-title">✅ 项目成果</div>
                <p>1. <span class="highlight">完整的分析流程</span>：实现了从数据清洗到模型部署的完整机器学习流程</p>
                <p>2. <span class="highlight">多种模型对比</span>：评估了6种不同机器学习算法的性能</p>
                <p>3. <span class="highlight">全面的可视化</span>：生成了丰富的统计图表和模型评估图表</p>
                <p>4. <span class="highlight">实用的分析工具</span>：为ICU临床决策提供了数据支持</p>
            </div>
            
            <div class="conclusion-box">
                <div class="conclusion-title">💡 改进建议</div>
                <p>1. <span class="highlight">数据质量提升</span>：收集更完整、更标准化的临床数据</p>
                <p>2. <span class="highlight">特征工程优化</span>：考虑更多临床相关的衍生特征</p>
                <p>3. <span class="highlight">模型集成</span>：尝试模型融合或集成学习方法</p>
                <p>4. <span class="highlight">实时预测</span>：开发实时预测系统，动态评估患者风险</p>
            </div>
            
            <div class="conclusion-box">
                <div class="conclusion-title">🚀 下一步计划</div>
                <p>1. 部署预测模型到临床信息系统</p>
                <p>2. 开发用户友好的临床决策支持界面</p>
                <p>3. 进行前瞻性研究验证模型效果</p>
                <p>4. 扩展应用到其他疾病预测场景</p>
            </div>
        </section>
        
        <!-- 技术栈 -->
        <section class="section">
            <h2>🛠️ 技术栈</h2>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
                <span class="badge">Python 3.8+</span>
                <span class="badge">Pandas</span>
                <span class="badge">NumPy</span>
                <span class="badge">Scikit-learn</span>
                <span class="badge">XGBoost</span>
                <span class="badge">LightGBM</span>
                <span class="badge">Matplotlib</span>
                <span class="badge">Seaborn</span>
                <span class="badge">Imbalanced-learn</span>
                <span class="badge">HTML/CSS</span>
            </div>
        </section>
        
        <!-- 页脚 -->
        <footer>
            <p>© 2024 PICU死亡率预测分析系统 - 医学数据分析项目</p>
            <p>本报告为学术研究用途，临床决策请结合专业医生判断</p>
            <p>项目代码：<a href="https://github.com/username/icu-mortality-prediction" target="_blank">GitHub Repository</a></p>
        </footer>
    </div>
    
    <script>
        // 简单的交互效果
        document.addEventListener('DOMContentLoaded', function() {{
            // 为表格行添加悬停效果
            const tableRows = document.querySelectorAll('tbody tr');
            tableRows.forEach(row => {{
                row.addEventListener('mouseenter', function() {{
                    this.style.backgroundColor = '#f0f5ff';
                }});
                row.addEventListener('mouseleave', function() {{
                    this.style.backgroundColor = '';
                }});
            }});
            
            // 平滑滚动到章节
            document.querySelectorAll('nav a').forEach(anchor => {{
                anchor.addEventListener('click', function(e) {{
                    e.preventDefault();
                    const targetId = this.getAttribute('href').substring(1);
                    const targetElement = document.getElementById(targetId);
                    if (targetElement) {{
                        window.scrollTo({{
                            top: targetElement.offsetTop - 20,
                            behavior: 'smooth'
                        }});
                    }}
                }});
            }});
            
            // 添加打印功能
            const printButton = document.createElement('button');
            printButton.textContent = '🖨️ 打印报告';
            printButton.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #1a73e8;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 25px;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                z-index: 1000;
                font-weight: bold;
            `;
            printButton.addEventListener('click', () => window.print());
            document.body.appendChild(printButton);
        }});
    </script>
</body>
</html>
    """
    
    return html_content

def create_simple_html_report(results):
    """创建简化版HTML报告（如果base64编码有问题）"""
    print("创建简化版HTML报告...")
    
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "outputs" / "figures"
    
    # 获取相对路径
    def get_relative_path(fig_path):
        try:
            return fig_path.relative_to(project_root).as_posix()
        except:
            return fig_path.name
    
    project_name = "PICU死亡率预测分析系统"
    current_date = pd.Timestamp.now().strftime('%Y年%m月%d日')
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - 分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background: #f4f4f4; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #1a73e8; }}
        h1 {{ color: #1a73e8; margin-bottom: 10px; }}
        h2 {{ color: #333; margin: 30px 0 15px 0; padding-bottom: 10px; border-bottom: 1px solid #ddd; }}
        .section {{ margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .figure-item {{ text-align: center; }}
        .figure-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .metrics {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 15px; min-width: 150px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1a73e8; }}
        .highlight {{ background-color: #e8f4ff; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏥 {project_name}</h1>
            <p>基于机器学习的重症监护室死亡率预测分析</p>
            <p><strong>报告生成日期：</strong>{current_date}</p>
        </header>
        
        <div class="section">
            <h2>📊 项目摘要</h2>
            <div class="metrics">
    """
    
    # 添加摘要指标
    if 'best_model' in results:
        best = results['best_model']
        html_content += f"""
                <div class="metric">
                    <div class="metric-value">{best['name']}</div>
                    <div>最佳模型</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{best['auc_roc']:.4f}</div>
                    <div>AUC-ROC</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{best['f1_score']:.4f}</div>
                    <div>F1分数</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(results['model_comparison']) if results['model_comparison'] is not None else 0}</div>
                    <div>模型数量</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>🤖 模型性能比较</h2>
    """
    
    # 添加模型性能表格
    if results['final_evaluation'] is not None:
        df = results['final_evaluation'][['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']]
        html_content += df.to_html(index=False, float_format='{:.4f}'.format, classes='model-table')
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>📈 可视化图表</h2>
            <div class="figure-grid">
    """
    
    # 添加主要图表
    important_charts = [
        ('ROC曲线', 'roc_curves_all_models'),
        ('混淆矩阵', 'confusion_matrices'),
        ('模型比较', 'model_performance_comparison'),
        ('特征重要性', 'feature_importance')
    ]
    
    for chart_name, chart_file in important_charts:
        for fig_path in results['figures']['model_evaluation']:
            if chart_file in str(fig_path):
                rel_path = get_relative_path(fig_path)
                html_content += f"""
                    <div class="figure-item">
                        <h3>{chart_name}</h3>
                        <img src="{rel_path}" alt="{chart_name}">
                    </div>
                """
                break
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 特征重要性</h2>
    """
    
    # 添加特征重要性表格
    if results['feature_importance'] is not None:
        df_top10 = results['feature_importance'].head(10)
        html_content += df_top10.to_html(index=False, float_format='{:.6f}'.format, classes='feature-table')
    
    html_content += """
            <div class="highlight">
                <p><strong>分析结论：</strong>特征重要性分析揭示了影响ICU患者预后的关键临床指标。这些发现有助于临床医生重点关注高风险患者的监测和干预。</p>
            </div>
        </div>
        
        <footer>
            <p>© 2024 PICU死亡率预测分析系统 | 医学数据分析项目</p>
            <p>本报告为学术研究用途 | 临床决策请结合专业医生判断</p>
        </footer>
    </div>
</body>
</html>
    """
    
    return html_content

def main_webpage_development():
    """
    主网页开发流程
    """
    print("="*60)
    print("项目展示网页开发模块")
    print("="*60)
    
    # 1. 加载所有结果
    print("\n步骤1: 加载所有分析结果...")
    results = load_all_results()
    
    if not results:
        print("未找到任何分析结果，请先运行前面的分析模块")
        return
    
    # 2. 创建HTML报告
    print("\n步骤2: 创建HTML报告...")
    
    # 尝试创建完整版报告
    try:
        html_content = create_html_report(results)
        report_type = "完整版"
    except Exception as e:
        print(f"创建完整版报告失败，尝试简化版: {e}")
        html_content = create_simple_html_report(results)
        report_type = "简化版"
    
    # 3. 保存HTML文件
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / "outputs" / "webpage"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    html_path = outputs_dir / "project_dashboard.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 4. 复制图表到网页目录（可选）
    print("\n步骤3: 准备网页资源...")
    
    # 复制图表文件
    figures_dir = project_root / "outputs" / "figures"
    webpage_figures_dir = outputs_dir / "figures"
    webpage_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制统计分析和模型评估图表
    for subdir in ['statistical_analysis', 'model_evaluation']:
        source_dir = figures_dir / subdir
        if source_dir.exists():
            target_dir = webpage_figures_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for fig_file in source_dir.glob('*.png'):
                try:
                    import shutil
                    shutil.copy2(fig_file, target_dir / fig_file.name)
                    print(f"  复制图表: {fig_file.name}")
                except:
                    pass
    
    # 5. 生成完成报告
    print("\n" + "="*60)
    print("网页开发完成！")
    print("="*60)
    
    # 获取文件路径（相对路径）
    relative_html_path = html_path.relative_to(project_root).as_posix()
    
    print(f"\n✅ {report_type}HTML报告已生成:")
    print(f"   文件位置: {relative_html_path}")
    
    print(f"\n📊 报告内容包含:")
    print(f"   1. 项目摘要和关键指标")
    print(f"   2. 数据统计分析结果")
    print(f"   3. 模型性能比较表格")
    print(f"   4. 可视化图表展示")
    print(f"   5. 特征重要性分析")
    print(f"   6. 项目结论与建议")
    
    print(f"\n🌐 查看报告:")
    print(f"   1. 使用浏览器打开: {html_path}")
    print(f"   2. 或双击HTML文件直接打开")
    
    print(f"\n📁 相关文件位置:")
    print(f"   - HTML报告: outputs/webpage/project_dashboard.html")
    print(f"   - 图表文件: outputs/webpage/figures/")
    print(f"   - 原始数据: data/")
    print(f"   - 分析结果: outputs/tables/")
    print(f"   - 训练模型: outputs/models/")
    
    print(f"\n🚀 快速开始:")
    print(f"   直接打开 '{html_path}' 查看完整分析报告!")
    
    return html_path

# 主程序入口
if __name__ == "__main__":
    html_path = main_webpage_development()