"""
预测模型评估与可视化模块
对应作业要求：预测模型评估与可视化（20分）

主要任务：
1. 加载训练好的模型
2. 综合评估所有模型
3. 创建可视化图表
4. 生成模型评估报告
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report
)
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_models_and_data():
    """加载模型和测试数据"""
    project_root = Path(__file__).parent.parent
    
    # 加载测试数据
    processed_dir = project_root / "data" / "processed"
    test_path = processed_dir / "test_data.csv"
    feature_path = processed_dir / "feature_list.csv"
    
    if not test_path.exists():
        print("Please run 02_data_preprocessing.py first")
        return None, None, None, None
    
    test_df = pd.read_csv(test_path)
    feature_df = pd.read_csv(feature_path)
    feature_cols = feature_df['feature'].tolist()
    
    X_test = test_df[feature_cols]
    y_test = test_df['HOSPITAL_EXPIRE_FLAG']
    
    # 加载标准化器
    models_dir = project_root / "outputs" / "models"
    scaler_path = models_dir / "scaler.pkl"
    
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values
    
    # 加载所有模型
    model_files = {
        'Logistic_Regression': 'logistic_regression.pkl',
        'Random_Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'LightGBM': 'lightgbm.pkl',
        'Gradient_Boosting': 'gradient_boosting.pkl',
        'Neural_Network': 'neural_network.pkl'
    }
    
    models = {}
    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
    
    print(f"\nLoaded {len(models)} models")
    print(f"Test data shape: {X_test_scaled.shape}")
    
    return models, X_test_scaled, y_test, feature_cols

def evaluate_all_models(models, X_test, y_test):
    """评估所有模型"""
    print("\n" + "="*60)
    print("Comprehensive Model Evaluation")
    print("="*60)
    
    evaluation_results = []
    predictions = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        try:
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 计算指标
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, 
                f1_score, roc_auc_score, confusion_matrix
            )
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # 保存结果
            evaluation_results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'AUC_ROC': auc_score,
                'True_Negative': tn,
                'False_Positive': fp,
                'False_Negative': fn,
                'True_Positive': tp
            })
            
            predictions[model_name] = {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC-ROC: {auc_score:.4f}")
            
        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(evaluation_results)
    
    # 保存结果
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / "outputs" / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(outputs_dir / "final_model_evaluation.csv", index=False)
    
    print("\n" + "="*60)
    print("Model Performance Summary:")
    print("="*60)
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']])
    
    return results_df, predictions

def plot_roc_curves(models, predictions, X_test, y_test):
    """绘制ROC曲线"""
    print("\nPlotting ROC curves...")
    
    # 创建输出目录
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "outputs" / "figures" / "model_evaluation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制每个模型的ROC曲线
    for model_name, model in models.items():
        if model_name in predictions:
            y_pred_proba = predictions[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for All Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'roc_curves_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to: {figures_dir / 'roc_curves_all_models.png'}")

def plot_precision_recall_curves(predictions, y_test):
    """绘制精确率-召回率曲线"""
    print("Plotting Precision-Recall curves...")
    
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "outputs" / "figures" / "model_evaluation"
    
    plt.figure(figsize=(10, 8))
    
    for model_name, pred_data in predictions.items():
        y_pred_proba = pred_data['y_pred_proba']
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        # 计算平均精确率
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
    
    # 计算随机模型的基准
    random_precision = len(y_test[y_test==1]) / len(y_test)
    plt.axhline(y=random_precision, color='navy', linestyle='--', label=f'Random (AP = {random_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves for All Models', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Precision-Recall curves saved to: {figures_dir / 'precision_recall_curves.png'}")

def plot_model_comparison(results_df):
    """绘制模型比较图"""
    print("Plotting model comparison charts...")
    
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "outputs" / "figures" / "model_evaluation"
    
    # 1. 模型性能比较条形图
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if idx < len(axes):
            ax = axes[idx]
            sorted_df = results_df.sort_values(metric, ascending=False)
            bars = ax.barh(sorted_df['Model'], sorted_df[metric], color='steelblue', alpha=0.8)
            
            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            ax.set_xlabel(metric, fontsize=11)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
    
    # 如果有空余的子图位置，添加混淆矩阵热力图
    if len(metrics) < len(axes):
        ax = axes[len(metrics)]
        
        # 获取最佳模型（基于AUC）
        best_model_row = results_df.loc[results_df['AUC_ROC'].idxmax()]
        model_name = best_model_row['Model']
        
        # 这里简化处理，实际应该从predictions中获取
        ax.text(0.5, 0.5, f'Best Model:\n{model_name}\nAUC: {best_model_row["AUC_ROC"]:.3f}',
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Best Model Info', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 模型性能雷达图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # 准备数据
    metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']
    num_vars = len(metrics_for_radar)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 绘制每个模型
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        values = [row[metric] for metric in metrics_for_radar]
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_for_radar, fontsize=10)
    
    # 设置y轴标签
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'model_performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison charts saved to: {figures_dir}")

def plot_confusion_matrices(models, predictions):
    """绘制混淆矩阵"""
    print("Plotting confusion matrices...")
    
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "outputs" / "figures" / "model_evaluation"
    
    num_models = len(models)
    n_cols = 3
    n_rows = (num_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (model_name, pred_data) in enumerate(predictions.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[col] if n_rows == 1 else axes[row]
        
        y_true = pred_data['y_true']
        y_pred = pred_data['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar_kws={"shrink": 0.8}, annot_kws={"size": 12})
        
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        
        # 设置刻度标签
        ax.set_xticklabels(['Alive (0)', 'Death (1)'], fontsize=9)
        ax.set_yticklabels(['Alive (0)', 'Death (1)'], fontsize=9, rotation=0)
    
    # 隐藏多余的子图
    for idx in range(len(models), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[col] if n_rows == 1 else axes[row]
        ax.axis('off')
    
    plt.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices saved to: {figures_dir / 'confusion_matrices.png'}")

def plot_feature_importance(best_model, feature_cols):
    """绘制特征重要性"""
    print("Plotting feature importance for best model...")
    
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "outputs" / "figures" / "model_evaluation"
    
    # 检查模型类型并获取特征重要性
    model_type = type(best_model).__name__
    
    if hasattr(best_model, 'feature_importances_'):
        # 树模型
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        # 线性模型
        importances = np.abs(best_model.coef_[0])
    else:
        print(f"Cannot extract feature importance for {model_type}")
        return
    
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    # 绘制水平条形图
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
    
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 20 Feature Importance - {model_type}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # 最重要的特征在顶部
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存特征重要性到CSV
    outputs_dir = project_root / "outputs" / "tables"
    feature_importance_df.to_csv(outputs_dir / 'feature_importance_best_model.csv', index=False)
    
    print(f"Feature importance saved to: {figures_dir / 'feature_importance.png'}")
    print(f"Feature importance data saved to: {outputs_dir / 'feature_importance_best_model.csv'}")

def generate_final_report(results_df, best_model_name):
    """生成最终评估报告"""
    print("\n" + "="*60)
    print("Generating Final Evaluation Report")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / "outputs" / "tables"
    
    # 获取最佳模型信息
    best_model_row = results_df[results_df['Model'] == best_model_name].iloc[0]
    
    # 创建报告内容
    report_content = f"""
ICU Mortality Prediction - Final Model Evaluation Report
========================================================

Summary
-------
Total Models Evaluated: {len(results_df)}
Best Model: {best_model_name}
Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Best Model Performance
----------------------
Accuracy:  {best_model_row['Accuracy']:.4f}
Precision: {best_model_row['Precision']:.4f}
Recall:    {best_model_row['Recall']:.4f}
F1-Score:  {best_model_row['F1_Score']:.4f}
AUC-ROC:   {best_model_row['AUC_ROC']:.4f}

Confusion Matrix (Best Model):
True Negative:  {best_model_row['True_Negative']}
False Positive: {best_model_row['False_Positive']}
False Negative: {best_model_row['False_Negative']}
True Positive:  {best_model_row['True_Positive']}

All Models Performance Summary
------------------------------
{results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']].to_string(index=False)}

Key Findings
------------
1. The {best_model_name} model achieved the highest AUC-ROC score of {best_model_row['AUC_ROC']:.4f}.
2. The model shows {'good' if best_model_row['AUC_ROC'] > 0.8 else 'moderate' if best_model_row['AUC_ROC'] > 0.7 else 'weak'} discriminatory power.
3. Precision of {best_model_row['Precision']:.4f} indicates {'high' if best_model_row['Precision'] > 0.8 else 'moderate' if best_model_row['Precision'] > 0.6 else 'low'} reliability of positive predictions.
4. Recall of {best_model_row['Recall']:.4f} shows {'good' if best_model_row['Recall'] > 0.7 else 'moderate' if best_model_row['Recall'] > 0.5 else 'poor'} ability to identify actual positive cases.

Recommendations
---------------
1. Consider using {best_model_name} for clinical decision support.
2. {'The model shows promising performance and could be valuable for early identification of high-risk patients.' if best_model_row['AUC_ROC'] > 0.7 else 'Further feature engineering or model tuning may improve performance.'}
3. Regular model retraining with new data is recommended.

Visualizations Generated
------------------------
All visualizations have been saved to: outputs/figures/model_evaluation/

Data Files
----------
1. Model evaluation results: outputs/tables/final_model_evaluation.csv
2. Feature importance: outputs/tables/feature_importance_best_model.csv
3. Individual model files: outputs/models/
"""
    
    # 保存报告
    report_path = outputs_dir / 'final_evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Final report saved to: {report_path}")
    
    # 打印报告摘要
    print("\n" + "="*60)
    print("REPORT SUMMARY")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Best AUC-ROC: {best_model_row['AUC_ROC']:.4f}")
    print(f"Best F1-Score: {best_model_row['F1_Score']:.4f}")
    print(f"Total Visualizations: 6+ charts")
    print("="*60)

def main_model_evaluation():
    """
    主模型评估流程
    """
    print("="*60)
    print("ICU Data Analysis - Model Evaluation Module")
    print("="*60)
    
    # 1. 加载模型和数据
    print("\nStep 1: Loading models and test data...")
    models, X_test, y_test, feature_cols = load_models_and_data()
    
    if not models:
        print("No models found. Please run 04_model_building.py first.")
        return
    
    # 2. 评估所有模型
    print("\nStep 2: Evaluating all models...")
    results_df, predictions = evaluate_all_models(models, X_test, y_test)
    
    # 3. 绘制ROC曲线
    print("\nStep 3: Creating ROC curves...")
    plot_roc_curves(models, predictions, X_test, y_test)
    
    # 4. 绘制精确率-召回率曲线
    print("\nStep 4: Creating Precision-Recall curves...")
    plot_precision_recall_curves(predictions, y_test)
    
    # 5. 绘制模型比较图
    print("\nStep 5: Creating model comparison charts...")
    plot_model_comparison(results_df)
    
    # 6. 绘制混淆矩阵
    print("\nStep 6: Creating confusion matrices...")
    plot_confusion_matrices(models, predictions)
    
    # 7. 绘制特征重要性
    print("\nStep 7: Creating feature importance chart...")
    # 确定最佳模型
    best_model_name = results_df.loc[results_df['AUC_ROC'].idxmax(), 'Model']
    best_model = models[best_model_name]
    plot_feature_importance(best_model, feature_cols)
    
    # 8. 生成最终报告
    print("\nStep 8: Generating final evaluation report...")
    generate_final_report(results_df, best_model_name)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nAll results have been saved to:")
    print("  - outputs/tables/ (tables and reports)")
    print("  - outputs/figures/model_evaluation/ (visualizations)")
    print("  - outputs/models/ (trained models)")
    
    return results_df, best_model_name

if __name__ == "__main__":
    results_df, best_model_name = main_model_evaluation()