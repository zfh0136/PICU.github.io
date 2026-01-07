

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import f_classif, chi2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_processed_data():
    """加载预处理后的数据"""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    
    train_path = processed_dir / "train_data.csv"
    test_path = processed_dir / "test_data.csv"
    feature_path = processed_dir / "feature_list.csv"
    
    if not (train_path.exists() and test_path.exists()):
        print("请先运行02_data_preprocessing.py")
        return None, None, None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_df = pd.read_csv(feature_path)
    feature_cols = feature_df['feature'].tolist()
    
    print(f"训练集: {train_df.shape}")
    print(f"测试集: {test_df.shape}")
    print(f"特征数: {len(feature_cols)}")
    
    return train_df, test_df, feature_cols

def descriptive_statistics(train_df, feature_cols):
    """
    描述性统计分析
    """
    print("\n" + "="*60)
    print(" 描述性统计分析")
    print("="*60)
    
    # 分离特征和目标
    X = train_df[feature_cols]
    y = train_df['HOSPITAL_EXPIRE_FLAG']
    
    # 1. 整体描述性统计
    print("\n整体描述性统计:")
    print(X.describe())
    
    # 2. 按目标变量分组统计
    print("\n按目标变量分组的描述性统计:")
    
    # 按目标变量分组
    X_0 = X[y == 0]  # 存活组
    X_1 = X[y == 1]  # 死亡组
    
    # 计算每组的均值和标准差
    stats_summary = pd.DataFrame({
        '特征': feature_cols[:20],  # 只显示前20个
        '总体均值': X.mean().values[:20],
        '总体标准差': X.std().values[:20],
        '存活组均值': X_0.mean().values[:20],
        '死亡组均值': X_1.mean().values[:20],
        '均值差异': (X_1.mean() - X_0.mean()).values[:20]
    })
    
    print(stats_summary.to_string())
    
    # 3. 保存描述性统计结果
    from pathlib import Path
    outputs_dir = Path(__file__).parent.parent / "outputs" / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整的描述性统计
    X.describe().to_csv(outputs_dir / "descriptive_statistics_full.csv")
    
    # 保存分组统计
    stats_summary.to_csv(outputs_dir / "group_statistics.csv", index=False)
    
    print(f"\n 描述性统计结果已保存到: {outputs_dir}")
    
    return X_0, X_1

def correlation_analysis(X, y, feature_cols):
    """
    相关性分析
    """
    print("\n" + "="*60)
    print(" 相关性分析")
    print("="*60)
    
    # 1. 计算特征与目标变量的相关性
    correlations = []
    for col in feature_cols:
        if X[col].nunique() > 1:  # 避免常数列
            corr = X[col].corr(y)
            correlations.append({
                '特征': col,
                '与目标相关性': corr,
                '相关性绝对值': abs(corr)
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('相关性绝对值', ascending=False)
    
    print(f"\n 与目标变量相关性最高的10个特征:")
    print(corr_df.head(10).to_string())
    
    print(f"\n 与目标变量相关性最低的10个特征:")
    print(corr_df.tail(10).to_string())
    
    # 2. 特征间的相关性矩阵（只计算前30个特征避免内存问题）
    if len(feature_cols) > 30:
        top_features = corr_df.head(30)['特征'].tolist()
        corr_matrix = X[top_features].corr()
        
        print(f"\n 前30个重要特征的相关性矩阵摘要:")
        print("   平均相关性:", corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean())
        print("   强相关特征对（|r| > 0.8）:")
        
        # 找出强相关的特征对
        strong_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    strong_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        if strong_corr_pairs:
            for pair in strong_corr_pairs[:5]:  # 只显示前5对
                print(f"     {pair[0]} - {pair[1]}: {pair[2]:.3f}")
        else:
            print("     无强相关特征对")
    
    # 3. 保存相关性结果
    outputs_dir = Path(__file__).parent.parent / "outputs" / "tables"
    corr_df.to_csv(outputs_dir / "feature_target_correlations.csv", index=False)
    
    print(f"\n相关性分析结果已保存")
    
    return corr_df

def statistical_tests(X_0, X_1, feature_cols):
    """
    统计检验：比较两组之间的差异
    """
    print("\n" + "="*60)
    print(" 统计检验（存活组 vs 死亡组）")
    print("="*60)
    
    # 进行t检验
    t_test_results = []
    
    for col in feature_cols[:50]:  # 只测试前50个特征避免计算量过大
        # 检查方差齐性
        levene_test = stats.levene(X_0[col].dropna(), X_1[col].dropna())
        
        # 根据方差齐性选择t检验类型
        if levene_test.pvalue > 0.05:
            # 方差齐，使用标准t检验
            t_stat, p_value = stats.ttest_ind(X_0[col].dropna(), X_1[col].dropna(), equal_var=True)
        else:
            # 方差不齐，使用Welch's t检验
            t_stat, p_value = stats.ttest_ind(X_0[col].dropna(), X_1[col].dropna(), equal_var=False)
        
        # 计算效应量（Cohen's d）
        mean_diff = X_1[col].mean() - X_0[col].mean()
        pooled_std = np.sqrt((X_0[col].std()**2 + X_1[col].std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
        
        t_test_results.append({
            '特征': col,
            't统计量': t_stat,
            'p值': p_value,
            '效应量(Cohen\'s d)': cohens_d,
            '显著(p<0.05)': p_value < 0.05,
            '非常显著(p<0.01)': p_value < 0.01,
            '极显著(p<0.001)': p_value < 0.001
        })
    
    t_test_df = pd.DataFrame(t_test_results)
    t_test_df = t_test_df.sort_values('p值')
    
    print(f"\n 统计检验结果（前10个最显著的特征）:")
    print(t_test_df.head(10).to_string())
    
    print(f"\n 统计显著性汇总:")
    print(f"   显著特征数(p<0.05): {t_test_df['显著(p<0.05)'].sum()}")
    print(f"   非常显著特征数(p<0.01): {t_test_df['非常显著(p<0.01)'].sum()}")
    print(f"   极显著特征数(p<0.001): {t_test_df['极显著(p<0.001)'].sum()}")
    
    # 保存统计检验结果
    outputs_dir = Path(__file__).parent.parent / "outputs" / "tables"
    t_test_df.to_csv(outputs_dir / "statistical_tests_results.csv", index=False)
    
    print(f"\n 统计检验结果已保存")
    
    return t_test_df

def feature_importance_analysis(X, y, feature_cols):
    """
    特征重要性分析（使用多种方法）
    """
    print("\n" + "="*60)
    print(" 特征重要性分析")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # 1. 使用方差分析（ANOVA）F值
    print("\n 方差分析（ANOVA）特征重要性:")
    
    # 选择前100个特征进行ANOVA（避免内存问题）
    if len(feature_cols) > 100:
        selected_features = feature_cols[:100]
    else:
        selected_features = feature_cols
    
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X[selected_features], y)
    
    anova_scores = pd.DataFrame({
        '特征': selected_features,
        'F值': selector.scores_,
        'p值': selector.pvalues_
    })
    anova_scores = anova_scores.sort_values('F值', ascending=False)
    
    print("   基于F值的前10个重要特征:")
    print(anova_scores.head(10).to_string())
    
    # 2. 使用随机森林特征重要性
    print("\n 随机森林特征重要性:")
    
    # 为了速度，使用较小的随机森林
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X[feature_cols], y)
    
    rf_importance = pd.DataFrame({
        '特征': feature_cols,
        '重要性': rf.feature_importances_
    })
    rf_importance = rf_importance.sort_values('重要性', ascending=False)
    
    print("   基于随机森林的前10个重要特征:")
    print(rf_importance.head(10).to_string())
    
    # 3. 合并多种重要性指标
    print("\n 综合特征重要性排名:")
    
    # 合并ANOVA和随机森林的结果
    combined_importance = pd.DataFrame({'特征': feature_cols})
    
    # 归一化ANOVA F值
    anova_scores_full = pd.DataFrame({'特征': feature_cols})
    anova_scores_full = anova_scores_full.merge(anova_scores[['特征', 'F值']], on='特征', how='left')
    anova_scores_full['F值'] = anova_scores_full['F值'].fillna(0)
    anova_scores_full['F值_归一化'] = anova_scores_full['F值'] / anova_scores_full['F值'].max()
    
    # 归一化随机森林重要性
    rf_importance['RF_归一化'] = rf_importance['重要性'] / rf_importance['重要性'].max()
    
    # 合并
    combined_importance = combined_importance.merge(
        anova_scores_full[['特征', 'F值_归一化']], on='特征', how='left'
    )
    combined_importance = combined_importance.merge(
        rf_importance[['特征', 'RF_归一化']], on='特征', how='left'
    )
    
    # 计算综合得分
    combined_importance['综合得分'] = (
        combined_importance['F值_归一化'] * 0.5 + 
        combined_importance['RF_归一化'] * 0.5
    )
    combined_importance = combined_importance.sort_values('综合得分', ascending=False)
    
    print("   综合排名前10的特征:")
    print(combined_importance.head(10).to_string())
    
    # 保存特征重要性结果
    outputs_dir = Path(__file__).parent.parent / "outputs" / "tables"
    combined_importance.to_csv(outputs_dir / "feature_importance_combined.csv", index=False)
    anova_scores.to_csv(outputs_dir / "feature_importance_anova.csv", index=False)
    rf_importance.to_csv(outputs_dir / "feature_importance_rf.csv", index=False)
    
    print(f"\n 特征重要性分析结果已保存")
    
    return combined_importance

def create_visualizations(train_df, feature_cols, corr_df, t_test_df, combined_importance):
    """
    创建统计可视化图表
    """
    print("\n" + "="*60)
    print(" 创建可视化图表")
    print("="*60)
    
    from pathlib import Path
    
    # 创建输出目录
    figures_dir = Path(__file__).parent.parent / "outputs" / "figures" / "statistical_analysis"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 分离特征和目标
    X = train_df[feature_cols]
    y = train_df['HOSPITAL_EXPIRE_FLAG']
    
    # 1. 目标变量分布图
    print(" 创建目标变量分布图...")
    plt.figure(figsize=(10, 6))
    y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('目标变量分布 (HOSPITAL_EXPIRE_FLAG)')
    plt.xlabel('类别 (0=存活, 1=死亡)')
    plt.ylabel('计数')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(y.value_counts()):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'target_distribution.png', dpi=300)
    plt.close()
    
    # 2. 特征与目标相关性条形图
    print(" 创建特征-目标相关性图...")
    plt.figure(figsize=(12, 8))
    top_corr = corr_df.head(15).copy()
    colors = ['red' if x < 0 else 'green' for x in top_corr['与目标相关性']]
    plt.barh(top_corr['特征'], top_corr['与目标相关性'], color=colors)
    plt.title('Top 15 特征与目标变量的相关性')
    plt.xlabel('相关系数')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'feature_target_correlation.png', dpi=300)
    plt.close()
    
    # 3. 统计检验p值分布
    print(" 创建统计检验p值分布图...")
    plt.figure(figsize=(10, 6))
    plt.hist(t_test_df['p值'], bins=50, alpha=0.7, color='steelblue')
    plt.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
    plt.axvline(x=0.01, color='darkred', linestyle='--', label='p=0.01')
    plt.title('统计检验p值分布')
    plt.xlabel('p值')
    plt.ylabel('频数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'pvalue_distribution.png', dpi=300)
    plt.close()
    
    # 4. 特征重要性比较
    print(" 创建特征重要性比较图...")
    plt.figure(figsize=(12, 8))
    
    top_10 = combined_importance.head(10)
    x = np.arange(len(top_10))
    width = 0.35
    
    plt.bar(x - width/2, top_10['F值_归一化'], width, label='ANOVA F值', alpha=0.8)
    plt.bar(x + width/2, top_10['RF_归一化'], width, label='随机森林', alpha=0.8)
    
    plt.xlabel('特征')
    plt.ylabel('归一化重要性')
    plt.title('Top 10 特征重要性比较 (ANOVA vs 随机森林)')
    plt.xticks(x, top_10['特征'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'feature_importance_comparison.png', dpi=300)
    plt.close()
    
    # 5. 最显著特征的箱线图
    print(" 创建最显著特征的箱线图...")
    
    # 选择前4个最显著的特征
    top_features = t_test_df.head(4)['特征'].tolist()
    
    if len(top_features) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features[:4]):
            # 准备数据
            data_to_plot = [X[y == 0][feature].dropna(), X[y == 1][feature].dropna()]
            
            # 创建箱线图
            bp = axes[i].boxplot(data_to_plot, patch_artist=True, 
                                 labels=['存活', '死亡'], showfliers=False)
            
            # 设置颜色
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[i].set_title(f'{feature} 分布 (p={t_test_df.iloc[i]["p值"]:.2e})')
            axes[i].set_ylabel('值')
            axes[i].grid(alpha=0.3)
        
        plt.suptitle('最显著特征的分布对比 (存活 vs 死亡)', fontsize=16)
        plt.tight_layout()
        plt.savefig(figures_dir / 'top_features_boxplot.png', dpi=300)
        plt.close()
    
    # 6. 相关性热力图（前30个特征）
    print(" 创建特征相关性热力图...")
    if len(feature_cols) > 30:
        top_features = corr_df.head(30)['特征'].tolist()
        corr_matrix = X[top_features].corr()
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                   center=0, square=True, cbar_kws={"shrink": .8})
        plt.title('Top 30 特征相关性热力图', fontsize=16)
        plt.tight_layout()
        plt.savefig(figures_dir / 'correlation_heatmap.png', dpi=300)
        plt.close()
    
    print(f"\n 所有可视化图表已保存到: {figures_dir}")
    print(f"   1. 目标变量分布图")
    print(f"   2. 特征-目标相关性图")
    print(f"   3. p值分布图")
    print(f"   4. 特征重要性比较图")
    print(f"   5. 最显著特征箱线图")
    print(f"   6. 特征相关性热力图")

def generate_statistical_report(train_df, test_df, feature_cols):
    """
    生成统计分析报告
    """
    print("\n" + "="*60)
    print(" 统计分析报告摘要")
    print("="*60)
    
    X = train_df[feature_cols]
    y = train_df['HOSPITAL_EXPIRE_FLAG']
    
    # 1. 数据集基本信息
    print("\n 数据集基本信息:")
    print(f"   训练集大小: {train_df.shape}")
    print(f"   测试集大小: {test_df.shape}")
    print(f"   特征数量: {len(feature_cols)}")
    print(f"   目标变量分布:")
    print(f"     存活 (0): {len(y[y==0])} ({len(y[y==0])/len(y):.1%})")
    print(f"     死亡 (1): {len(y[y==1])} ({len(y[y==1])/len(y):.1%})")
    
    # 2. 特征统计
    print("\n 特征统计:")
    print(f"   数值特征: {len(X.select_dtypes(include=[np.number]).columns)}")
    print(f"   分类特征: {len(X.select_dtypes(include=['object', 'category']).columns)}")
    
    # 3. 保存报告
    from pathlib import Path
    
    outputs_dir = Path(__file__).parent.parent / "outputs" / "tables"
    report_content = f"""
ICU数据分析 - 统计分析报告
===========================

1. 数据集信息
-------------
- 训练集大小: {train_df.shape}
- 测试集大小: {test_df.shape}
- 特征数量: {len(feature_cols)}
- 目标变量分布: 存活={len(y[y==0])} ({len(y[y==0])/len(y):.1%}), 死亡={len(y[y==1])} ({len(y[y==1])/len(y):.1%})

2. 数据预处理总结
-----------------
- 原始特征数: 1667
- 处理后特征数: {len(feature_cols)}
- 缺失值处理: 中位数填充
- 类别平衡: SMOTE过采样 (阳性率从5.9%提升到23.1%)

3. 关键发现
-----------
- 最相关的特征: 见feature_target_correlations.csv
- 最显著的特征: 见statistical_tests_results.csv
- 最重要的特征: 见feature_importance_combined.csv

4. 可视化图表
-------------
所有图表已保存到 outputs/figures/statistical_analysis/
"""
    
    with open(outputs_dir / 'statistical_report_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n 统计分析报告已保存到: {outputs_dir}/statistical_report_summary.txt")

def main_statistical_analysis():
    """
    主统计分析流程
    """
    print("="*60)
    print("ICU数据统计分析模块")
    print("="*60)
    
    # 1. 加载数据
    print("\n 步骤1: 加载预处理后的数据")
    train_df, test_df, feature_cols = load_processed_data()
    
    if train_df is None:
        return
    
    # 2. 描述性统计分析
    print("\n 步骤2: 描述性统计分析")
    X_0, X_1 = descriptive_statistics(train_df, feature_cols)
    
    # 3. 相关性分析
    print("\n 步骤3: 相关性分析")
    X = train_df[feature_cols]
    y = train_df['HOSPITAL_EXPIRE_FLAG']
    corr_df = correlation_analysis(X, y, feature_cols)
    
    # 4. 统计检验
    print("\n 步骤4: 统计检验")
    t_test_df = statistical_tests(X_0, X_1, feature_cols)
    
    # 5. 特征重要性分析
    print("\n 步骤5: 特征重要性分析")
    combined_importance = feature_importance_analysis(X, y, feature_cols)
    
    # 6. 创建可视化
    print("\n 步骤6: 创建可视化图表")
    create_visualizations(train_df, feature_cols, corr_df, t_test_df, combined_importance)
    
    # 7. 生成报告
    print("\n 步骤7: 生成统计分析报告")
    generate_statistical_report(train_df, test_df, feature_cols)
    
    print("\n" + "="*60)
    print(" 统计分析完成！")
    print("="*60)
    print("\n 所有分析结果已保存到 outputs/ 目录")
    print("   表格: outputs/tables/")
    print("   图表: outputs/figures/statistical_analysis/")

# 主程序入口
if __name__ == "__main__":
    main_statistical_analysis()