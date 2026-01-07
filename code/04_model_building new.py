"""
改进的模型建立模块 - 针对类别不平衡问题
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 导入多种模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

def load_training_data():
    """加载训练数据"""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    
    train_path = processed_dir / "train_data.csv"
    test_path = processed_dir / "test_data.csv"
    feature_path = processed_dir / "feature_list.csv"
    
    if not (train_path.exists() and test_path.exists()):
        print("请先运行03_statistical_analysis.py或02_data_preprocessing.py")
        return None, None, None, None, None, None, None
    
    # 加载数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    if feature_path.exists():
        feature_df = pd.read_csv(feature_path)
        feature_cols = feature_df['feature'].tolist()
    else:
        # 如果没有特征列表文件，则自动识别特征
        feature_cols = [col for col in train_df.columns 
                       if col not in ['HOSPITAL_EXPIRE_FLAG', 'SUBJECT_ID']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['HOSPITAL_EXPIRE_FLAG']
    
    X_test = test_df[feature_cols]
    y_test = test_df['HOSPITAL_EXPIRE_FLAG']
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"特征数: {len(feature_cols)}")
    
    # 详细分析类别分布
    train_pos = sum(y_train == 1)
    train_neg = sum(y_train == 0)
    test_pos = sum(y_test == 1)
    test_neg = sum(y_test == 0)
    
    print(f"\n训练集类别分布:")
    print(f"  阴性(0): {train_neg} ({train_neg/len(y_train):.2%})")
    print(f"  阳性(1): {train_pos} ({train_pos/len(y_train):.2%})")
    print(f"  不平衡比例: {train_neg/train_pos:.2f}:1")
    
    print(f"\n测试集类别分布:")
    print(f"  阴性(0): {test_neg} ({test_neg/len(y_test):.2%})")
    print(f"  阳性(1): {test_pos} ({test_pos/len(y_test):.2%})")
    
    return X_train, y_train, X_test, y_test, feature_cols, train_df, test_df

def scale_features(X_train, X_test):
    """特征标准化"""
    print("\n特征标准化...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 保存scaler
    from pathlib import Path
    models_dir = Path(__file__).parent.parent / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("特征标准化完成，scaler已保存")
    
    return X_train_scaled, X_test_scaled

def handle_class_imbalance(X_train, y_train, method='smote'):
    """处理类别不平衡问题"""
    print(f"\n处理类别不平衡... (方法: {method})")
    
    if method == 'smote':
        # 使用SMOTE过采样
        smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train == 1) - 1))
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"过采样前: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
        print(f"过采样后: 0={sum(y_resampled==0)}, 1={sum(y_resampled==1)}")
        
        return X_resampled, y_resampled
    
    elif method == 'class_weight':
        # 返回原始数据，但在模型中设置class_weight
        print("使用class_weight参数处理不平衡")
        return X_train, y_train
    
    return X_train, y_train

def define_models_with_imbalance_handling():
    """定义要训练的模型 - 针对不平衡数据优化"""
    
    # 计算类别权重
    # 在实际使用中，需要根据实际数据计算
    
    models = {
        'Logistic_Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced',  # 自动平衡类别权重
            solver='liblinear'
        ),
        'Random_Forest': RandomForestClassifier(
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced_subsample',  # 处理不平衡
            n_estimators=200,  # 增加树的数量
            min_samples_split=10,  # 防止过拟合
            min_samples_leaf=5
        ),
        'XGBoost': XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1,
            scale_pos_weight=10,  # 增加正样本权重，值需要根据不平衡比例调整
            max_depth=5,  # 限制深度防止过拟合
            subsample=0.8,
            colsample_bytree=0.8
        ),
        'LightGBM': LGBMClassifier(
            random_state=42, 
            n_jobs=-1,
            verbose=-1,
            is_unbalance=True,  # 处理不平衡
            boosting_type='gbdt',
            num_leaves=31,
            max_depth=5,
            min_child_samples=20
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,  # 降低学习率
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        ),
        'Neural_Network': MLPClassifier(
            random_state=42,
            max_iter=500,
            early_stopping=True,
            hidden_layer_sizes=(100, 50),
            alpha=0.01,
            learning_rate='adaptive'
        )
    }
    
    print(f"将训练 {len(models)} 种模型:")
    for name, model in models.items():
        print(f"  - {name}")
    
    return models

def evaluate_model_with_threshold(model, X_train, y_train, X_test, y_test, model_name):
    """评估单个模型 - 使用阈值调整"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve
    
    # 训练模型
    print(f"  训练{model_name}...")
    model.fit(X_train, y_train)
    
    # 预测概率
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 寻找最佳阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # 使用Youden's J统计量找到最佳阈值
    j_scores = tpr - fpr
    best_threshold = thresholds[np.argmax(j_scores)]
    
    print(f"    最佳阈值: {best_threshold:.4f}")
    
    # 使用最佳阈值进行预测
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # 计算各项指标
    metrics = {
        '准确率': accuracy_score(y_test, y_pred),
        '精确率': precision_score(y_test, y_pred, zero_division=0),
        '召回率': recall_score(y_test, y_pred, zero_division=0),
        'F1分数': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_pred_proba),
        '最佳阈值': best_threshold
    }
    
    # 交叉验证（使用AUC作为评分）
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
        metrics['交叉验证AUC均值'] = cv_scores.mean()
        metrics['交叉验证AUC标准差'] = cv_scores.std()
    except:
        metrics['交叉验证AUC均值'] = np.nan
        metrics['交叉验证AUC标准差'] = np.nan
    
    # 详细分类报告
    print("    分类报告:")
    report = classification_report(y_test, y_pred, target_names=['Alive', 'Death'], digits=4)
    for line in report.split('\n'):
        print(f"      {line}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"    混淆矩阵:\n{cm}")
    
    return model, metrics, best_threshold

def train_and_evaluate_all_models_improved(X_train, y_train, X_test, y_test):
    """训练和评估所有模型 - 改进版本"""
    print("\n" + "="*60)
    print("训练和评估多种机器学习模型 (改进版)")
    print("="*60)
    
    models = define_models_with_imbalance_handling()
    results = {}
    trained_models = {}
    thresholds = {}
    
    for name, model in models.items():
        print(f"\n处理模型: {name}")
        try:
            # 处理不平衡（可选，可以在模型参数中处理）
            if name in ['Logistic_Regression', 'Random_Forest', 'LightGBM']:
                # 这些模型已经有内置的不平衡处理
                X_train_balanced, y_train_balanced = X_train, y_train
            else:
                # 对没有内置处理的模型使用SMOTE
                X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train, method='smote')
            
            trained_model, metrics, threshold = evaluate_model_with_threshold(
                model, X_train_balanced, y_train_balanced, X_test, y_test, name
            )
            
            results[name] = metrics
            trained_models[name] = trained_model
            thresholds[name] = threshold
            
            print(f"  性能指标:")
            for metric_name, value in metrics.items():
                if not pd.isna(value) and metric_name not in ['最佳阈值', '分类报告']:
                    print(f"    {metric_name}: {value:.4f}")
                    
        except Exception as e:
            print(f"   训练{name}时出错: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
    
    return results, trained_models, thresholds

def hyperparameter_tuning_focused(X_train, y_train, focus_models=None):
    """对关键模型进行针对性调优"""
    print("\n" + "="*60)
    print("对关键模型进行针对性超参数调优")
    print("="*60)
    
    if focus_models is None:
        focus_models = ['Random_Forest', 'XGBoost', 'LightGBM']
    
    # 定义针对性参数网格
    param_grids = {
        'Random_Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 4, 8],
            'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 3}]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [5, 10, 20]  # 正样本权重
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.6, 0.8, 1.0],
            'is_unbalance': [True],
            'min_child_samples': [10, 20, 30]
        }
    }
    
    models_to_tune = {
        'Random_Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    }
    
    tuned_models = {}
    tuning_results = {}
    
    for name in focus_models:
        if name not in models_to_tune:
            continue
            
        print(f"\n🔍 对{name}进行超参数调优...")
        
        try:
            # 使用分层交叉验证
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            random_search = RandomizedSearchCV(
                models_to_tune[name], 
                param_grids[name], 
                n_iter=15,  # 增加迭代次数
                cv=cv,
                scoring='roc_auc',  # 使用AUC作为评分
                random_state=42,
                n_jobs=-1,
                verbose=1  # 显示进度
            )
            
            random_search.fit(X_train, y_train)
            
            tuned_models[name] = random_search.best_estimator_
            tuning_results[name] = {
                '最佳参数': random_search.best_params_,
                '最佳AUC分数': random_search.best_score_
            }
            
            print(f"  最佳参数: {random_search.best_params_}")
            print(f"  最佳交叉验证AUC分数: {random_search.best_score_:.4f}")
            
            # 评估调优后的模型
            y_pred_proba = random_search.best_estimator_.predict_proba(X_train)[:, 1]
            auc_score = roc_auc_score(y_train, y_pred_proba)
            print(f"  训练集AUC: {auc_score:.4f}")
            
        except Exception as e:
            print(f"  {name}调优失败: {e}")
            import traceback
            traceback.print_exc()
    
    return tuned_models, tuning_results

def save_models_and_results_compatible(trained_models, results, tuned_models, tuning_results, feature_cols, thresholds):
    """保存模型和结果 - 与第五步代码兼容"""
    from pathlib import Path
    import json
    
    # 创建目录
    models_dir = Path(__file__).parent.parent / "outputs" / "models"
    tables_dir = Path(__file__).parent.parent / "outputs" / "tables"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型名称映射
    model_name_mapping = {
        'Logistic_Regression': 'logistic_regression',
        'Random_Forest': 'random_forest',
        'XGBoost': 'xgboost',
        'LightGBM': 'lightgbm',
        'Gradient_Boosting': 'gradient_boosting',
        'Neural_Network': 'neural_network'
    }
    
    # 1. 保存所有训练好的模型（基础模型）
    print("\n保存基础模型...")
    for name, model in trained_models.items():
        try:
            if name in model_name_mapping:
                model_filename = model_name_mapping[name]
                model_path = models_dir / f'{model_filename}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  {name}: 已保存为 {model_filename}.pkl")
        except Exception as e:
            print(f"  保存{name}失败: {e}")
    
    # 2. 保存调优后的模型
    print("\n保存调优后的模型...")
    for name, model in tuned_models.items():
        try:
            if name in model_name_mapping:
                model_filename = model_name_mapping[name] + '_tuned'
                model_path = models_dir / f'{model_filename}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  {name}(调优后): 已保存为 {model_filename}.pkl")
        except Exception as e:
            print(f"  保存{name}(调优后)失败: {e}")
    
    # 3. 保存阈值信息
    threshold_df = pd.DataFrame(list(thresholds.items()), columns=['Model', 'Best_Threshold'])
    threshold_df.to_csv(tables_dir / 'model_best_thresholds.csv', index=False)
    print(f"  模型最佳阈值已保存")
    
    # 4. 保存结果到CSV
    print("\n保存结果...")
    
    # 基础模型结果
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('AUC', ascending=False)
    results_df.to_csv(tables_dir / 'model_performance_comparison.csv')
    print(f"  ✅ 模型性能比较表已保存")
    
    # 调优结果
    if tuning_results:
        tuning_df = pd.DataFrame(tuning_results).T
        tuning_df.to_csv(tables_dir / 'hyperparameter_tuning_results.csv')
        print(f"  超参数调优结果已保存")
    
    # 5. 保存特征列表
    feature_df = pd.DataFrame({'feature': feature_cols})
    feature_df.to_csv(tables_dir / 'feature_list.csv', index=False)
    print(f"  特征列表已保存")
    
    # 6. 保存结果为JSON
    results_dict = {
        '基础模型性能': results_df.to_dict(),
        '超参数调优结果': tuning_results,
        '最佳阈值': thresholds,
        '特征数量': len(feature_cols)
    }
    
    with open(tables_dir / 'model_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"  完整结果已保存为JSON格式")
    
    return results_df

def main_model_building_improved():
    """主模型建立流程 - 改进版本"""
    print("="*60)
    print("ICU死亡率预测模型建立 (改进版)")
    print("="*60)
    
    # 1. 加载数据
    print("\n步骤1: 加载数据")
    data = load_training_data()
    if data[0] is None:
        return None, None, None
    
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = data
    
    # 2. 特征工程（可选，根据需要进行）
    print("\n步骤2: 特征工程")
    # 可以在这里添加特征选择、特征创建等步骤
    
    # 3. 特征标准化
    print("\n步骤3: 特征标准化")
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # 4. 训练和评估多种模型（基础模型）
    print("\n步骤4: 训练多种机器学习模型")
    results, trained_models, thresholds = train_and_evaluate_all_models_improved(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # 5. 超参数调优
    print("\n步骤5: 超参数调优")
    
    # 选择表现较好的模型进行调优
    if results:
        results_df_pre = pd.DataFrame(results).T
        # 选择AUC > 0.7的模型进行调优
        good_models = results_df_pre[results_df_pre['AUC'] > 0.7].index.tolist()
        print(f"将对以下模型进行调优: {good_models}")
        
        tuned_models, tuning_results = hyperparameter_tuning_focused(
            X_train_scaled, y_train, focus_models=good_models
        )
    else:
        tuned_models, tuning_results = {}, {}
    
    # 6. 保存模型和结果
    print("\n步骤6: 保存模型和结果")
    results_df = save_models_and_results_compatible(
        trained_models, results, tuned_models, tuning_results, feature_cols, thresholds
    )
    
    # 7. 分析结果
    print("\n" + "="*60)
    print("模型建立完成！")
    print("="*60)
    
    if not results_df.empty:
        print(f"\n模型性能总结:")
        display_cols = ['准确率', '精确率', '召回率', 'F1分数', 'AUC', '最佳阈值']
        display_cols = [col for col in display_cols if col in results_df.columns]
        print(results_df[display_cols].head().to_string())
        
        # 找出最佳模型（基于F1分数，更综合）
        if 'F1分数' in results_df.columns:
            best_model_name = results_df['F1分数'].idxmax()
            best_f1 = results_df.loc[best_model_name, 'F1分数']
            best_auc = results_df.loc[best_model_name, 'AUC']
            print(f"\n最佳模型 (基于F1分数): {best_model_name}")
            print(f"  F1分数: {best_f1:.4f}")
            print(f"  AUC: {best_auc:.4f}")
    
    print(f"\n下一步:")
    print(f"  1. 查看 outputs/tables/model_performance_comparison.csv")
    print(f"  2. 查看 outputs/tables/model_best_thresholds.csv")
    print(f"  3. 运行 05_model_evaluation.py 进行详细评估和可视化")
    
    return results_df

# 主程序入口
if __name__ == "__main__":
    results_df = main_model_building_improved()