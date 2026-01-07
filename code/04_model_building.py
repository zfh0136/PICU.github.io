
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 导入多种模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def load_training_data():
    """加载训练数据"""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    
    train_path = processed_dir / "train_data.csv"
    test_path = processed_dir / "test_data.csv"
    
    if not (train_path.exists() and test_path.exists()):
        print("请先运行03_statistical_analysis.py或02_data_preprocessing.py")
        return None, None, None, None
    
    # 加载数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 分离特征和目标
    feature_cols = [col for col in train_df.columns if col not in ['HOSPITAL_EXPIRE_FLAG', 'SUBJECT_ID']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['HOSPITAL_EXPIRE_FLAG']
    
    X_test = test_df[feature_cols]
    y_test = test_df['HOSPITAL_EXPIRE_FLAG']
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"特征数: {len(feature_cols)}")
    print(f"训练集目标分布: 0={sum(y_train==0)} ({sum(y_train==0)/len(y_train):.1%}), "
          f"1={sum(y_train==1)} ({sum(y_train==1)/len(y_train):.1%})")
    
    return X_train, y_train, X_test, y_test, feature_cols

def scale_features(X_train, X_test):
    """特征标准化"""
    print("\n 特征标准化...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 保存scaler
    from pathlib import Path
    models_dir = Path(__file__).parent.parent / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(" 特征标准化完成，scaler已保存")
    
    return X_train_scaled, X_test_scaled

def define_models():
    """定义要训练的模型"""
    models = {
        '逻辑回归': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        '随机森林': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        '梯度提升': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1),
        '支持向量机': SVC(random_state=42, probability=True, class_weight='balanced'),
        'K近邻': KNeighborsClassifier(n_jobs=-1),
        '朴素贝叶斯': GaussianNB()
    }
    
    print(f"将训练 {len(models)} 种模型:")
    for name, model in models.items():
        print(f"  - {name}")
    
    return models

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """评估单个模型"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt
    
    # 训练模型
    print(f"  训练{model_name}...")
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 计算各项指标
    metrics = {
        '准确率': accuracy_score(y_test, y_pred),
        '精确率': precision_score(y_test, y_pred, zero_division=0),
        '召回率': recall_score(y_test, y_pred, zero_division=0),
        'F1分数': f1_score(y_test, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['AUC'] = roc_auc_score(y_test, y_pred_proba)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    metrics['交叉验证F1均值'] = cv_scores.mean()
    metrics['交叉验证F1标准差'] = cv_scores.std()
    
    return model, metrics

def train_and_evaluate_all_models(X_train, y_train, X_test, y_test):
    """训练和评估所有模型"""
    print("\n" + "="*60)
    print(" 训练和评估多种机器学习模型")
    print("="*60)
    
    models = define_models()
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n 处理模型: {name}")
        try:
            trained_model, metrics = evaluate_model(model, X_train, y_train, X_test, y_test, name)
            results[name] = metrics
            trained_models[name] = trained_model
            
            print(f"  性能指标:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")
                
        except Exception as e:
            print(f"   训练{name}时出错: {e}")
            results[name] = None
    
    return results, trained_models

def hyperparameter_tuning(X_train, y_train):
    """对最佳模型进行超参数调优"""
    print("\n" + "="*60)
    print(" 对最佳模型进行超参数调优")
    print("="*60)
    
    # 定义参数网格
    param_grids = {
        '随机森林': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 1.0]
        }
    }
    
    # 只调优几个关键模型
    models_to_tune = {
        '随机森林': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1)
    }
    
    tuned_models = {}
    tuning_results = {}
    
    for name, model in models_to_tune.items():
        print(f"\n🔍 对{name}进行超参数调优...")
        
        try:
            # 使用随机搜索（比网格搜索更快）
            random_search = RandomizedSearchCV(
                model, param_grids[name], 
                n_iter=20,  # 随机尝试20组参数
                cv=3, 
                scoring='f1',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            random_search.fit(X_train, y_train)
            
            tuned_models[name] = random_search.best_estimator_
            tuning_results[name] = {
                '最佳参数': random_search.best_params_,
                '最佳分数': random_search.best_score_
            }
            
            print(f"  最佳参数: {random_search.best_params_}")
            print(f"  最佳交叉验证F1分数: {random_search.best_score_:.4f}")
            
        except Exception as e:
            print(f"   {name}调优失败: {e}")
    
    return tuned_models, tuning_results

def save_models_and_results(trained_models, results, tuned_models, tuning_results):
    """保存模型和结果"""
    from pathlib import Path
    import json
    
    # 创建目录
    models_dir = Path(__file__).parent.parent / "outputs" / "models"
    tables_dir = Path(__file__).parent.parent / "outputs" / "tables"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存所有训练好的模型
    print("\n 保存模型...")
    for name, model in trained_models.items():
        try:
            model_path = models_dir / f'{name.replace(" ", "_")}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"   {name}: 已保存")
        except Exception as e:
            print(f"   保存{name}失败: {e}")
    
    # 保存调优后的模型
    for name, model in tuned_models.items():
        try:
            model_path = models_dir / f'{name.replace(" ", "_")}_tuned.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"   {name}(调优后): 已保存")
        except Exception as e:
            print(f"   保存{name}(调优后)失败: {e}")
    
    # 2. 保存结果到CSV
    print("\n 保存结果...")
    
    # 基础模型结果
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('F1分数', ascending=False)
    results_df.to_csv(tables_dir / 'model_performance_comparison.csv')
    print(f"  ✅ 模型性能比较表已保存")
    
    # 调优结果
    if tuning_results:
        tuning_df = pd.DataFrame(tuning_results).T
        tuning_df.to_csv(tables_dir / 'hyperparameter_tuning_results.csv')
        print(f"   超参数调优结果已保存")
    
    # 3. 保存结果为JSON（便于报告使用）
    results_dict = {
        '基础模型性能': results_df.to_dict(),
        '超参数调优结果': tuning_results
    }
    
    with open(tables_dir / 'model_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"   完整结果已保存为JSON格式")
    
    return results_df

def identify_best_model(results_df, trained_models, tuned_models):
    """确定最佳模型"""
    print("\n" + "="*60)
    print(" 确定最佳模型")
    print("="*60)
    
    # 找出基础模型中F1分数最高的
    best_basic_model = results_df.index[0]
    best_basic_score = results_df.iloc[0]['F1分数']
    
    print(f"基础模型中最佳: {best_basic_model}")
    print(f"F1分数: {best_basic_score:.4f}")
    
    # 如果有调优模型，比较调优后的性能
    if tuned_models:
        print(f"\n调优后的模型:")
        for name, model in tuned_models.items():
            # 需要重新评估调优模型
            from sklearn.metrics import f1_score
            
            # 这里需要X_test和y_test，暂时跳过具体评估
            print(f"  {name}: 已调优，参数已保存")
    
    # 推荐最佳模型
    print(f"\n 推荐模型: {best_basic_model}")
    print(f"理由: 在基础模型中F1分数最高")
    
    return best_basic_model

def feature_importance_analysis_for_best_model(best_model_name, trained_models, X_train, feature_cols):
    """分析最佳模型的特征重要性"""
    print("\n" + "="*60)
    print(" 最佳模型特征重要性分析")
    print("="*60)
    
    model = trained_models.get(best_model_name)
    
    if model is None:
        print(f"找不到模型: {best_model_name}")
        return None
    
    # 检查模型是否有特征重要性属性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            '特征': feature_cols,
            '重要性': importances
        }).sort_values('重要性', ascending=False)
        
        print(f"前10个最重要的特征:")
        print(importance_df.head(10).to_string())
        
        # 保存结果
        from pathlib import Path
        tables_dir = Path(__file__).parent.parent / "outputs" / "tables"
        importance_df.to_csv(tables_dir / f'{best_model_name.replace(" ", "_")}_feature_importance.csv', index=False)
        
        print(f"\n 特征重要性已保存")
        
        return importance_df
    
    elif hasattr(model, 'coef_'):  # 逻辑回归等线性模型
        coefficients = model.coef_[0]
        
        # 创建系数DataFrame
        coef_df = pd.DataFrame({
            '特征': feature_cols,
            '系数': coefficients,
            '系数绝对值': np.abs(coefficients)
        }).sort_values('系数绝对值', ascending=False)
        
        print(f"前10个最重要的特征（按系数绝对值）:")
        print(coef_df.head(10).to_string())
        
        # 保存结果
        from pathlib import Path
        tables_dir = Path(__file__).parent.parent / "outputs" / "tables"
        coef_df.to_csv(tables_dir / f'{best_model_name.replace(" ", "_")}_coefficients.csv', index=False)
        
        print(f"\n 模型系数已保存")
        
        return coef_df
    
    else:
        print(f"模型 {best_model_name} 没有特征重要性或系数属性")
        return None

def main_model_building():
    """主模型建立流程"""
    print("="*60)
    print(" ICU死亡率预测模型建立")
    print("="*60)
    
    # 1. 加载数据
    print("\n 步骤1: 加载数据")
    data = load_training_data()
    if data[0] is None:
        return
    
    X_train, y_train, X_test, y_test, feature_cols = data
    
    # 2. 特征标准化
    print("\n 步骤2: 特征标准化")
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # 3. 训练和评估多种模型
    print("\n 步骤3: 训练多种机器学习模型")
    results, trained_models = train_and_evaluate_all_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 4. 超参数调优
    print("\n 步骤4: 超参数调优")
    tuned_models, tuning_results = hyperparameter_tuning(X_train_scaled, y_train)
    
    # 5. 保存模型和结果
    print("\n 步骤5: 保存模型和结果")
    results_df = save_models_and_results(trained_models, results, tuned_models, tuning_results)
    
    # 6. 确定最佳模型
    print("\n 步骤6: 确定最佳模型")
    best_model = identify_best_model(results_df, trained_models, tuned_models)
    
    # 7. 分析最佳模型的特征重要性
    print("\n 步骤7: 分析最佳模型的特征重要性")
    feature_importance = feature_importance_analysis_for_best_model(
        best_model, trained_models, X_train_scaled, feature_cols
    )
    
    # 8. 生成模型建立报告
    print("\n" + "="*60)
    print("模型建立完成！")
    print("="*60)
    
    print(f"\n 模型性能总结:")
    print(results_df[['准确率', '精确率', '召回率', 'F1分数', 'AUC']].head().to_string())
    
    print(f"\n 下一步:")
    print(f"  1. 查看 outputs/tables/model_performance_comparison.csv")
    print(f"  2. 查看 outputs/models/ 中的保存的模型")
    print(f"  3. 运行 05_model_evaluation.py 进行详细评估和可视化")
    
    return best_model, results_df, trained_models

# 主程序入口
if __name__ == "__main__":
    best_model, results_df, trained_models = main_model_building()