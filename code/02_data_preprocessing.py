
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data():
    """加载已经读取的数据"""
    from pathlib import Path
    import sys
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    file_path = processed_dir / "icu_data_loaded.csv"
    
    if not file_path.exists():
        print("请先运行01_data_loading.py")
        return None
    
    df = pd.read_csv(file_path)
    return df

def remove_useless_target(df):
    """
    处理目标变量问题
    is_early_death 全是0，删除此列
    """
    if 'is_early_death' in df.columns:
        print(f"删除无效目标变量 'is_early_death'（全是0）")
        df = df.drop(columns=['is_early_death'])
    
    # 检查 HOSPITAL_EXPIRE_FLAG 是否有效
    if 'HOSPITAL_EXPIRE_FLAG' in df.columns:
        positive_rate = df['HOSPITAL_EXPIRE_FLAG'].mean()
        print(f"目标变量 'HOSPITAL_EXPIRE_FLAG' 阳性率: {positive_rate:.2%}")
        print(f"阳性样本数: {df['HOSPITAL_EXPIRE_FLAG'].sum()}")
        print(f"阴性样本数: {(df['HOSPITAL_EXPIRE_FLAG'] == 0).sum()}")
    
    return df

def handle_outliers(df):
    """
    处理异常值
    """
    print("\n 处理异常值：")
    
    # 1. 年龄异常值：-1个月
    age_negative = (df['age_month'] < 0).sum()
    if age_negative > 0:
        print(f"  发现 {age_negative} 条年龄为负值的记录")
                # 设为0（新生儿）
        df.loc[df['age_month'] < 0, 'age_month'] = 0
        print("  已将负年龄设为0（新生儿）")
    
    # 2. 体重异常值：0kg
    weight_zero = (df['weight_kg'] == 0).sum()
    if weight_zero > 0:
        print(f"  发现 {weight_zero} 条体重为0的记录（占体重记录的{weight_zero/df['weight_kg'].notna().sum():.1%}）")
                # 先不处理，后续用年龄估算
    
    # 3. 检查年龄范围
    print(f"  年龄范围: {df['age_month'].min():.0f}-{df['age_month'].max():.0f} 个月")
    print(f"  体重范围: {df['weight_kg'].min():.1f}-{df['weight_kg'].max():.1f} kg")
    
    return df

def extract_date_features(df):
    """
    从ADMITTIME提取时间特征
    """
    if 'ADMITTIME' in df.columns:
        print("\n 提取时间特征：")
        
        # 将字符串转换为datetime
        df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'], errors='coerce')
        
        # 提取时间特征
        df['admit_year'] = df['ADMITTIME'].dt.year
        df['admit_month'] = df['ADMITTIME'].dt.month
        df['admit_day'] = df['ADMITTIME'].dt.day
        df['admit_hour'] = df['ADMITTIME'].dt.hour
        df['admit_dayofweek'] = df['ADMITTIME'].dt.dayofweek  # 周一=0, 周日=6
        df['admit_season'] = df['ADMITTIME'].dt.month % 12 // 3 + 1  # 季节
        
        print(f"  提取了 {df['admit_year'].nunique()} 个不同年份的数据")
        print(f"  入院时间范围: {df['ADMITTIME'].min()} 到 {df['ADMITTIME'].max()}")
    
    return df

def feature_selection_via_missingness(df, missing_threshold=0.8):
    """
    基于缺失率进行特征选择
    删除缺失率过高的特征
    """
    print(f"\n 基于缺失率的特征选择（阈值={missing_threshold:.0%}）：")
    
    # 计算每个特征的缺失率
    missing_rates = df.isnull().sum() / len(df)
    
    # 选择缺失率低于阈值的特征
    selected_features = missing_rates[missing_rates < missing_threshold].index.tolist()
    
    print(f"  原始特征数: {len(df.columns)}")
    print(f"  删除缺失率 > {missing_threshold:.0%} 的特征后: {len(selected_features)}")
    print(f"  删除了 {len(df.columns) - len(selected_features)} 个特征")
    
    # 保存被删除的特征
    removed_features = missing_rates[missing_rates >= missing_threshold].index.tolist()
    print(f"  前10个被删除的高缺失特征: {removed_features[:10]}")
    
    return df[selected_features]

def handle_missing_values(df, strategy='median'):
    """
    处理缺失值
    """
    print(f"\n 处理缺失值（策略: {strategy}）：")
    
    # 分离数值特征和分类特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 移除目标变量（如果有）
    target_cols = ['HOSPITAL_EXPIRE_FLAG', 'SUBJECT_ID', 'HADM_ID']
    numeric_cols = [col for col in numeric_cols if col not in target_cols]
    
    # 计算填充前的缺失情况
    missing_before = df[numeric_cols].isnull().sum().sum()
    total_cells = df[numeric_cols].size
    print(f"  数值特征缺失值: {missing_before} ({missing_before/total_cells:.1%})")
    
    # 填充数值特征
    if strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 计算填充后的缺失情况
    missing_after = df[numeric_cols].isnull().sum().sum()
    print(f"  填充后数值特征缺失值: {missing_after}")
    
    # 处理分类特征（如果有的话）
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in target_cols]
    
    if categorical_cols:
        # 用众数填充
        for col in categorical_cols:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_value)
    
    return df

def handle_class_imbalance(X, y):
    """
    处理类别不平衡
    使用SMOTE过采样
    """
    print("\n 处理类别不平衡：")
    print(f"  采样前 - 类别0: {(y == 0).sum()}, 类别1: {(y == 1).sum()}")
    print(f"  不平衡比例: {(y == 0).sum() / len(y):.1%} : {(y == 1).sum() / len(y):.1%}")
    
    # 使用SMOTE过采样
    try:
        smote = SMOTE(random_state=42, sampling_strategy=0.3)  # 将少数类增加到30%
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"  采样后 - 类别0: {(y_resampled == 0).sum()}, 类别1: {(y_resampled == 1).sum()}")
        print(f"  新比例: {(y_resampled == 0).sum() / len(y_resampled):.1%} : {(y_resampled == 1).sum() / len(y_resampled):.1%}")
        
        return X_resampled, y_resampled
    except Exception as e:
        print(f"  SMOTE失败: {e}")
        print("  返回原始数据")
        return X, y

def prepare_features_and_target(df):
    """
    准备特征和目标变量
    """
    print("\n 准备特征和目标变量：")
    
    # 确定目标变量
    target = 'HOSPITAL_EXPIRE_FLAG'
    
    # 要排除的特征（ID、时间戳等）
    exclude_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', target]
    
    # 特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  目标变量: {target}")
    
    X = df[feature_cols]
    y = df[target]
    
    return X, y, feature_cols

def split_data_by_patient(df, test_size=0.2):
    """
    按患者分割数据，确保同一个患者不出现在训练集和测试集中
    """
    print("\n 按患者分割数据：")
    
    # 获取患者ID列表
    patient_ids = df['SUBJECT_ID'].unique()
    print(f"  唯一患者数: {len(patient_ids)}")
    print(f"  总记录数: {len(df)}")
    
    # 按患者分割
    train_patients, test_patients = train_test_split(
        patient_ids, test_size=test_size, random_state=42, stratify=None
    )
    
    # 创建训练集和测试集
    train_df = df[df['SUBJECT_ID'].isin(train_patients)]
    test_df = df[df['SUBJECT_ID'].isin(test_patients)]
    
    print(f"  训练集患者数: {len(train_patients)} ({len(train_df)} 条记录)")
    print(f"  测试集患者数: {len(test_patients)} ({len(test_df)} 条记录)")
    
    return train_df, test_df

def save_processed_data(train_df, test_df, feature_cols):
    """
    保存处理后的数据
    """
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据
    train_df.to_csv(processed_dir / "train_data.csv", index=False)
    test_df.to_csv(processed_dir / "test_data.csv", index=False)
    
    # 保存特征列表
    feature_df = pd.DataFrame({'feature': feature_cols})
    feature_df.to_csv(processed_dir / "feature_list.csv", index=False)
    
    print(f"\n💾 数据已保存到: {processed_dir}")
    print(f"  训练集: train_data.csv ({len(train_df)} 行)")
    print(f"  测试集: test_data.csv ({len(test_df)} 行)")
    print(f"  特征列表: feature_list.csv ({len(feature_cols)} 个特征)")

def main_preprocessing_pipeline():
    """
    主预处理流程
    """
    print("=" * 60)
    print("ICU数据预处理模块")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n 步骤1: 加载数据")
    df = load_preprocessed_data()
    if df is None:
        return
    
    print(f"  原始数据形状: {df.shape}")
    
    # 2. 处理目标变量
    print("\n 步骤2: 处理目标变量")
    df = remove_useless_target(df)
    
    # 3. 处理异常值
    df = handle_outliers(df)
    
    # 4. 提取时间特征
    df = extract_date_features(df)
    
    # 5. 基于缺失率的特征选择
    print("\n 步骤3: 特征选择")
    df = feature_selection_via_missingness(df, missing_threshold=0.5)  # 50%阈值
    
    # 6. 处理缺失值
    print("\n 步骤4: 处理缺失值")
    df = handle_missing_values(df, strategy='median')
    
    # 7. 按患者分割数据
    print("\n 步骤5: 数据分割")
    train_df, test_df = split_data_by_patient(df, test_size=0.2)
    
    # 8. 准备训练集的特征和目标
    print("\n 步骤6: 准备训练数据")
    X_train, y_train, feature_cols = prepare_features_and_target(train_df)
    
    # 9. 处理类别不平衡（只在训练集上）
    print("\n 步骤7: 处理类别不平衡（仅训练集）")
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)
    
    # 10. 准备测试集的特征和目标
    X_test, y_test, _ = prepare_features_and_target(test_df)
    
    # 11. 保存处理后的数据
    print("\n 步骤8: 保存数据")
    # 创建包含所有特征的训练集和测试集
    train_df_processed = pd.DataFrame(X_train_resampled, columns=feature_cols)
    train_df_processed['HOSPITAL_EXPIRE_FLAG'] = y_train_resampled
    
    test_df_processed = pd.DataFrame(X_test, columns=feature_cols)
    test_df_processed['HOSPITAL_EXPIRE_FLAG'] = y_test
    
    # 添加ID信息（可选）
    train_df_processed['SUBJECT_ID'] = np.random.choice(train_df['SUBJECT_ID'].unique(), 
                                                       size=len(train_df_processed))
    test_df_processed['SUBJECT_ID'] = test_df['SUBJECT_ID'].values
    
    save_processed_data(train_df_processed, test_df_processed, feature_cols)
    
    # 12. 打印最终报告
    print("\n" + "=" * 60)
    print(" 数据预处理完成！")
    print("=" * 60)
    print("\n 最终数据报告：")
    print(f"  训练集形状: {train_df_processed.shape}")
    print(f"  测试集形状: {test_df_processed.shape}")
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  目标变量分布（训练集）:")
    print(f"    类别0: {(train_df_processed['HOSPITAL_EXPIRE_FLAG'] == 0).sum()}")
    print(f"    类别1: {(train_df_processed['HOSPITAL_EXPIRE_FLAG'] == 1).sum()}")
    print(f"    阳性率: {train_df_processed['HOSPITAL_EXPIRE_FLAG'].mean():.2%}")
    
    return train_df_processed, test_df_processed, feature_cols

# 主程序入口
if __name__ == "__main__":
    train_df, test_df, features = main_preprocessing_pipeline()