# 1. 导入必要的库
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到路径，这样可以从任何地方运行
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_icu_data(file_path=None):

    # 2. 设置数据文件路径
    if file_path is None:
        # 默认路径：从data/raw/目录读取
        data_dir = project_root / "data" / "raw"
        file_path = data_dir / "icu_first24hours.csv"
    
    print(f"正在读取数据文件: {file_path}")
    print("-" * 50)
    
    # 3. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件不存在！")
        print(f"请确保文件位于: {file_path}")
        print("请将icu_first24hours.csv文件复制到data/raw/目录下")
        return None
    
    # 4. 读取CSV文件
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(file_path)
        print(" 数据读取成功！")
        
        # 5. 显示基本信息
        print("\n 数据基本信息：")
        print(f"   - 数据形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
        print(f"   - 内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 显示前几行数据
        print("\n 数据前5行：")
        print(df.head())
        
        # 显示列名
        print("\n 列名列表（前20个）：")
        print(df.columns[:20].tolist())
        
        # 显示数据类型
        print("\n 数据类型：")
        print(df.dtypes.value_counts())
        
        # 检查缺失值
        print("\n 缺失值统计：")
        missing_values = df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        if len(columns_with_missing) > 0:
            print(f"有{len(columns_with_missing)}列包含缺失值：")
            print(columns_with_missing.head(10))  # 只显示前10列
        else:
            print("没有缺失值！")
        
        # 6. 保存数据副本到processed文件夹（可选）
        processed_dir = project_root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)  # 创建目录如果不存在
        
        output_path = processed_dir / "icu_data_loaded.csv"
        df.to_csv(output_path, index=False)
        print(f"\n 数据已保存到: {output_path}")
        
        return df
        
    except Exception as e:
        print(f" 读取数据时出错: {e}")
        return None

def display_basic_statistics(df):
    """
    显示数据的基本统计信息
    
    参数:
    df: Pandas DataFrame
    """
    if df is None:
        print("数据为空，无法显示统计信息")
        return
    
    print("\n 数据统计信息：")
    print("-" * 50)
    
    # 基本信息
    print(f"数据集信息:")
    print(f"  总行数: {len(df)}")
    print(f"  总列数: {len(df.columns)}")
    
    # 列分类
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime']).columns
    
    print(f"  数值列: {len(numeric_cols)}")
    print(f"  分类列: {len(categorical_cols)}")
    print(f"  日期时间列: {len(datetime_cols)}")
    
    # 显示目标变量的基本信息（如果有的话）
    target_columns = ['HOSPITAL_EXPIRE_FLAG', 'is_early_death']
    for target in target_columns:
        if target in df.columns:
            print(f"\n 目标变量 '{target}' 分布:")
            print(df[target].value_counts())
            if df[target].dtype == 'int64' or df[target].dtype == 'float64':
                print(f"  阳性比例: {df[target].mean():.2%}")
    
    # 显示数值列的基本统计
    if len(numeric_cols) > 0:
        print(f"\n 数值列统计摘要（前5列）:")
        print(df[numeric_cols[:5]].describe())

def check_data_quality(df):
    """
    检查数据质量
    
    参数:
    df: Pandas DataFrame
    """
    if df is None:
        print("数据为空，无法检查数据质量")
        return
    
    print("\n 数据质量检查：")
    print("-" * 50)
    
    # 1. 重复行检查
    duplicate_rows = df.duplicated().sum()
    print(f"重复行数: {duplicate_rows}")
    
    # 2. 缺失值检查
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    missing_percentage = (total_missing / total_cells) * 100
    print(f"总缺失值: {total_missing} ({missing_percentage:.2f}%)")
    
    # 3. 列的数据类型分布
    print(f"\n数据类型分布:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}列")
    
    # 4. 显示一些关键列的信息
    key_columns = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'age_month', 
                   'gender_is_male', 'weight_kg', 'HOSPITAL_EXPIRE_FLAG', 'is_early_death']
    
    print(f"\n 关键列信息:")
    for col in key_columns:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype}, 唯一值数量: {df[col].nunique()}")
            if df[col].dtype == 'object':
                print(f"    示例值: {df[col].iloc[0] if not df[col].isnull().all() else '全为空'}")

# 主程序入口
if __name__ == "__main__":
    """
    如果直接运行这个脚本，会执行以下代码
    """
    print("=" * 60)
    print("ICU数据读取模块")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_icu_data()
    
    if df is not None:
        # 2. 显示基本统计信息
        display_basic_statistics(df)
        
        # 3. 检查数据质量
        check_data_quality(df)
        
        print("\n" + "=" * 60)
        print(" 数据读取完成！")
        print("=" * 60)
        # 检查 is_early_death 的所有值
unique_values = df['is_early_death'].unique()
print("is_early_death 的唯一值:", unique_values)
print("值的分布:")
print(df['is_early_death'].value_counts())
weight_stats = df['weight_kg'].describe()
print("体重统计:")
print(weight_stats)

# 检查体重为0的记录数
zero_weight = df[df['weight_kg'] == 0].shape[0]
print(f"体重为0的记录数: {zero_weight}")
# 检查年龄分布
age_stats = df['age_month'].describe()
print("年龄统计（月）:")
print(age_stats)

# 转换为年
print(f"\n年龄范围（年）: {age_stats['min']/12:.1f} - {age_stats['max']/12:.1f}")