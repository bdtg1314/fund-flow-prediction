"""
资金流入流出预测 - 特征工程
构建时间序列预测所需的各类特征
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_raw_data(data_path='../data/'):
    """加载原始数据"""
    # 用户申购赎回数据
    balance = pd.read_csv(data_path + 'user_balance_table.csv')
    balance['report_date'] = pd.to_datetime(balance['report_date'], format='%Y%m%d')
    
    # 每日收益率
    interest = pd.read_csv(data_path + 'mfd_day_share_interest.csv')
    interest['mfd_date'] = pd.to_datetime(interest['mfd_date'], format='%Y%m%d')
    
    # 银行间拆借利率
    shibor = pd.read_csv(data_path + 'mfd_bank_shibor.csv')
    shibor['mfd_date'] = pd.to_datetime(shibor['mfd_date'], format='%Y%m%d')
    
    return balance, interest, shibor

def create_daily_aggregation(balance):
    """按日期聚合，生成每日总申购和总赎回"""
    daily_data = balance.groupby('report_date').agg({
        'total_purchase_amt': 'sum',
        'total_redeem_amt': 'sum',
        'direct_purchase_amt': 'sum',
        'purchase_bal_amt': 'sum',
        'direct_redeem_amt': 'sum',
        'redeem_bal_amt': 'sum',
        'tBalance': 'sum',
        'yBalance': 'sum',
        'user_id': 'nunique'
    }).reset_index()
    
    daily_data.columns = [
        'report_date', 'total_purchase', 'total_redeem',
        'direct_purchase', 'purchase_bal',
        'direct_redeem', 'redeem_bal',
        'tbalance', 'ybalance', 'active_users'
    ]
    
    # 计算净流入
    daily_data['net_flow'] = daily_data['total_purchase'] - daily_data['total_redeem']
    
    return daily_data

def add_time_features(df):
    """添加时间特征"""
    df = df.copy()
    
    # 基础时间特征
    df['year'] = df['report_date'].dt.year
    df['month'] = df['report_date'].dt.month
    df['day'] = df['report_date'].dt.day
    df['weekday'] = df['report_date'].dt.weekday  # 0=周一, 6=周日
    df['weekofyear'] = df['report_date'].dt.isocalendar().week.astype(int)
    df['dayofyear'] = df['report_date'].dt.dayofyear
    
    # 是否周末
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # 月初、月末特征
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)
    df['is_mid_month'] = ((df['day'] >= 10) & (df['day'] <= 20)).astype(int)
    
    # 季度特征
    df['quarter'] = df['report_date'].dt.quarter
    df['is_quarter_end'] = ((df['month'] % 3 == 0) & (df['day'] >= 25)).astype(int)
    
    # 星期几的独热编码
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday')
    df = pd.concat([df, weekday_dummies], axis=1)
    
    # 月份的独热编码
    month_dummies = pd.get_dummies(df['month'], prefix='month')
    df = pd.concat([df, month_dummies], axis=1)
    
    # 周期特征（正弦余弦编码）
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def add_lag_features(df, target_cols, lags=[1, 2, 3, 7, 14, 30]):
    """添加滞后特征"""
    df = df.copy()
    
    for col in target_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def add_rolling_features(df, target_cols, windows=[3, 7, 14, 30]):
    """添加滑动窗口统计特征"""
    df = df.copy()
    
    for col in target_cols:
        for window in windows:
            # 均值
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
            # 标准差
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
            # 最大值
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
            # 最小值
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
            # 中位数
            df[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window).median()
    
    return df

def add_weekday_weekend_features(df, target_cols):
    """添加工作日/周末分别的统计特征"""
    df = df.copy()
    
    for col in target_cols:
        # 过去7天中工作日的均值
        weekday_mask = df['weekday'] < 5
        df[f'{col}_weekday_mean_7'] = df[col].where(weekday_mask).rolling(7).mean()
        
        # 过去7天中周末的均值
        weekend_mask = df['weekday'] >= 5
        df[f'{col}_weekend_mean_7'] = df[col].where(weekend_mask).rolling(7).mean()
    
    return df

def add_diff_features(df, target_cols, periods=[1, 7, 30]):
    """添加差分特征"""
    df = df.copy()
    
    for col in target_cols:
        for period in periods:
            df[f'{col}_diff_{period}'] = df[col].diff(period)
    
    return df

def add_ratio_features(df):
    """添加比例特征"""
    df = df.copy()
    
    # 直接申购占比
    df['direct_purchase_ratio'] = df['direct_purchase'] / (df['total_purchase'] + 1e-6)
    
    # 余额申购占比
    df['purchase_bal_ratio'] = df['purchase_bal'] / (df['total_purchase'] + 1e-6)
    
    # 直接赎回占比
    df['direct_redeem_ratio'] = df['direct_redeem'] / (df['total_redeem'] + 1e-6)
    
    # 余额赎回占比
    df['redeem_bal_ratio'] = df['redeem_bal'] / (df['total_redeem'] + 1e-6)
    
    # 活跃用户人均申购
    df['purchase_per_user'] = df['total_purchase'] / (df['active_users'] + 1e-6)
    
    # 活跃用户人均赎回
    df['redeem_per_user'] = df['total_redeem'] / (df['active_users'] + 1e-6)
    
    return df

def add_external_features(df, interest, shibor):
    """添加外部特征（收益率、拆借利率）"""
    df = df.copy()
    
    # 合并收益率数据
    df = df.merge(interest, left_on='report_date', right_on='mfd_date', how='left')
    df = df.drop('mfd_date', axis=1)
    
    # 合并拆借利率数据
    df = df.merge(shibor, left_on='report_date', right_on='mfd_date', how='left')
    df = df.drop('mfd_date', axis=1)
    
    # 拆借利率的滞后特征
    shibor_cols = ['mfd_daily_1d', 'mfd_daily_1w', 'mfd_daily_1m', 'mfd_daily_3m']
    for col in shibor_cols:
        df[f'{col}_lag_1'] = df[col].shift(1)
        df[f'{col}_lag_7'] = df[col].shift(7)
        df[f'{col}_diff_1'] = df[col].diff(1)
    
    # 收益率的滞后特征
    df['mfd_daily_7day_annual_lag_1'] = df['mfd_daily_7day_annual'].shift(1)
    df['mfd_daily_7day_annual_diff_1'] = df['mfd_daily_7day_annual'].diff(1)
    
    return df

def add_growth_features(df, target_cols):
    """添加增长率特征"""
    df = df.copy()
    
    for col in target_cols:
        # 日增长率
        df[f'{col}_growth_1d'] = df[col].pct_change(1)
        
        # 周增长率
        df[f'{col}_growth_7d'] = df[col].pct_change(7)
        
        # 月增长率
        df[f'{col}_growth_30d'] = df[col].pct_change(30)
    
    return df

def add_cumulative_features(df, target_cols):
    """添加累计特征"""
    df = df.copy()
    
    for col in target_cols:
        # 月度累计
        df[f'{col}_monthly_cum'] = df.groupby(['year', 'month'])[col].cumsum()
    
    return df

def create_feature_matrix(data_path='../data/', for_training=True):
    """
    创建完整的特征矩阵
    
    Args:
        data_path: 数据路径
        for_training: 是否用于训练（True时包含标签，False时用于预测）
    
    Returns:
        特征矩阵DataFrame
    """
    print("开始构建特征矩阵...")
    
    # 1. 加载原始数据
    balance, interest, shibor = load_raw_data(data_path)
    print(f"  加载数据完成: {len(balance)} 条记录")
    
    # 2. 按日聚合
    daily_data = create_daily_aggregation(balance)
    print(f"  按日聚合完成: {len(daily_data)} 天")
    
    # 3. 添加时间特征
    daily_data = add_time_features(daily_data)
    print("  添加时间特征完成")
    
    # 4. 添加比例特征
    daily_data = add_ratio_features(daily_data)
    print("  添加比例特征完成")
    
    # 5. 添加滞后特征
    target_cols = ['total_purchase', 'total_redeem', 'active_users', 'net_flow']
    daily_data = add_lag_features(daily_data, target_cols, lags=[1, 2, 3, 7, 14, 30])
    print("  添加滞后特征完成")
    
    # 6. 添加滑动窗口特征
    daily_data = add_rolling_features(daily_data, target_cols, windows=[3, 7, 14, 30])
    print("  添加滑动窗口特征完成")
    
    # 7. 添加差分特征
    daily_data = add_diff_features(daily_data, target_cols, periods=[1, 7, 30])
    print("  添加差分特征完成")
    
    # 8. 添加增长率特征
    daily_data = add_growth_features(daily_data, target_cols)
    print("  添加增长率特征完成")
    
    # 9. 添加累计特征
    daily_data = add_cumulative_features(daily_data, target_cols)
    print("  添加累计特征完成")
    
    # 10. 添加外部特征
    daily_data = add_external_features(daily_data, interest, shibor)
    print("  添加外部特征完成")
    
    # 11. 添加工作日/周末特征
    daily_data = add_weekday_weekend_features(daily_data, target_cols)
    print("  添加工作日/周末特征完成")
    
    # 按日期排序
    daily_data = daily_data.sort_values('report_date').reset_index(drop=True)
    
    print(f"\n特征矩阵构建完成: {daily_data.shape[0]} 行, {daily_data.shape[1]} 列")
    
    return daily_data

def prepare_training_data(feature_df, target_col='total_purchase', test_days=30):
    """
    准备训练数据
    
    Args:
        feature_df: 特征矩阵
        target_col: 目标列名
        test_days: 测试集天数
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    # 去掉前30天（因为有30天的滞后特征）
    df = feature_df.dropna().reset_index(drop=True)
    
    # 特征列（排除目标列和日期）
    exclude_cols = ['report_date', 'total_purchase', 'total_redeem', 'net_flow',
                    'direct_purchase', 'purchase_bal', 'direct_redeem', 'redeem_bal',
                    'tbalance', 'ybalance']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 划分训练集和测试集
    train_df = df.iloc[:-test_days]
    test_df = df.iloc[-test_days:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")
    
    return X_train, y_train, X_test, y_test, feature_cols

def main():
    """主函数"""
    print("=" * 60)
    print("资金流入流出预测 - 特征工程")
    print("=" * 60)
    
    # 构建特征矩阵
    feature_df = create_feature_matrix()
    
    # 保存特征矩阵
    feature_df.to_csv('../output/feature_matrix.csv', index=False, encoding='utf-8-sig')
    print("\n特征矩阵已保存: ../output/feature_matrix.csv")
    
    # 准备申购预测的训练数据
    print("\n" + "=" * 60)
    print("准备申购预测训练数据")
    print("=" * 60)
    X_train_p, y_train_p, X_test_p, y_test_p, feature_cols = prepare_training_data(
        feature_df, target_col='total_purchase'
    )
    
    # 准备赎回预测的训练数据
    print("\n" + "=" * 60)
    print("准备赎回预测训练数据")
    print("=" * 60)
    X_train_r, y_train_r, X_test_r, y_test_r, _ = prepare_training_data(
        feature_df, target_col='total_redeem'
    )
    
    print("\n特征工程完成！")

if __name__ == '__main__':
    main()
