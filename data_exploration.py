"""
资金流入流出预测 - 数据探索与分析（EDA）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_PATH = '../data/Purchase Redemption Data/'
OUTPUT_PATH = '../output/'

def load_data():
    """加载所有数据"""
    print("正在加载数据...")
    
    # 用户基本信息
    user_profile = pd.read_csv(os.path.join(DATA_PATH, 'user_profile_table.csv'))
    print(f"用户信息表: {user_profile.shape[0]} 行, {user_profile.shape[1]} 列")
    
    # 用户余额表
    user_balance = pd.read_csv(os.path.join(DATA_PATH, 'user_balance_table.csv'))
    print(f"用户余额表: {user_balance.shape[0]} 行, {user_balance.shape[1]} 列")
    
    # 每日收益率
    daily_interest = pd.read_csv(os.path.join(DATA_PATH, 'mfd_day_share_interest.csv'))
    print(f"收益率表: {daily_interest.shape[0]} 行, {daily_interest.shape[1]} 列")
    
    # 银行间拆借利率
    bank_shibor = pd.read_csv(os.path.join(DATA_PATH, 'mfd_bank_shibor.csv'))
    print(f"拆借利率表: {bank_shibor.shape[0]} 行, {bank_shibor.shape[1]} 列")
    
    return user_profile, user_balance, daily_interest, bank_shibor

def analyze_user_profile(user_profile):
    """分析用户基本信息"""
    print("\n" + "="*50)
    print("用户基本信息分析")
    print("="*50)
    
    print("\n前5行数据:")
    print(user_profile.head())
    
    print("\n数据统计:")
    print(user_profile.describe(include='all'))
    
    print("\n性别分布:")
    print(user_profile['sex'].value_counts())
    
    print("\n城市分布 (Top 10):")
    print(user_profile['city'].value_counts().head(10))
    
    print("\n星座分布:")
    print(user_profile['constellation'].value_counts())
    
    # 性别分布饼图
    plt.figure(figsize=(10, 6))
    user_profile['sex'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['男', '女'])
    plt.title('用户性别分布')
    plt.savefig(os.path.join(OUTPUT_PATH, 'user_sex_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 城市分布柱状图
    plt.figure(figsize=(12, 6))
    city_counts = user_profile['city'].value_counts().head(15)
    city_counts.plot(kind='bar')
    plt.title('用户城市分布 (Top 15)')
    plt.xlabel('城市ID')
    plt.ylabel('用户数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'user_city_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

def analyze_daily_flow(user_balance):
    """分析每日资金流入流出"""
    print("\n" + "="*50)
    print("每日资金流入流出分析")
    print("="*50)
    
    # 按日期汇总
    daily_flow = user_balance.groupby('report_date').agg({
        'total_purchase_amt': 'sum',
        'total_redeem_amt': 'sum',
        'user_id': 'nunique'
    }).reset_index()
    
    daily_flow.columns = ['report_date', 'total_purchase', 'total_redeem', 'active_users']
    
    # 转换日期格式
    daily_flow['report_date'] = pd.to_datetime(daily_flow['report_date'], format='%Y%m%d')
    daily_flow = daily_flow.sort_values('report_date')
    
    print(f"\n数据时间范围: {daily_flow['report_date'].min()} 至 {daily_flow['report_date'].max()}")
    print(f"总天数: {len(daily_flow)} 天")
    
    print("\n资金流动统计:")
    print(f"平均日申购金额: {daily_flow['total_purchase'].mean():,.2f}")
    print(f"平均日赎回金额: {daily_flow['total_redeem'].mean():,.2f}")
    print(f"平均日活跃用户: {daily_flow['active_users'].mean():,.0f}")
    
    # 添加时间特征
    daily_flow['year'] = daily_flow['report_date'].dt.year
    daily_flow['month'] = daily_flow['report_date'].dt.month
    daily_flow['day'] = daily_flow['report_date'].dt.day
    daily_flow['weekday'] = daily_flow['report_date'].dt.weekday
    daily_flow['week'] = daily_flow['report_date'].dt.isocalendar().week
    daily_flow['is_weekend'] = daily_flow['weekday'].isin([5, 6]).astype(int)
    
    # 绘制资金流动趋势图
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    axes[0].plot(daily_flow['report_date'], daily_flow['total_purchase'], label='申购金额', alpha=0.7)
    axes[0].plot(daily_flow['report_date'], daily_flow['total_redeem'], label='赎回金额', alpha=0.7)
    axes[0].set_title('每日资金流入流出趋势')
    axes[0].set_ylabel('金额')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(daily_flow['report_date'], daily_flow['active_users'], color='green', alpha=0.7)
    axes[1].set_title('每日活跃用户数趋势')
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('用户数')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'daily_flow_trend.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 月度分析
    monthly_flow = daily_flow.groupby('month').agg({
        'total_purchase': 'mean',
        'total_redeem': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(monthly_flow))
    width = 0.35
    
    plt.bar(x - width/2, monthly_flow['total_purchase'], width, label='平均日申购')
    plt.bar(x + width/2, monthly_flow['total_redeem'], width, label='平均日赎回')
    
    plt.xlabel('月份')
    plt.ylabel('金额')
    plt.title('各月平均日资金流动对比')
    plt.xticks(x, monthly_flow['month'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'monthly_flow_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 周内分析
    weekday_flow = daily_flow.groupby('weekday').agg({
        'total_purchase': 'mean',
        'total_redeem': 'mean'
    }).reset_index()
    
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(7)
    width = 0.35
    
    plt.bar(x - width/2, weekday_flow['total_purchase'], width, label='平均日申购')
    plt.bar(x + width/2, weekday_flow['total_redeem'], width, label='平均日赎回')
    
    plt.xlabel('星期')
    plt.ylabel('金额')
    plt.title('周内各天平均资金流动对比')
    plt.xticks(x, weekday_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'weekday_flow_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return daily_flow

def analyze_interest_rates(daily_interest, bank_shibor):
    """分析利率数据"""
    print("\n" + "="*50)
    print("利率数据分析")
    print("="*50)
    
    # 处理余额宝收益率
    daily_interest['mfd_date'] = pd.to_datetime(daily_interest['mfd_date'], format='%Y%m%d')
    daily_interest = daily_interest.sort_values('mfd_date')
    
    print(f"\n余额宝收益率时间范围: {daily_interest['mfd_date'].min()} 至 {daily_interest['mfd_date'].max()}")
    print(f"万份收益范围: {daily_interest['mfd_daily_yield'].min():.4f} - {daily_interest['mfd_daily_yield'].max():.4f}")
    print(f"7日年化收益率范围: {daily_interest['mfd_7daily_yield'].min():.4f}% - {daily_interest['mfd_7daily_yield'].max():.4f}%")
    
    # 处理银行拆借利率
    bank_shibor['mfd_date'] = pd.to_datetime(bank_shibor['mfd_date'], format='%Y%m%d')
    bank_shibor = bank_shibor.sort_values('mfd_date')
    
    print(f"\n银行拆借利率时间范围: {bank_shibor['mfd_date'].min()} 至 {bank_shibor['mfd_date'].max()}")
    
    # 绘制收益率趋势
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    axes[0].plot(daily_interest['mfd_date'], daily_interest['mfd_daily_yield'], label='万份收益', color='blue')
    axes[0].set_title('余额宝万份收益趋势')
    axes[0].set_ylabel('万份收益（元）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(daily_interest['mfd_date'], daily_interest['mfd_7daily_yield'], label='7日年化收益率', color='orange')
    axes[1].set_title('余额宝7日年化收益率趋势')
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('收益率（%）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'yield_trend.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 绘制银行拆借利率
    plt.figure(figsize=(15, 8))
    
    for col in ['Interest_O_N', 'Interest_1_W', 'Interest_1_M', 'Interest_3_M', 'Interest_6_M']:
        plt.plot(bank_shibor['mfd_date'], bank_shibor[col], label=col, alpha=0.7)
    
    plt.title('银行间拆借利率趋势')
    plt.xlabel('日期')
    plt.ylabel('利率（%）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'shibor_trend.png'), dpi=150, bbox_inches='tight')
    plt.close()

def analyze_correlation(daily_flow, daily_interest, bank_shibor):
    """分析相关性"""
    print("\n" + "="*50)
    print("相关性分析")
    print("="*50)
    
    # 合并数据
    daily_flow = daily_flow.copy()
    daily_flow['mfd_date'] = daily_flow['report_date']
    
    # 合并收益率
    merged = pd.merge(daily_flow, daily_interest, on='mfd_date', how='left')
    
    # 合并拆借利率
    merged = pd.merge(merged, bank_shibor, on='mfd_date', how='left')
    
    # 选择数值列计算相关性
    numeric_cols = ['total_purchase', 'total_redeem', 'active_users', 
                    'mfd_daily_yield', 'mfd_7daily_yield',
                    'Interest_O_N', 'Interest_1_W', 'Interest_1_M', 'Interest_3_M']
    
    corr_matrix = merged[numeric_cols].corr()
    
    print("\n相关性矩阵:")
    print(corr_matrix.round(3))
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f', square=True)
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("="*60)
    print("资金流入流出预测 - 数据探索与分析")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 加载数据
    user_profile, user_balance, daily_interest, bank_shibor = load_data()
    
    # 用户信息分析
    analyze_user_profile(user_profile)
    
    # 每日资金流动分析
    daily_flow = analyze_daily_flow(user_balance)
    
    # 利率分析
    analyze_interest_rates(daily_interest, bank_shibor)
    
    # 相关性分析
    analyze_correlation(daily_flow, daily_interest, bank_shibor)
    
    print("\n" + "="*60)
    print("数据分析完成！图表已保存到 output 目录")
    print("="*60)

if __name__ == '__main__':
    main()
