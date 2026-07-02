"""
资金流入流出预测 - 数据探索与分析（EDA）
对竞赛数据进行探索性分析，了解数据分布、特征关系等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(data_path='../data/'):
    """加载所有数据"""
    print("=" * 60)
    print("数据加载中...")
    print("=" * 60)
    
    # 用户基本信息
    user_profile = pd.read_csv(data_path + 'user_profile_table.csv')
    print(f"\n1. 用户基本信息表: {user_profile.shape}")
    print(f"   字段: {list(user_profile.columns)}")
    
    # 用户申购赎回数据
    balance = pd.read_csv(data_path + 'user_balance_table.csv')
    print(f"\n2. 用户申购赎回数据表: {balance.shape}")
    print(f"   字段: {list(balance.columns)}")
    
    # 每日收益率
    interest = pd.read_csv(data_path + 'mfd_day_share_interest.csv')
    print(f"\n3. 每日收益率表: {interest.shape}")
    print(f"   字段: {list(interest.columns)}")
    
    # 银行间拆借利率
    shibor = pd.read_csv(data_path + 'mfd_bank_shibor.csv')
    print(f"\n4. 银行间拆借利率表: {shibor.shape}")
    print(f"   字段: {list(shibor.columns)}")
    
    return user_profile, balance, interest, shibor

def analyze_user_profile(user_profile):
    """分析用户基本信息"""
    print("\n" + "=" * 60)
    print("用户基本信息分析")
    print("=" * 60)
    
    # 基本统计
    print(f"\n用户总数: {len(user_profile)}")
    
    # 性别分布
    print("\n性别分布:")
    sex_dist = user_profile['sex'].value_counts()
    print(sex_dist)
    
    # 城市分布
    print("\n城市分布 (Top 10):")
    city_dist = user_profile['city_id'].value_counts().head(10)
    print(city_dist)
    
    # 星座分布
    print("\n星座分布:")
    const_dist = user_profile['constellation'].value_counts()
    print(const_dist)
    
    # 绘制性别分布图
    plt.figure(figsize=(8, 6))
    sex_dist.plot(kind='pie', autopct='%1.1f%%', labels=['女', '男'])
    plt.title('用户性别分布')
    plt.savefig('../output/user_sex_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n性别分布图已保存: ../output/user_sex_distribution.png")

def analyze_balance_data(balance):
    """分析用户申购赎回数据"""
    print("\n" + "=" * 60)
    print("用户申购赎回数据分析")
    print("=" * 60)
    
    # 转换日期格式
    balance['report_date'] = pd.to_datetime(balance['report_date'], format='%Y%m%d')
    
    # 基本信息
    print(f"\n时间范围: {balance['report_date'].min()} ~ {balance['report_date'].max()}")
    print(f"用户数量: {balance['user_id'].nunique()}")
    print(f"记录总数: {len(balance)}")
    
    # 按日期统计每日总申购和总赎回
    daily_stats = balance.groupby('report_date').agg({
        'total_purchase_amt': 'sum',
        'total_redeem_amt': 'sum',
        'user_id': 'nunique'
    }).reset_index()
    daily_stats.columns = ['report_date', 'total_purchase', 'total_redeem', 'active_users']
    
    print(f"\n每日统计信息:")
    print(daily_stats.describe())
    
    # 绘制每日申购赎回趋势
    plt.figure(figsize=(15, 8))
    plt.plot(daily_stats['report_date'], daily_stats['total_purchase'], label='总申购金额', alpha=0.8)
    plt.plot(daily_stats['report_date'], daily_stats['total_redeem'], label='总赎回金额', alpha=0.8)
    plt.title('每日资金流入流出趋势')
    plt.xlabel('日期')
    plt.ylabel('金额')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../output/daily_purchase_redeem_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n每日申购赎回趋势图已保存: ../output/daily_purchase_redeem_trend.png")
    
    # 绘制净流入趋势
    daily_stats['net_flow'] = daily_stats['total_purchase'] - daily_stats['total_redeem']
    plt.figure(figsize=(15, 6))
    plt.bar(daily_stats['report_date'], daily_stats['net_flow'], alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('每日资金净流入趋势')
    plt.xlabel('日期')
    plt.ylabel('净流入金额')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../output/daily_net_flow_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("每日净流入趋势图已保存: ../output/daily_net_flow_trend.png")
    
    # 按星期分析
    daily_stats['weekday'] = daily_stats['report_date'].dt.weekday
    weekday_stats = daily_stats.groupby('weekday').agg({
        'total_purchase': 'mean',
        'total_redeem': 'mean'
    }).reset_index()
    weekday_stats['weekday_name'] = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    
    print("\n按星期统计 (平均金额):")
    print(weekday_stats[['weekday_name', 'total_purchase', 'total_redeem']])
    
    # 绘制星期分布
    plt.figure(figsize=(10, 6))
    x = np.arange(len(weekday_stats))
    width = 0.35
    plt.bar(x - width/2, weekday_stats['total_purchase'], width, label='平均申购')
    plt.bar(x + width/2, weekday_stats['total_redeem'], width, label='平均赎回')
    plt.xticks(x, weekday_stats['weekday_name'])
    plt.title('星期对资金流动的影响')
    plt.ylabel('平均金额')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../output/weekday_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("星期影响图已保存: ../output/weekday_effect.png")
    
    # 按月分析
    daily_stats['month'] = daily_stats['report_date'].dt.month
    monthly_stats = daily_stats.groupby('month').agg({
        'total_purchase': 'sum',
        'total_redeem': 'sum'
    }).reset_index()
    
    print("\n按月统计 (总金额):")
    print(monthly_stats)
    
    return daily_stats

def analyze_interest(interest):
    """分析收益率数据"""
    print("\n" + "=" * 60)
    print("收益率数据分析")
    print("=" * 60)
    
    interest['mfd_date'] = pd.to_datetime(interest['mfd_date'], format='%Y%m%d')
    
    print(f"\n时间范围: {interest['mfd_date'].min()} ~ {interest['mfd_date'].max()}")
    print("\n收益率统计信息:")
    print(interest.describe())
    
    # 绘制收益率趋势
    plt.figure(figsize=(15, 6))
    plt.plot(interest['mfd_date'], interest['mfd_daily_7day_annual'])
    plt.title('7日年化收益率趋势')
    plt.xlabel('日期')
    plt.ylabel('收益率 (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../output/interest_rate_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n收益率趋势图已保存: ../output/interest_rate_trend.png")

def analyze_shibor(shibor):
    """分析银行间拆借利率"""
    print("\n" + "=" * 60)
    print("银行间拆借利率分析")
    print("=" * 60)
    
    shibor['mfd_date'] = pd.to_datetime(shibor['mfd_date'], format='%Y%m%d')
    
    print(f"\n时间范围: {shibor['mfd_date'].min()} ~ {shibor['mfd_date'].max()}")
    print("\n拆借利率统计信息:")
    print(shibor.describe())
    
    # 绘制主要期限的拆借利率趋势
    plt.figure(figsize=(15, 8))
    plt.plot(shibor['mfd_date'], shibor['mfd_daily_1d'], label='隔夜', alpha=0.8)
    plt.plot(shibor['mfd_date'], shibor['mfd_daily_1w'], label='1周', alpha=0.8)
    plt.plot(shibor['mfd_date'], shibor['mfd_daily_1m'], label='1个月', alpha=0.8)
    plt.plot(shibor['mfd_date'], shibor['mfd_daily_3m'], label='3个月', alpha=0.8)
    plt.title('银行间拆借利率趋势')
    plt.xlabel('日期')
    plt.ylabel('利率 (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../output/shibor_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n拆借利率趋势图已保存: ../output/shibor_trend.png")

def analyze_correlation(daily_stats, interest, shibor):
    """分析相关性"""
    print("\n" + "=" * 60)
    print("相关性分析")
    print("=" * 60)
    
    # 合并数据
    merged = daily_stats.copy()
    merged = merged.merge(interest, left_on='report_date', right_on='mfd_date', how='left')
    merged = merged.merge(shibor, left_on='report_date', right_on='mfd_date', how='left')
    
    # 选择数值列计算相关性
    numeric_cols = ['total_purchase', 'total_redeem', 'active_users', 'net_flow',
                    'mfd_daily_10000', 'mfd_daily_7day_annual',
                    'mfd_daily_1d', 'mfd_daily_1w', 'mfd_daily_1m', 'mfd_daily_3m']
    
    corr_data = merged[numeric_cols].corr()
    
    print("\n相关性矩阵:")
    print(corr_data.round(3))
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig('../output/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n相关性热力图已保存: ../output/correlation_heatmap.png")

def main():
    """主函数"""
    print("资金流入流出预测 - 探索性数据分析")
    print("=" * 60)
    
    # 1. 加载数据
    user_profile, balance, interest, shibor = load_data()
    
    # 2. 用户信息分析
    analyze_user_profile(user_profile)
    
    # 3. 申购赎回数据分析
    daily_stats = analyze_balance_data(balance)
    
    # 4. 收益率分析
    analyze_interest(interest)
    
    # 5. 拆借利率分析
    analyze_shibor(shibor)
    
    # 6. 相关性分析
    analyze_correlation(daily_stats, interest, shibor)
    
    print("\n" + "=" * 60)
    print("数据分析完成！所有图表保存在 ../output/ 目录下")
    print("=" * 60)

if __name__ == '__main__':
    main()
