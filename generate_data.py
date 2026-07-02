"""
资金流入流出预测 - 模拟数据生成脚本
生成天池竞赛格式的模拟数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 设置随机种子保证可复现
np.random.seed(42)
random.seed(42)

def generate_user_profile(num_users=1000):
    """生成用户基本信息表"""
    user_ids = [f'u_{i:06d}' for i in range(1, num_users + 1)]
    
    # 性别：0-女，1-男
    sex = np.random.choice([0, 1], size=num_users, p=[0.52, 0.48])
    
    # 城市ID：1-10个城市
    city_id = np.random.randint(1, 11, size=num_users)
    
    # 星座：12个星座
    constellations = ['白羊座', '金牛座', '双子座', '巨蟹座', '狮子座', '处女座',
                      '天秤座', '天蝎座', '射手座', '摩羯座', '水瓶座', '双鱼座']
    constellation = np.random.choice(constellations, size=num_users)
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'sex': sex,
        'city_id': city_id,
        'constellation': constellation
    })
    
    return df

def generate_balance_data(user_profile, start_date='2014-03-01', end_date='2014-08-31'):
    """生成用户申购赎回数据表"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    user_ids = user_profile['user_id'].values
    
    records = []
    
    for date in dates:
        # 每天有部分用户活跃
        active_users = np.random.choice(user_ids, size=int(len(user_ids) * 0.3), replace=False)
        
        for user_id in active_users:
            # 基础申购金额（受周末、月初、节假日影响）
            base_purchase = np.random.lognormal(mean=8, sigma=1.5)
            
            # 时间因素调整
            day_factor = 1.0
            if date.weekday() >= 5:  # 周末
                day_factor = 0.7
            if date.day <= 3:  # 月初
                day_factor *= 1.3
            if date.month == 6 and date.day >= 15:  # 618活动
                day_factor *= 1.5
            
            # 直接申购
            direct_purchase_amt = base_purchase * day_factor * np.random.uniform(0.8, 1.2)
            
            # 余额宝收益申购
            purchase_bal_amt = base_purchase * day_factor * np.random.uniform(0.1, 0.3)
            
            # 总申购
            total_purchase_amt = direct_purchase_amt + purchase_bal_amt
            
            # 赎回金额
            base_redeem = np.random.lognormal(mean=7.5, sigma=1.8)
            redeem_factor = 1.0
            if date.weekday() == 0:  # 周一赎回多
                redeem_factor = 1.2
            if date.day >= 25:  # 月末赎回多
                redeem_factor *= 1.15
            
            # 直接赎回
            direct_redeem_amt = base_redeem * redeem_factor * np.random.uniform(0.8, 1.2)
            
            # 余额宝消费赎回
            redeem_bal_amt = base_redeem * redeem_factor * np.random.uniform(0.2, 0.4)
            
            # 总赎回
            total_redeem_amt = direct_redeem_amt + redeem_bal_amt
            
            # 余额
            tbalance_amt = np.random.lognormal(mean=10, sigma=1.2)
            ybalance_amt = tbalance_amt + total_purchase_amt - total_redeem_amt
            
            records.append({
                'user_id': user_id,
                'report_date': date.strftime('%Y%m%d'),
                'tBalance': round(tbalance_amt, 2),
                'yBalance': round(ybalance_amt, 2),
                'total_purchase_amt': round(total_purchase_amt, 2),
                'direct_purchase_amt': round(direct_purchase_amt, 2),
                'purchase_bal_amt': round(purchase_bal_amt, 2),
                'buy_amt': round(total_purchase_amt * np.random.uniform(0.05, 0.15), 2),
                'total_redeem_amt': round(total_redeem_amt, 2),
                'direct_redeem_amt': round(direct_redeem_amt, 2),
                'redeem_bal_amt': round(redeem_bal_amt, 2),
                'category': np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            })
    
    df = pd.DataFrame(records)
    return df

def generate_daily_interest(start_date='2014-03-01', end_date='2014-08-31'):
    """生成每日收益率表"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 基础收益率在4%-5%之间波动
    base_rate = 0.045
    rates = []
    
    for i, date in enumerate(dates):
        # 模拟收益率波动
        rate = base_rate + np.sin(i / 30) * 0.003 + np.random.normal(0, 0.001)
        rate = max(0.035, min(0.055, rate))  # 限制在3.5%-5.5%之间
        
        # 万份收益
        mfd_daily_10000 = rate / 365 * 10000
        
        rates.append({
            'mfd_date': date.strftime('%Y%m%d'),
            'mfd_daily_10000': round(mfd_daily_10000, 4),
            'mfd_daily_7day_annual': round(rate * 100, 4)
        })
    
    df = pd.DataFrame(rates)
    return df

def generate_shibor(start_date='2014-03-01', end_date='2014-08-31'):
    """生成银行间拆借利率表"""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
    
    records = []
    
    for i, date in enumerate(dates):
        # 基础利率
        base_1d = 0.025 + np.sin(i / 20) * 0.005 + np.random.normal(0, 0.002)
        base_1w = 0.030 + np.sin(i / 25) * 0.004 + np.random.normal(0, 0.0015)
        base_2w = 0.032 + np.sin(i / 28) * 0.003 + np.random.normal(0, 0.001)
        base_1m = 0.035 + np.sin(i / 30) * 0.002 + np.random.normal(0, 0.001)
        base_3m = 0.040 + np.sin(i / 35) * 0.0015 + np.random.normal(0, 0.0008)
        base_6m = 0.042 + np.sin(i / 40) * 0.001 + np.random.normal(0, 0.0005)
        base_9m = 0.043 + np.sin(i / 45) * 0.0008 + np.random.normal(0, 0.0005)
        base_1y = 0.045 + np.sin(i / 50) * 0.0005 + np.random.normal(0, 0.0003)
        
        records.append({
            'mfd_date': date.strftime('%Y%m%d'),
            'mfd_daily_1d': round(base_1d * 100, 4),
            'mfd_daily_1w': round(base_1w * 100, 4),
            'mfd_daily_2w': round(base_2w * 100, 4),
            'mfd_daily_1m': round(base_1m * 100, 4),
            'mfd_daily_3m': round(base_3m * 100, 4),
            'mfd_daily_6m': round(base_6m * 100, 4),
            'mfd_daily_9m': round(base_9m * 100, 4),
            'mfd_daily_1y': round(base_1y * 100, 4)
        })
    
    df = pd.DataFrame(records)
    return df

def generate_submission_template():
    """生成提交格式模板"""
    dates = pd.date_range(start='2014-09-01', end='2014-09-30', freq='D')
    
    df = pd.DataFrame({
        'report_date': [d.strftime('%Y%m%d') for d in dates],
        'purchase': 0,
        'redeem': 0
    })
    
    return df

def main():
    print("开始生成模拟数据...")
    
    # 1. 生成用户基本信息
    print("生成用户基本信息表...")
    user_profile = generate_user_profile(num_users=1000)
    user_profile.to_csv('../data/user_profile_table.csv', index=False, encoding='utf-8-sig')
    print(f"  生成 {len(user_profile)} 条用户记录")
    
    # 2. 生成用户申购赎回数据
    print("生成用户申购赎回数据表...")
    balance_data = generate_balance_data(user_profile)
    balance_data.to_csv('../data/user_balance_table.csv', index=False, encoding='utf-8-sig')
    print(f"  生成 {len(balance_data)} 条余额记录")
    
    # 3. 生成每日收益率表
    print("生成每日收益率表...")
    daily_interest = generate_daily_interest()
    daily_interest.to_csv('../data/mfd_day_share_interest.csv', index=False, encoding='utf-8-sig')
    print(f"  生成 {len(daily_interest)} 条收益率记录")
    
    # 4. 生成银行间拆借利率表
    print("生成银行间拆借利率表...")
    shibor = generate_shibor()
    shibor.to_csv('../data/mfd_bank_shibor.csv', index=False, encoding='utf-8-sig')
    print(f"  生成 {len(shibor)} 条拆借利率记录")
    
    # 5. 生成提交模板
    print("生成提交格式模板...")
    submission = generate_submission_template()
    submission.to_csv('../data/comp_predict_table.csv', index=False, encoding='utf-8-sig')
    print(f"  生成 {len(submission)} 条预测模板")
    
    print("\n数据生成完成！所有文件保存在 ../data/ 目录下")

if __name__ == '__main__':
    main()
