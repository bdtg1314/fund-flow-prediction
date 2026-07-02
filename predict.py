"""
资金流入流出预测 - 最终预测与结果生成
生成竞赛提交格式的预测结果
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 尝试导入XGBoost和LightGBM
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from feature_engineering import (
    load_raw_data, create_daily_aggregation,
    add_time_features, add_lag_features, add_rolling_features,
    add_diff_features, add_ratio_features, add_external_features,
    add_growth_features, add_cumulative_features, add_weekday_weekend_features
)


def generate_future_dates(start_date='2014-09-01', end_date='2014-09-30'):
    """生成预测日期范围"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return pd.DataFrame({'report_date': dates})


def build_features_for_prediction(historical_data, future_dates, interest, shibor):
    """
    为预测日期构建特征
    使用递归预测的方式，逐步预测并更新特征
    """
    # 合并历史数据和未来日期
    all_dates = pd.concat([
        historical_data[['report_date']],
        future_dates
    ], ignore_index=True)
    
    # 构建完整的特征框架
    df = all_dates.copy()
    
    # 添加时间特征
    df = add_time_features(df)
    
    # 合并历史的真实值
    df = df.merge(historical_data[['report_date', 'total_purchase', 'total_redeem', 
                                     'active_users', 'net_flow',
                                     'direct_purchase', 'purchase_bal',
                                     'direct_redeem', 'redeem_bal',
                                     'tbalance', 'ybalance']], 
                  on='report_date', how='left')
    
    # 添加比例特征（需要先有基础数据）
    df = add_ratio_features(df)
    
    # 添加外部特征
    df = add_external_features(df, interest, shibor)
    
    # 递归填充预测值
    target_cols = ['total_purchase', 'total_redeem', 'active_users', 'net_flow']
    
    # 找到第一个预测日期的索引
    first_pred_idx = df[df['report_date'] == future_dates['report_date'].iloc[0]].index[0]
    
    # 逐步预测每一天
    for i in range(first_pred_idx, len(df)):
        # 对于预测日期，用前一天的预测值来计算滞后和滑窗特征
        # 这里简化处理，直接使用历史数据的统计特征
        
        # 填充活跃用户（用历史均值）
        if pd.isna(df.loc[i, 'active_users']):
            df.loc[i, 'active_users'] = df.loc[:i-1, 'active_users'].mean()
        
        # 填充净流入（先设为0，后面再计算）
        if pd.isna(df.loc[i, 'net_flow']):
            df.loc[i, 'net_flow'] = 0
    
    # 添加滞后特征（使用历史数据）
    df = add_lag_features(df, target_cols, lags=[1, 2, 3, 7, 14, 30])
    
    # 添加滑动窗口特征
    df = add_rolling_features(df, target_cols, windows=[3, 7, 14, 30])
    
    # 添加差分特征
    df = add_diff_features(df, target_cols, periods=[1, 7, 30])
    
    # 添加增长率特征
    df = add_growth_features(df, target_cols)
    
    # 添加累计特征
    df = add_cumulative_features(df, target_cols)
    
    # 添加工作日/周末特征
    df = add_weekday_weekend_features(df, target_cols)
    
    return df


def train_final_model(X_train, y_train, model_type='random_forest'):
    """训练最终模型"""
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    elif model_type == 'gbdt':
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.9, random_state=42
        )
    elif model_type == 'xgboost' and HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.9, random_state=42, n_jobs=-1
        )
    elif model_type == 'lightgbm' and HAS_LIGHTGBM:
        model = LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.9, random_state=42, n_jobs=-1, verbose=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    return model


def predict_future(feature_df, model_purchase, model_redeem, future_dates):
    """预测未来的申购和赎回"""
    # 特征列
    exclude_cols = ['report_date', 'total_purchase', 'total_redeem', 'net_flow',
                    'direct_purchase', 'purchase_bal', 'direct_redeem', 'redeem_bal',
                    'tbalance', 'ybalance']
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
    
    # 获取预测日期的特征
    pred_mask = feature_df['report_date'].isin(future_dates['report_date'])
    X_pred = feature_df[pred_mask][feature_cols].fillna(0)
    
    # 预测
    purchase_pred = model_purchase.predict(X_pred)
    redeem_pred = model_redeem.predict(X_pred)
    
    # 确保预测值为正
    purchase_pred = np.maximum(purchase_pred, 0)
    redeem_pred = np.maximum(redeem_pred, 0)
    
    # 构建结果
    result = pd.DataFrame({
        'report_date': future_dates['report_date'].dt.strftime('%Y%m%d'),
        'purchase': purchase_pred.round(2),
        'redeem': redeem_pred.round(2)
    })
    
    return result


def ensemble_predict(models_purchase, models_redeem, X_pred, weights=None):
    """模型集成预测"""
    if weights is None:
        weights = [1/len(models_purchase)] * len(models_purchase)
    
    purchase_preds = []
    redeem_preds = []
    
    for model in models_purchase:
        pred = model.predict(X_pred)
        purchase_preds.append(pred)
    
    for model in models_redeem:
        pred = model.predict(X_pred)
        redeem_preds.append(pred)
    
    # 加权平均
    purchase_ensemble = np.average(purchase_preds, axis=0, weights=weights)
    redeem_ensemble = np.average(redeem_preds, axis=0, weights=weights)
    
    return purchase_ensemble, redeem_ensemble


def main():
    """主函数"""
    print("=" * 60)
    print("资金流入流出预测 - 最终预测生成")
    print("=" * 60)
    
    # 1. 加载原始数据
    print("\n步骤1: 加载原始数据")
    balance, interest, shibor = load_raw_data()
    print(f"  加载 {len(balance)} 条用户余额记录")
    
    # 2. 按日聚合
    print("\n步骤2: 按日聚合数据")
    daily_data = create_daily_aggregation(balance)
    print(f"  聚合为 {len(daily_data)} 天的数据")
    
    # 3. 生成预测日期
    print("\n步骤3: 生成预测日期范围")
    future_dates = generate_future_dates('2014-09-01', '2014-09-30')
    print(f"  预测 {len(future_dates)} 天 (2014-09-01 ~ 2014-09-30)")
    
    # 4. 构建特征
    print("\n步骤4: 构建特征矩阵")
    feature_df = build_features_for_prediction(daily_data, future_dates, interest, shibor)
    print(f"  特征矩阵: {feature_df.shape[0]} 行, {feature_df.shape[1]} 列")
    
    # 5. 准备训练数据
    print("\n步骤5: 准备训练数据")
    train_mask = feature_df['report_date'] < '2014-09-01'
    train_df = feature_df[train_mask].dropna()
    
    exclude_cols = ['report_date', 'total_purchase', 'total_redeem', 'net_flow',
                    'direct_purchase', 'purchase_bal', 'direct_redeem', 'redeem_bal',
                    'tbalance', 'ybalance']
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train_purchase = train_df['total_purchase']
    y_train_redeem = train_df['total_redeem']
    
    print(f"  训练样本: {len(X_train)}")
    print(f"  特征数量: {len(feature_cols)}")
    
    # 6. 训练多个模型
    print("\n步骤6: 训练预测模型")
    
    # 申购预测模型
    print("  训练申购预测模型...")
    model_rf_p = train_final_model(X_train, y_train_purchase, 'random_forest')
    model_gbdt_p = train_final_model(X_train, y_train_purchase, 'gbdt')
    
    models_purchase = [model_rf_p, model_gbdt_p]
    if HAS_XGBOOST:
        model_xgb_p = train_final_model(X_train, y_train_purchase, 'xgboost')
        models_purchase.append(model_xgb_p)
    if HAS_LIGHTGBM:
        model_lgb_p = train_final_model(X_train, y_train_purchase, 'lightgbm')
        models_purchase.append(model_lgb_p)
    
    # 赎回预测模型
    print("  训练赎回预测模型...")
    model_rf_r = train_final_model(X_train, y_train_redeem, 'random_forest')
    model_gbdt_r = train_final_model(X_train, y_train_redeem, 'gbdt')
    
    models_redeem = [model_rf_r, model_gbdt_r]
    if HAS_XGBOOST:
        model_xgb_r = train_final_model(X_train, y_train_redeem, 'xgboost')
        models_redeem.append(model_xgb_r)
    if HAS_LIGHTGBM:
        model_lgb_r = train_final_model(X_train, y_train_redeem, 'lightgbm')
        models_redeem.append(model_lgb_r)
    
    print(f"  共训练 {len(models_purchase)} 个申购模型, {len(models_redeem)} 个赎回模型")
    
    # 7. 生成预测
    print("\n步骤7: 生成预测结果")
    pred_mask = feature_df['report_date'] >= '2014-09-01'
    X_pred = feature_df[pred_mask][feature_cols].fillna(0)
    
    # 集成预测
    purchase_pred, redeem_pred = ensemble_predict(models_purchase, models_redeem, X_pred)
    
    # 确保预测值为正
    purchase_pred = np.maximum(purchase_pred, 0)
    redeem_pred = np.maximum(redeem_pred, 0)
    
    # 8. 构建提交结果
    print("\n步骤8: 生成提交格式文件")
    submission = pd.DataFrame({
        'report_date': future_dates['report_date'].dt.strftime('%Y%m%d'),
        'purchase': purchase_pred.round(2),
        'redeem': redeem_pred.round(2)
    })
    
    # 保存结果
    submission.to_csv('../output/submission.csv', index=False, encoding='utf-8-sig')
    print(f"  提交文件已保存: ../output/submission.csv")
    
    # 打印预测结果概览
    print("\n预测结果概览:")
    print(submission.head(10).to_string(index=False))
    print(f"\n... 共 {len(submission)} 条记录")
    print(f"\n申购预测统计:")
    print(f"  均值: {submission['purchase'].mean():.2f}")
    print(f"  最大值: {submission['purchase'].max():.2f}")
    print(f"  最小值: {submission['purchase'].min():.2f}")
    print(f"\n赎回预测统计:")
    print(f"  均值: {submission['redeem'].mean():.2f}")
    print(f"  最大值: {submission['redeem'].max():.2f}")
    print(f"  最小值: {submission['redeem'].min():.2f}")
    
    print("\n" + "=" * 60)
    print("预测生成完成！")
    print("=" * 60)
    
    return submission


if __name__ == '__main__':
    main()
