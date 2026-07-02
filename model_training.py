"""
资金流入流出预测 - 模型训练与调优
实现多种机器学习算法，进行模型对比和参数调优
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入XGBoost和LightGBM
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: 未安装XGBoost，将跳过XGBoost模型")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("警告: 未安装LightGBM，将跳过LightGBM模型")

from feature_engineering import create_feature_matrix, prepare_training_data


def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    
    print(f"\n{model_name} 评估结果:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def calculate_competition_score(y_true, y_pred):
    """
    计算竞赛得分
    误差为0时得10分，误差>0.3时得0分，线性插值
    """
    errors = np.abs(y_true - y_pred) / (y_true + 1e-6)
    scores = np.maximum(0, 10 - errors / 0.03 * 10)  # 假设3%误差对应0分
    scores = np.clip(scores, 0, 10)
    return np.mean(scores)


def get_models():
    """获取所有待比较的模型"""
    models = {
        '线性回归': LinearRegression(),
        '岭回归': Ridge(alpha=1.0),
        'Lasso回归': Lasso(alpha=0.1),
        '决策树': DecisionTreeRegressor(max_depth=10, random_state=42),
        '随机森林': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'GBDT': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
    
    return models


def train_and_evaluate_models(X_train, y_train, X_test, y_test, target_name='申购'):
    """训练并评估所有模型"""
    print(f"\n{'='*60}")
    print(f"{target_name}预测 - 模型训练与评估")
    print(f"{'='*60}")
    
    # 数据标准化（对SVR等模型需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = get_models()
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # SVR需要标准化数据
        if name == 'SVR':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # 评估
        metrics = evaluate_model(y_test, y_pred, name)
        
        # 计算竞赛风格得分
        comp_score = calculate_competition_score(y_test, y_pred)
        metrics['competition_score'] = comp_score
        print(f"  竞赛风格得分: {comp_score:.2f}/10")
        
        results.append(metrics)
        trained_models[name] = model
    
    # 结果对比
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    print(f"\n{'='*60}")
    print(f"{target_name}预测 - 模型对比 (按RMSE排序)")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    return results_df, trained_models, scaler


def plot_model_comparison(results_df, target_name='申购'):
    """绘制模型对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSE对比
    axes[0, 0].barh(results_df['model'], results_df['RMSE'], color='skyblue')
    axes[0, 0].set_title(f'{target_name}预测 - RMSE对比')
    axes[0, 0].set_xlabel('RMSE')
    
    # MAE对比
    axes[0, 1].barh(results_df['model'], results_df['MAE'], color='lightgreen')
    axes[0, 1].set_title(f'{target_name}预测 - MAE对比')
    axes[0, 1].set_xlabel('MAE')
    
    # R²对比
    axes[1, 0].barh(results_df['model'], results_df['R2'], color='salmon')
    axes[1, 0].set_title(f'{target_name}预测 - R²对比')
    axes[1, 0].set_xlabel('R²')
    
    # MAPE对比
    axes[1, 1].barh(results_df['model'], results_df['MAPE'], color='gold')
    axes[1, 1].set_title(f'{target_name}预测 - MAPE对比 (%)')
    axes[1, 1].set_xlabel('MAPE (%)')
    
    plt.tight_layout()
    plt.savefig(f'../output/{target_name}_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n模型对比图已保存: ../output/{target_name}_model_comparison.png")


def plot_predictions(y_test, y_pred, model_name, target_name='申购'):
    """绘制预测值与真实值对比图"""
    plt.figure(figsize=(15, 6))
    
    x = range(len(y_test))
    plt.plot(x, y_test.values, label='真实值', alpha=0.8, linewidth=2)
    plt.plot(x, y_pred, label='预测值', alpha=0.8, linewidth=2, linestyle='--')
    
    plt.title(f'{target_name}预测 - {model_name} 预测结果对比')
    plt.xlabel('天数')
    plt.ylabel('金额')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../output/{target_name}_{model_name}_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_cols, model_name, top_n=20, target_name='申购'):
    """绘制特征重要性"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_cols[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.title(f'{target_name}预测 - {model_name} 特征重要性 (Top {top_n})')
        plt.xlabel('重要性')
        
        plt.tight_layout()
        plt.savefig(f'../output/{target_name}_{model_name}_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"特征重要性图已保存: ../output/{target_name}_{model_name}_feature_importance.png")


def tune_best_model(X_train, y_train, model_name='随机森林', target_name='申购'):
    """对最佳模型进行超参数调优"""
    print(f"\n{'='*60}")
    print(f"{target_name}预测 - {model_name} 超参数调优")
    print(f"{'='*60}")
    
    if model_name == '随机森林':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    elif model_name == 'GBDT':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = GradientBoostingRegressor(random_state=42)
    
    elif model_name == 'XGBoost' and HAS_XGBOOST:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    elif model_name == 'LightGBM' and HAS_LIGHTGBM:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    
    else:
        print(f"不支持对 {model_name} 进行调优")
        return None
    
    # 网格搜索
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分 (RMSE): {np.sqrt(-grid_search.best_score_):.2f}")
    
    return grid_search.best_estimator_


def main():
    """主函数"""
    print("=" * 60)
    print("资金流入流出预测 - 模型训练与评估")
    print("=" * 60)
    
    # 1. 构建特征矩阵
    print("\n步骤1: 构建特征矩阵")
    feature_df = create_feature_matrix()
    
    # 2. 准备申购预测数据
    print("\n步骤2: 准备申购预测训练数据")
    X_train_p, y_train_p, X_test_p, y_test_p, feature_cols = prepare_training_data(
        feature_df, target_col='total_purchase', test_days=30
    )
    
    # 3. 训练并评估申购预测模型
    results_purchase, models_purchase, scaler_purchase = train_and_evaluate_models(
        X_train_p, y_train_p, X_test_p, y_test_p, target_name='申购'
    )
    
    # 4. 绘制申购模型对比
    plot_model_comparison(results_purchase, target_name='申购')
    
    # 5. 绘制最佳模型的预测结果
    best_purchase_model = results_purchase.iloc[0]['model']
    if best_purchase_model == 'SVR':
        y_pred_best_p = models_purchase[best_purchase_model].predict(
            scaler_purchase.transform(X_test_p)
        )
    else:
        y_pred_best_p = models_purchase[best_purchase_model].predict(X_test_p)
    
    plot_predictions(y_test_p, y_pred_best_p, best_purchase_model, target_name='申购')
    
    # 6. 绘制特征重要性
    if hasattr(models_purchase[best_purchase_model], 'feature_importances_'):
        plot_feature_importance(
            models_purchase[best_purchase_model], feature_cols,
            best_purchase_model, target_name='申购'
        )
    
    # 7. 准备赎回预测数据
    print("\n步骤3: 准备赎回预测训练数据")
    X_train_r, y_train_r, X_test_r, y_test_r, _ = prepare_training_data(
        feature_df, target_col='total_redeem', test_days=30
    )
    
    # 8. 训练并评估赎回预测模型
    results_redeem, models_redeem, scaler_redeem = train_and_evaluate_models(
        X_train_r, y_train_r, X_test_r, y_test_r, target_name='赎回'
    )
    
    # 9. 绘制赎回模型对比
    plot_model_comparison(results_redeem, target_name='赎回')
    
    # 10. 绘制最佳模型的预测结果
    best_redeem_model = results_redeem.iloc[0]['model']
    if best_redeem_model == 'SVR':
        y_pred_best_r = models_redeem[best_redeem_model].predict(
            scaler_redeem.transform(X_test_r)
        )
    else:
        y_pred_best_r = models_redeem[best_redeem_model].predict(X_test_r)
    
    plot_predictions(y_test_r, y_pred_best_r, best_redeem_model, target_name='赎回')
    
    # 11. 绘制特征重要性
    if hasattr(models_redeem[best_redeem_model], 'feature_importances_'):
        plot_feature_importance(
            models_redeem[best_redeem_model], feature_cols,
            best_redeem_model, target_name='赎回'
        )
    
    # 12. 保存结果
    results_purchase.to_csv('../output/purchase_model_results.csv', index=False, encoding='utf-8-sig')
    results_redeem.to_csv('../output/redeem_model_results.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("模型训练完成！")
    print(f"申购最佳模型: {best_purchase_model}")
    print(f"赎回最佳模型: {best_redeem_model}")
    print("所有结果保存在 ../output/ 目录下")
    print("=" * 60)


if __name__ == '__main__':
    main()
