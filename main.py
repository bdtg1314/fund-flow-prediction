"""
资金流入流出预测 - 主程序入口
完整的机器学习竞赛项目流程
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def print_banner(title):
    """打印标题横幅"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """主函数"""
    print_banner("资金流入流出预测 - 机器学习课程设计项目")
    print("\n项目: 天池大赛 - 资金流入流出预测-挑战Baseline")
    print("算法: 多种机器学习算法（随机森林、GBDT、XGBoost、LightGBM等）")
    print("任务: 预测2014年9月每日资金申购和赎回金额")
    
    # 检查数据文件是否存在
    data_path = '../data/'
    required_files = [
        'user_profile_table.csv',
        'user_balance_table.csv',
        'mfd_day_share_interest.csv',
        'mfd_bank_shibor.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_path, f))]
    
    if missing_files:
        print(f"\n警告: 缺少以下数据文件: {missing_files}")
        print("请先运行 generate_data.py 生成模拟数据，或放入真实竞赛数据")
        print("\n是否继续？(y/n): ", end='')
        # 这里简化处理，直接继续
        print("继续执行...")
    
    # 步骤1: 数据生成（可选）
    print_banner("步骤 1/5: 数据准备")
    if not os.path.exists(os.path.join(data_path, 'user_balance_table.csv')):
        print("正在生成模拟数据...")
        from generate_data import main as generate_data_main
        generate_data_main()
    else:
        print("数据文件已存在，跳过生成步骤")
    
    # 步骤2: 探索性数据分析
    print_banner("步骤 2/5: 探索性数据分析 (EDA)")
    try:
        from eda_analysis import main as eda_main
        eda_main()
    except Exception as e:
        print(f"EDA分析出错: {e}")
        print("继续下一步...")
    
    # 步骤3: 特征工程
    print_banner("步骤 3/5: 特征工程")
    try:
        from feature_engineering import main as feature_main
        feature_main()
    except Exception as e:
        print(f"特征工程出错: {e}")
        print("继续下一步...")
    
    # 步骤4: 模型训练与评估
    print_banner("步骤 4/5: 模型训练与评估")
    try:
        from model_training import main as training_main
        training_main()
    except Exception as e:
        print(f"模型训练出错: {e}")
        print("继续下一步...")
    
    # 步骤5: 最终预测
    print_banner("步骤 5/5: 最终预测与结果生成")
    try:
        from predict import main as predict_main
        predict_main()
    except Exception as e:
        print(f"预测生成出错: {e}")
    
    # 总结
    print_banner("项目执行完成！")
    print("\n产出文件:")
    print("  - 数据文件: ../data/*.csv")
    print("  - 分析图表: ../output/*.png")
    print("  - 特征矩阵: ../output/feature_matrix.csv")
    print("  - 模型结果: ../output/*_model_results.csv")
    print("  - 提交文件: ../output/submission.csv")
    
    print("\n项目结构:")
    print("  fund_flow_prediction/")
    print("  ├── data/          # 数据文件")
    print("  ├── src/           # 源代码")
    print("  │   ├── generate_data.py      # 数据生成")
    print("  │   ├── eda_analysis.py       # 探索性数据分析")
    print("  │   ├── feature_engineering.py # 特征工程")
    print("  │   ├── model_training.py     # 模型训练")
    print("  │   ├── predict.py            # 预测生成")
    print("  │   └── main.py               # 主程序入口")
    print("  ├── docs/          # 文档")
    print("  └── output/        # 输出结果")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
