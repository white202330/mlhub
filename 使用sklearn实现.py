import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import joblib  # 模型持久化库，用于保存和加载训练好的模型

warnings.filterwarnings('ignore')

# 设置matplotlib中文字体，确保图表能正确显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 100  # 设置图表分辨率


# 数据预处理函数
def load_and_preprocess_data(filepath):
    # 读取Excel数据文件，pandas会自动识别格式并将数据加载到DataFrame中
    df = pd.read_excel(filepath)

    # 数据清洗：只保留退回列中值为"是"或"否"的行
    df = df[df['退回'].isin(['是', '否'])]

    # 进一步数据清洗：过滤掉销售额为负和数量为0的数据
    df = df[(df['销售额'] >= 0) & (df['数量'] > 0)]

    # 特征工程开始：从原始数据中创建新的特征以提升模型性能
    # 计算发货间隔天数：发货日期减去订单日期
    df['发货间隔天数'] = (pd.to_datetime(df['发货日期']) - pd.to_datetime(df['订单日期'])).dt.days

    # 处理发货间隔天数的异常值和缺失值
    df['发货间隔天数'] = df['发货间隔天数'].clip(0, 365).fillna(df['发货间隔天数'].median())

    # 计算利润率：利润除以销售额
    df['利润率'] = df['利润'] / df['销售额'].replace(0, 1)

    # 创建是否亏损特征：利润小于0则为亏损订单
    df['是否亏损'] = (df['利润'] < 0).astype(int)

    # 计算单价：销售额除以数量
    df['单价'] = df['销售额'] / df['数量'].replace(0, 1)

    # 创建是否大额订单特征：销售额超过1000的订单
    df['是否大额订单'] = (df['销售额'] > 1000).astype(int)

    # 创建发货是否延迟特征：发货间隔超过7天的订单
    df['发货是否延迟'] = (df['发货间隔天数'] > 7).astype(int)

    # 将邮寄方式从分类变量转换为数值编码
    df['邮寄方式'] = df['邮寄方式'].astype('category').cat.codes

    # 选择最终用于模型训练的特征列
    features = ['数量', '销售额', '发货间隔天数', '利润率', '是否亏损', '单价',
                '是否大额订单', '发货是否延迟', '邮寄方式']

    # 处理特征中的缺失值
    for col in features:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())

    # 标签编码：将目标变量从文本("是"/"否")转换为数值(1/0)
    df['退回_label'] = df['退回'].map({'否': 0, '是': 1})

    # 从DataFrame中提取特征矩阵X和目标变量y
    X = df[features].values
    y = df['退回_label'].values

    return X, y, features, df


# 数据划分函数
def create_multiple_data_splits(X, y, splits=[0.3, 0.2, 0.4]):
    # 创建多种不同的训练集/测试集划分，用于比较不同数据划分对模型性能的影响
    data_dict = {}

    for test_size in splits:
        # 使用train_test_split划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 处理类别不平衡问题：使用简单过采样方法
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]

        # 如果正样本少于负样本，对正样本进行过采样
        if len(pos_idx) < len(neg_idx):
            pos_resampled = np.random.choice(pos_idx, size=len(neg_idx), replace=True)
            train_idx = np.concatenate([neg_idx, pos_resampled])
        else:
            neg_resampled = np.random.choice(neg_idx, size=len(pos_idx), replace=True)
            train_idx = np.concatenate([pos_idx, neg_resampled])

        # 使用过采样后的索引构建新的训练集
        X_train_resampled = X_train[train_idx]
        y_train_resampled = y_train[train_idx]

        # 特征标准化：将特征缩放到均值为0，方差为1
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)

        # 生成划分名称
        split_name = f"{int((1 - test_size) * 10)}_{int(test_size * 10)}"

        # 将所有数据存储到字典中
        data_dict[split_name] = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_resampled,
            'y_test': y_test,
            'scaler': scaler,
            'original_split': f"{int((1 - test_size) * 10)}:{int(test_size * 10)}"
        }

    return data_dict


# 模型训练和评估函数
def train_and_evaluate_models(data_dict):
    # 使用随机森林分类器训练多个模型，并评估它们在各种数据划分上的性能
    results = {}

    for split_name, data in data_dict.items():
        print(f"\n正在训练 {data['original_split']} 划分的模型...")

        # 记录训练开始时间
        start_time = time.time()

        # 创建随机森林分类器实例
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # 在训练集上训练模型
        model.fit(data['X_train'], data['y_train'])
        train_time = time.time() - start_time

        # 记录预测开始时间
        start_time = time.time()
        # 使用训练好的模型预测测试集的概率
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
        predict_time = time.time() - start_time

        # 寻找最佳分类阈值
        best_threshold = 0.5
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        y_pred = None

        # 尝试不同的阈值
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(data['y_test'], y_pred_thresh, zero_division=0)
            precision = precision_score(data['y_test'], y_pred_thresh, zero_division=0)
            recall = recall_score(data['y_test'], y_pred_thresh, zero_division=0)

            # 选择F1分数最高的阈值
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
                y_pred = y_pred_thresh

        # 计算混淆矩阵
        cm = confusion_matrix(data['y_test'], y_pred)

        # 收集所有评估指标
        metrics = {
            'accuracy': accuracy_score(data['y_test'], y_pred),
            'precision': best_precision,
            'recall': best_recall,
            'f1': best_f1,
            'train_time': train_time,
            'predict_time': predict_time,
            'confusion_matrix': cm,
            'tn': cm[0, 0],
            'fp': cm[0, 1],
            'fn': cm[1, 0],
            'tp': cm[1, 1],
            'threshold': best_threshold,
            'y_pred_proba': y_pred_proba
        }

        # 存储当前划分的结果
        results[split_name] = {
            'model': model,
            'metrics': metrics,
            'scaler': data['scaler'],
            'original_name': data['original_split'],
            'X_test': data['X_test'],
            'y_test': data['y_test']
        }

        # 打印当前划分的结果摘要
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"最佳阈值: {metrics['threshold']:.2f}")
        print(f"训练时间: {metrics['train_time']:.2f}秒")

    return results


# 可视化函数
def visualize_model_results(results, features, save_dir):
    # 生成可视化图表
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 图表1: 性能对比柱状图
    split_names = [results[split]['original_name'] for split in results.keys()]
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_titles = ['准确率', '精确率', '召回率', 'F1分数']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_names):
        values = [results[list(results.keys())[i]]['metrics'][metric] for i in range(len(split_names))]

        bars = axes[idx].bar(split_names, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[idx].set_title(f'{metric_titles[idx]}对比', fontsize=12)
        axes[idx].set_ylabel('分数', fontsize=10)
        axes[idx].set_ylim(0, 1)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('不同数据集划分下的模型性能对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_性能对比.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("图表1已保存: 1_性能对比.png")

    # 图表2: 混淆矩阵对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, split_key in enumerate(results.keys()):
        split_name = results[split_key]['original_name']
        cm = results[split_key]['metrics']['confusion_matrix']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['非退单', '退单'],
                    yticklabels=['非退单', '退单'],
                    ax=axes[idx])

        axes[idx].set_title(f'{split_name}划分 (F1={results[split_key]["metrics"]["f1"]:.3f})', fontsize=12)
        axes[idx].set_xlabel('预测标签', fontsize=10)
        axes[idx].set_ylabel('真实标签', fontsize=10)

    plt.suptitle('不同数据集划分的混淆矩阵对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_混淆矩阵对比.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("  图表2已保存: 2_混淆矩阵对比.png")

    # ==================== 图表3: 时间对比图 ====================
    print("正在生成图表3: 时间对比图...")

    plt.figure(figsize=(12, 6))

    split_names_display = [results[split]['original_name'] for split in results.keys()]
    train_times = [results[split]['metrics']['train_time'] for split in results.keys()]
    predict_times = [results[split]['metrics']['predict_time'] for split in results.keys()]

    x = np.arange(len(split_names_display))
    width = 0.35

    plt.bar(x - width / 2, train_times, width, label='训练时间', color='#FF6B6B')
    plt.bar(x + width / 2, predict_times, width, label='预测时间', color='#4ECDC4')

    plt.xlabel('数据集划分', fontsize=12)
    plt.ylabel('时间（秒）', fontsize=12)
    plt.title('不同划分下的训练和预测时间对比', fontsize=14)
    plt.xticks(x, split_names_display)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')

    for i, (train_time, predict_time) in enumerate(zip(train_times, predict_times)):
        plt.text(i - width / 2, train_time + max(train_times) * 0.01, f'{train_time:.2f}s',
                 ha='center', va='bottom', fontsize=9)
        plt.text(i + width / 2, predict_time + max(predict_times) * 0.01, f'{predict_time:.2f}s',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_时间对比.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("  图表3已保存: 3_时间对比.png")

    # ==================== 图表4: 特征重要性对比图 ====================
    print("正在生成图表4: 特征重要性对比图...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, split_key in enumerate(results.keys()):
        split_name = results[split_key]['original_name']
        model = results[split_key]['model']

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        axes[idx].barh(range(len(features)), importances[indices], color='#2E86AB')
        axes[idx].set_yticks(range(len(features)))
        axes[idx].set_yticklabels([features[i] for i in indices], fontsize=9)
        axes[idx].set_xlabel('特征重要性', fontsize=10)
        axes[idx].set_title(f'{split_name}划分 - 特征重要性', fontsize=11)

        for i, (feature_idx, importance) in enumerate(zip(indices, importances[indices])):
            axes[idx].text(importance + 0.001, i, f'{importance:.3f}',
                           va='center', fontsize=8)

    plt.suptitle('不同数据集划分的特征重要性对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_特征重要性对比.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("  图表4已保存: 4_特征重要性对比.png")

    # ==================== 图表5: ROC曲线对比 ====================
    print("正在生成图表5: ROC曲线对比...")

    plt.figure(figsize=(10, 8))

    for split_key in results.keys():
        split_name = results[split_key]['original_name']
        y_test = results[split_key]['y_test']
        y_pred_proba = results[split_key]['metrics']['y_pred_proba']

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{split_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    plt.title('不同数据集划分的ROC曲线对比', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '5_ROC曲线对比.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("  图表5已保存: 5_ROC曲线对比.png")

    # ==================== 图表6: 精确率-召回率曲线对比 ====================
    print("正在生成图表6: 精确率-召回率曲线对比...")

    plt.figure(figsize=(10, 8))

    for split_key in results.keys():
        split_name = results[split_key]['original_name']
        y_test = results[split_key]['y_test']
        y_pred_proba = results[split_key]['metrics']['y_pred_proba']

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, lw=2, label=f'{split_name} (AUC = {pr_auc:.3f})')

    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('不同数据集划分的精确率-召回率曲线对比', fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '6_精确率召回率曲线对比.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("  图表6已保存: 6_精确率召回率曲线对比.png")

    # ==================== 图表7: 阈值影响分析图 ====================
    print("正在生成图表7: 阈值影响分析图...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, split_key in enumerate(results.keys()):
        split_name = results[split_key]['original_name']
        y_test = results[split_key]['y_test']
        y_pred_proba = results[split_key]['metrics']['y_pred_proba']

        thresholds = np.linspace(0.1, 0.9, 50)
        precisions = []
        recalls = []
        f1s = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))

        axes[idx].plot(thresholds, precisions, 'b-', label='精确率', linewidth=2)
        axes[idx].plot(thresholds, recalls, 'g-', label='召回率', linewidth=2)
        axes[idx].plot(thresholds, f1s, 'r-', label='F1分数', linewidth=2)

        best_threshold = results[split_key]['metrics']['threshold']
        best_f1 = results[split_key]['metrics']['f1']
        best_idx = np.argmin(np.abs(thresholds - best_threshold))

        axes[idx].axvline(x=best_threshold, color='k', linestyle='--', alpha=0.5,
                          label=f'最佳阈值: {best_threshold:.2f}')
        axes[idx].scatter([best_threshold], [f1s[best_idx]], color='k', s=50,
                          label=f'F1={best_f1:.3f}')

        axes[idx].set_xlabel('阈值', fontsize=10)
        axes[idx].set_ylabel('分数', fontsize=10)
        axes[idx].set_title(f'{split_name}划分 - 阈值影响分析', fontsize=11)
        axes[idx].legend(loc='best', fontsize=9)
        axes[idx].grid(alpha=0.3)
        axes[idx].set_ylim(0, 1)

    plt.suptitle('阈值对模型性能的影响分析', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '7_阈值影响分析.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("  图表7已保存: 7_阈值影响分析.png")

    print("=" * 50)
    print(f"所有7张可视化图表已保存到目录: {save_dir}")
    print("=" * 50)


# ==================== 结果保存函数 ====================
def save_final_results(results, features, save_dir):
    # 这个函数保存最终的分析结果，包括性能汇总、特征重要性和最佳模型

    print("正在保存最终结果...")
    print("=" * 50)

    # 1. 保存性能结果汇总到CSV
    print("正在保存性能结果汇总...")

    results_data = []

    for split_key, result in results.items():
        metrics = result['metrics']
        results_data.append({
            '划分': result['original_name'],
            '准确率': metrics['accuracy'],
            '精确率': metrics['precision'],
            '召回率': metrics['recall'],
            'F1分数': metrics['f1'],
            '阈值': metrics['threshold'],
            '训练时间': metrics['train_time'],
            '预测时间': metrics['predict_time'],
            '真负例(TN)': metrics['tn'],
            '假正例(FP)': metrics['fp'],
            '假负例(FN)': metrics['fn'],
            '真正例(TP)': metrics['tp']
        })

    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(save_dir, '性能结果汇总.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 性能结果汇总已保存: {csv_path}")

    # 2. 特征重要性分析
    print("正在分析特征重要性...")

    best_f1 = 0
    best_split_key = ""

    for split_key, result in results.items():
        if result['metrics']['f1'] > best_f1:
            best_f1 = result['metrics']['f1']
            best_split_key = split_key

    print(f"最佳划分: {results[best_split_key]['original_name']}, F1分数: {best_f1:.4f}")

    best_model = results[best_split_key]['model']
    feature_importance = best_model.feature_importances_

    importance_df = pd.DataFrame({
        '特征': features,
        '重要性': feature_importance
    }).sort_values('重要性', ascending=False)

    print("\n特征重要性排序:")
    print("-" * 30)
    for idx, row in importance_df.iterrows():
        print(f"  {row['特征']}: {row['重要性']:.4f}")

    importance_path = os.path.join(save_dir, '特征重要性.csv')
    importance_df.to_csv(importance_path, index=False, encoding='utf-8')
    print(f"✓ 特征重要性已保存: {importance_path}")

    # 3. 保存最佳模型
    print("正在保存最佳模型...")

    safe_filename = f'最佳模型_{results[best_split_key]["original_name"].replace(":", "_")}.joblib'
    model_path = os.path.join(save_dir, safe_filename)

    joblib.dump(best_model, model_path)
    print(f"✓ 最佳模型已保存: {model_path}")

    # 4. 保存模型训练配置信息
    print("正在保存模型配置信息...")

    config_info = f"""模型配置信息
===============
保存时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
最佳划分: {results[best_split_key]['original_name']}
最佳F1分数: {best_f1:.4f}
特征数量: {len(features)}

模型参数:
-----------
算法: 随机森林分类器(RandomForestClassifier)
树的数量(n_estimators): 300
最大深度(max_depth): 20
最小分裂样本数(min_samples_split): 5
最小叶节点样本数(min_samples_leaf): 2
类别权重(class_weight): balanced
随机种子(random_state): 42

特征列表:
-----------
{chr(10).join([f'{i + 1}. {feature}' for i, feature in enumerate(features)])}

性能指标:
-----------
"""

    for split_key, result in results.items():
        metrics = result['metrics']
        config_info += f"""
{result['original_name']}划分:
  准确率: {metrics['accuracy']:.4f}
  精确率: {metrics['precision']:.4f}
  召回率: {metrics['recall']:.4f}
  F1分数: {metrics['f1']:.4f}
  最佳阈值: {metrics['threshold']:.2f}
  训练时间: {metrics['train_time']:.2f}秒
"""

    config_path = os.path.join(save_dir, '模型配置信息.txt')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_info)

    print(f"✓ 模型配置信息已保存: {config_path}")
    print("=" * 50)
    print("所有结果保存完成!")
    print("=" * 50)


# ==================== 主函数 ====================
def main_sklearn():
    # 这是整个机器学习流程的主函数，协调所有步骤的执行

    print("=" * 60)
    print("开始退单预测机器学习分析")
    print("=" * 60)

    # 1. 设置文件路径和保存目录
    data_path = r"客户销售退定单.xlsx"
    save_dir = "sklearn算法结果"

    print(f"数据文件: {data_path}")
    print(f"结果保存目录: {save_dir}")
    print("-" * 60)

    # 2. 加载和预处理数据
    print("步骤1: 加载和预处理数据")
    print("-" * 40)

    X, y, features, df = load_and_preprocess_data(data_path)

    print(f"✓ 数据加载完成")
    print(f"  数据集大小: {X.shape}")
    print(f"  特征数量: {len(features)}")
    print(f"  特征列表: {features}")
    print(f"  标签分布-退单: {sum(y == 1)}, 非退单: {sum(y == 0)}")
    print(f"  退单比例: {sum(y == 1) / len(y) * 100:.2f}%")

    # 3. 创建多种数据划分
    print("\n步骤2: 创建多种数据划分")
    print("-" * 40)

    data_dict = create_multiple_data_splits(X, y, splits=[0.3, 0.2, 0.4])

    for split_key, data in data_dict.items():
        print(f"  {data['original_split']}划分: 训练集={data['X_train'].shape}, 测试集={data['X_test'].shape}")

    # 4. 训练和评估模型
    print("\n步骤3: 训练和评估模型")
    print("-" * 40)

    results = train_and_evaluate_models(data_dict)

    # 5. 可视化结果
    print("\n步骤4: 生成可视化图表")
    print("-" * 40)

    visualize_model_results(results, features, save_dir)

    # 6. 保存最终结果
    print("\n步骤5: 保存最终结果")
    print("-" * 40)

    save_final_results(results, features, save_dir)

    # 7. 结果分析和总结
    print("\n步骤6: 结果分析总结")
    print("-" * 40)

    print("性能对比汇总:")
    print("-" * 80)
    print(f"{'划分':<8} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'阈值':<8} {'训练时间':<10}")
    print("-" * 80)

    best_f1 = 0
    best_split_name = ""

    for split_key, result in results.items():
        metrics = result['metrics']
        split_name = result['original_name']

        print(f"{split_name:<8} {metrics['accuracy']:.4f}    {metrics['precision']:.4f}    "
              f"{metrics['recall']:.4f}    {metrics['f1']:.4f}    {metrics['threshold']:.2f}    "
              f"{metrics['train_time']:.2f}秒")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_split_name = split_name

    print("-" * 80)
    print(f"✓ 最优划分: {best_split_name} (F1分数: {best_f1:.4f})")

    # 8. 最终总结
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    print(f"所有结果已保存在目录: {save_dir}")
    print("\n目录内容:")
    print("-" * 40)

    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        for file in sorted(files):
            if file.endswith('.png'):
                print(f"图表: {file}")
            elif file.endswith('.csv'):
                print(f"数据: {file}")
            elif file.endswith('.joblib'):
                print(f"模型: {file}")
            elif file.endswith('.txt'):
                print(f"配置: {file}")


if __name__ == "__main__":
    main_sklearn()