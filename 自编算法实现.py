import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

matplotlib.use('Agg')  # 使用Agg后端，不显示图形界面
warnings.filterwarnings('ignore')  # 忽略所有警告信息，避免输出干扰

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

def load_and_preprocess_data(filepath):
    # 读取Excel文件
    df = pd.read_excel(filepath)  # 使用pandas读取指定路径的Excel文件，返回DataFrame对象

    # 基本数据清洗
    df = df[df['退回'].isin(['是', '否'])]  # 筛选出"退回"列只包含"是"或"否"的行，去除无效或异常标签
    df = df[(df['销售额'] >= 0) & (df['数量'] > 0)]  # 筛选出销售额非负且数量大于0的行，去除无效数据

    # 特征工程
    df['发货间隔天数'] = (pd.to_datetime(df['发货日期']) - pd.to_datetime(df['订单日期'])).dt.days  # 计算发货日期与订单日期的差值，得到发货间隔天数
    df['发货间隔天数'] = df['发货间隔天数'].clip(0, 365).fillna(df['发货间隔天数'].median())  # 将间隔天数限制在0到365天内，并用中位数填充缺失值

    df['利润率'] = df['利润'] / df['销售额'].replace(0, 1)  # 计算利润率，为避免除以0错误，将销售额为0的值替换为1

    df['是否亏损'] = (df['利润'] < 0).astype(int)  # 判断利润是否小于0，小于0则为亏损，转换为整数类型（0或1）

    df['单价'] = df['销售额'] / df['数量'].replace(0, 1)  # 计算单价，同样避免除以0的情况

    df['是否大额订单'] = (df['销售额'] > 1000).astype(int)  # 判断销售额是否大于1000，大于则为大额订单，转换为整数类型

    df['发货是否延迟'] = (df['发货间隔天数'] > 7).astype(int)  # 判断发货间隔是否大于7天，大于则为延迟发货，转换为整数类型

    df['邮寄方式'] = df['邮寄方式'].astype('category').cat.codes  # 将邮寄方式转换为分类类型，并用编码代替原始值，便于模型处理

    features = ['数量', '销售额', '发货间隔天数', '利润率', '是否亏损', '单价',
                '是否大额订单', '发货是否延迟', '邮寄方式']  # 定义用于模型训练的特征列列表

    # 填充缺失值
    for col in features:  # 遍历每个特征列
        if df[col].dtype in ['int64', 'float64']:  # 如果列的数据类型是整型或浮点型
            df[col] = df[col].fillna(df[col].median())  # 用该列的中位数填充缺失值

    # 标签编码
    df['退回_label'] = df['退回'].map({'否': 0, '是': 1})  # 将"退回"列中的"否"映射为0，"是"映射为1，创建新的标签列

    X = df[features].values  # 从DataFrame中提取特征列，转换为numpy数组，作为特征矩阵
    y = df['退回_label'].values  # 提取标签列，转换为numpy数组，作为目标向量

    return X, y, features, df  # 返回特征矩阵、目标向量、特征列表和原始DataFrame，供后续使用


def create_multiple_data_splits(X, y, splits=[0.3, 0.2, 0.4]):
    data_dict = {}  # 创建一个空字典，用于存储不同划分方式下的数据集

    for test_size in splits:  # 遍历传入的测试集比例列表
        # 分层划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y  # 使用train_test_split函数划分数据集，设置随机种子为42，并保持标签分布的一致性
        )

        # 处理类别不平衡问题
        pos_idx = np.where(y_train == 1)[0]  # 获取训练集中正类（退单）样本的索引
        neg_idx = np.where(y_train == 0)[0]  # 获取训练集中负类（非退单）样本的索引

        if len(pos_idx) < len(neg_idx):  # 如果正类样本数量少于负类样本数量
            pos_resampled = np.random.choice(pos_idx, size=len(neg_idx), replace=True)  # 对正类样本进行有放回抽样，使其数量与负类样本相同
            train_idx = np.concatenate([neg_idx, pos_resampled])  # 将负类样本索引和重采样后的正类样本索引连接起来
        else:  # 如果负类样本数量少于正类样本数量
            neg_resampled = np.random.choice(neg_idx, size=len(pos_idx), replace=True)  # 对负类样本进行有放回抽样，使其数量与正类样本相同
            train_idx = np.concatenate([pos_idx, neg_resampled])  # 将正类样本索引和重采样后的负类样本索引连接起来

        X_train_resampled = X_train[train_idx]  # 根据新的索引获取重采样后的训练特征
        y_train_resampled = y_train[train_idx]  # 根据新的索引获取重采样后的训练标签

        # 特征标准化
        scaler = StandardScaler()  # 创建StandardScaler对象，用于标准化特征
        X_train_scaled = scaler.fit_transform(X_train_resampled)  # 对训练特征进行拟合并转换，使其均值为0，方差为1
        X_test_scaled = scaler.transform(X_test)  # 使用训练集的均值和方差对测试特征进行转换，保持一致性

        # 创建划分名称
        split_name = f"{int((1 - test_size) * 10)}_{int(test_size * 10)}"  # 根据训练集和测试集比例生成划分名称，例如7_3表示训练集70%，测试集30%

        data_dict[split_name] = {  # 将划分后的数据存储到字典中，键为划分名称
            'X_train': X_train_scaled,  # 存储标准化后的训练特征
            'X_test': X_test_scaled,  # 存储标准化后的测试特征
            'y_train': y_train_resampled,  # 存储重采样后的训练标签
            'y_test': y_test,  # 存储测试标签
            'scaler': scaler,  # 存储用于标准化的scaler对象，以便后续对预测数据使用相同的标准化参数
            'original_split': f"{int((1 - test_size) * 10)}:{int(test_size * 10)}"  # 存储原始划分比例，用于可视化显示
        }

    return data_dict  # 返回包含不同划分数据集的字典


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth  # 决策树的最大深度，用于控制树的复杂度，防止过拟合
        self.min_samples_split = min_samples_split  # 节点分裂所需的最小样本数，用于控制树的生长
        self.tree = {}  # 存储决策树结构的字典

    def _gini(self, y):
        if len(y) == 0:  # 如果样本集为空，基尼不纯度为0
            return 0
        p = np.bincount(y) / len(y)  # 计算每个类别的比例，bincount统计每个类别的数量
        return 1 - np.sum(p ** 2)  # 计算基尼不纯度，公式为1 - sum(p_i^2)

    def _best_split(self, X, y, feature_subset):
        best_gini, best_feature, best_threshold = float('inf'), None, None  # 初始化最佳基尼不纯度、最佳特征和最佳阈值
        n_features = X.shape[1]  # 获取特征的数量

        features = np.random.choice(n_features, feature_subset, replace=False) if feature_subset else range(n_features)  # 如果指定了特征子集大小，则随机选择特征；否则使用所有特征

        for feat in features:  # 遍历每个特征
            values = X[:, feat]  # 获取当前特征的所有值
            for val in np.unique(values):  # 遍历当前特征的所有唯一值作为候选阈值
                left_idx = values <= val  # 左子节点：特征值小于等于阈值的样本索引
                right_idx = values > val   # 右子节点：特征值大于阈值的样本索引

                if len(y[left_idx]) < self.min_samples_split or len(y[right_idx]) < self.min_samples_split:  # 如果分裂后的任一侧样本数小于最小分裂样本数，则跳过
                    continue

                gini = (len(y[left_idx]) * self._gini(y[left_idx]) +
                        len(y[right_idx]) * self._gini(y[right_idx])) / len(y)  # 计算加权基尼不纯度，作为分裂质量的评价指标

                if gini < best_gini:  # 如果当前基尼不纯度更小，则更新最佳值
                    best_gini, best_feature, best_threshold = gini, feat, val

        return best_feature, best_threshold  # 返回最佳分裂特征和阈值

    def _build_tree(self, X, y, depth, feature_subset):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:  # 如果达到最大深度、所有样本属于同一类别或样本数小于最小分裂样本数，则停止分裂
            return np.argmax(np.bincount(y))  # 返回当前节点中数量最多的类别作为预测结果

        best_feat, best_thresh = self._best_split(X, y, feature_subset)  # 寻找最佳分裂特征和阈值

        if best_feat is None:  # 如果没有找到合适的分裂，则停止分裂
            return np.argmax(np.bincount(y))

        left_idx = X[:, best_feat] <= best_thresh  # 左子节点的样本索引
        right_idx = X[:, best_feat] > best_thresh   # 右子节点的样本索引

        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1, feature_subset)  # 递归构建左子树
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1, feature_subset)  # 递归构建右子树

        return {'feature': best_feat, 'threshold': best_thresh, 'left': left_tree, 'right': right_tree}  # 返回当前节点，包含分裂特征、阈值和左右子树

    def fit(self, X, y, feature_subset=None):
        np.random.seed(42)  # 设置随机种子，确保结果可复现
        self.tree = self._build_tree(X, y, 0, feature_subset)  # 从根节点开始构建决策树，深度从0开始

    def _predict_sample(self, x, tree):
        if not isinstance(tree, dict):  # 如果当前节点不是字典（即叶节点），则返回预测类别
            return tree

        if x[tree['feature']] <= tree['threshold']:  # 如果样本的特征值小于等于阈值，则进入左子树
            return self._predict_sample(x, tree['left'])
        else:  # 否则进入右子树
            return self._predict_sample(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])  # 对每个样本进行预测，返回预测类别数组

    def predict_proba(self, X):
        predictions = self.predict(X)  # 获取预测类别
        proba_positive = predictions.astype(float)  # 将预测类别转换为浮点数，正类为1.0，负类为0.0
        return np.column_stack([1 - proba_positive, proba_positive])  # 返回每个样本属于两个类别的概率，第一列为负类概率，第二列为正类概率


class RandomForestCustom:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators  # 森林中决策树的数量
        self.max_depth = max_depth  # 每棵决策树的最大深度
        self.min_samples_split = min_samples_split  # 每棵决策树分裂所需的最小样本数
        self.random_state = random_state  # 随机种子，用于控制随机性
        self.trees = []  # 存储所有决策树的列表
        self.feature_importances_ = None  # 特征重要性数组

    def fit(self, X, y):
        np.random.seed(self.random_state)  # 设置随机种子
        n_samples, n_features = X.shape  # 获取样本数量和特征数量
        feature_subset = int(np.sqrt(n_features))  # 每棵树随机选择的特征子集大小，通常取特征数量的平方根
        self.feature_importances_ = np.zeros(n_features)  # 初始化特征重要性为零数组

        for _ in range(self.n_estimators):  # 构建每棵决策树
            sample_idx = np.random.choice(n_samples, n_samples, replace=True)  # 使用自助采样法（bootstrap）从训练集中有放回地抽取样本

            tree = DecisionTree(self.max_depth, self.min_samples_split)  # 创建决策树对象
            tree.fit(X[sample_idx], y[sample_idx], feature_subset)  # 使用采样后的数据训练决策树

            self.trees.append(tree)  # 将训练好的决策树添加到列表中
            self._update_feature_importance(tree.tree)  # 更新特征重要性，基于该树使用的特征

        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)  # 将特征重要性归一化，使其和为1

    def _update_feature_importance(self, tree_node):
        if isinstance(tree_node, dict):  # 如果当前节点是字典（非叶节点）
            self.feature_importances_[tree_node['feature']] += 1  # 增加该特征的重要性计数（使用次数）
            self._update_feature_importance(tree_node['left'])  # 递归更新左子树
            self._update_feature_importance(tree_node['right'])  # 递归更新右子树

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # 获取每棵树的预测结果，组成二维数组（树数量×样本数）
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 0, tree_preds)  # 对每个样本，统计所有树的预测类别，取出现次数最多的类别作为最终预测（多数投票）

    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # 获取每棵树的预测结果
        proba_positive = np.mean(tree_preds == 1, axis=0)  # 计算每棵树的预测为正类的比例，作为正类的概率
        return np.column_stack([1 - proba_positive, proba_positive])  # 返回每个样本属于两个类别的概率


def train_and_evaluate_custom_models(data_dict):
    results = {}  # 存储每个划分的训练结果

    for split_name, data in data_dict.items():  # 遍历每个数据划分
        print(f"正在训练 {data['original_split']} 划分的自编模型...")

        # 训练模型
        start_time = time.time()  # 记录训练开始时间
        model = RandomForestCustom(
            n_estimators=100,  # 设置森林中树的数量为100
            max_depth=15,  # 设置每棵树的最大深度为15
            min_samples_split=2,  # 设置分裂所需的最小样本数为2
            random_state=42  # 设置随机种子
        )
        model.fit(data['X_train'], data['y_train'])  # 使用训练数据训练模型
        train_time = time.time() - start_time  # 计算训练耗时

        # 预测
        start_time = time.time()  # 记录预测开始时间
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]  # 预测测试集的正类概率
        predict_time = time.time() - start_time  # 计算预测耗时

        # 寻找最佳阈值
        best_threshold = 0.5  # 初始化最佳阈值为0.5
        best_f1 = 0  # 初始化最佳F1分数为0
        best_precision = 0  # 初始化最佳精确率为0
        best_recall = 0  # 初始化最佳召回率为0
        y_pred = None  # 初始化最佳阈值对应的预测结果

        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:  # 尝试不同的阈值
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)  # 根据阈值将概率转换为类别
            f1 = f1_score(data['y_test'], y_pred_thresh, zero_division=0)  # 计算F1分数
            precision = precision_score(data['y_test'], y_pred_thresh, zero_division=0)  # 计算精确率
            recall = recall_score(data['y_test'], y_pred_thresh, zero_division=0)  # 计算召回率

            if f1 > best_f1:  # 如果当前阈值下的F1分数更高
                best_f1 = f1  # 更新最佳F1分数
                best_threshold = threshold  # 更新最佳阈值
                best_precision = precision  # 更新最佳精确率
                best_recall = recall  # 更新最佳召回率
                y_pred = y_pred_thresh  # 更新最佳预测结果

        cm = confusion_matrix(data['y_test'], y_pred)  # 计算混淆矩阵

        metrics = {  # 收集评估指标
            'accuracy': accuracy_score(data['y_test'], y_pred),  # 准确率
            'precision': best_precision,  # 精确率
            'recall': best_recall,  # 召回率
            'f1': best_f1,  # F1分数
            'train_time': train_time,  # 训练时间
            'predict_time': predict_time,  # 预测时间
            'confusion_matrix': cm,  # 混淆矩阵
            'tn': cm[0, 0],  # 真负例
            'fp': cm[0, 1],  # 假正例
            'fn': cm[1, 0],  # 假负例
            'tp': cm[1, 1],  # 真正例
            'threshold': best_threshold,  # 最佳阈值
            'y_pred_proba': y_pred_proba  # 预测概率
        }

        results[split_name] = {  # 存储该划分的结果
            'model': model,  # 训练好的模型
            'metrics': metrics,  # 评估指标
            'scaler': data['scaler'],  # 标准化器
            'original_name': data['original_split'],  # 划分名称
            'X_test': data['X_test'],  # 测试特征
            'y_test': data['y_test']  # 测试标签
        }

        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"最佳阈值: {metrics['threshold']:.2f}")
        print(f"训练时间: {metrics['train_time']:.2f}秒\n")

    return results  # 返回所有划分的结果


def visualize_custom_model_results(results, features, save_dir):
    if not os.path.exists(save_dir):  # 如果保存目录不存在
        os.makedirs(save_dir)  # 创建目录

    # 1. 性能对比柱状图
    split_names = [results[split]['original_name'] for split in results.keys()]  # 获取所有划分的名称
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']  # 要展示的指标名称
    metric_titles = ['准确率', '精确率', '召回率', 'F1分数']  # 指标的中文标题

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 创建2行2列的子图，总尺寸14x10英寸
    axes = axes.flatten()  # 将子图数组展平为一维，便于遍历

    for idx, metric in enumerate(metrics_names):  # 遍历每个指标
        values = [results[list(results.keys())[i]]['metrics'][metric] for i in range(len(split_names))]  # 获取每个划分的指标值

        bars = axes[idx].bar(split_names, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])  # 绘制柱状图，设置颜色
        axes[idx].set_title(f'{metric_titles[idx]}对比（自编算法）', fontsize=12)  # 设置子图标题
        axes[idx].set_ylabel('分数', fontsize=10)  # 设置y轴标签
        axes[idx].set_ylim(0, 1)  # 设置y轴范围从0到1

        for bar, val in zip(bars, values):  # 在每个柱子上方添加数值标签
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('自编算法在不同数据集划分下的性能对比', fontsize=14)  # 设置总标题
    plt.tight_layout()  # 调整子图布局，避免重叠
    plt.savefig(os.path.join(save_dir, '1_性能对比_自编.png'), dpi=100, bbox_inches='tight')  # 保存图表为PNG文件
    plt.close()  # 关闭图表，释放内存

    # 2. 混淆矩阵对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 创建1行3列的子图

    for idx, split_key in enumerate(results.keys()):  # 遍历每个划分
        split_name = results[split_key]['original_name']  # 获取划分名称
        cm = results[split_key]['metrics']['confusion_matrix']  # 获取混淆矩阵

        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',  # 使用seaborn绘制热力图，显示数值，使用橙色系配色
                    xticklabels=['非退单', '退单'],  # 设置x轴刻度标签
                    yticklabels=['非退单', '退单'],  # 设置y轴刻度标签
                    ax=axes[idx])  # 指定绘制在哪个子图上

        axes[idx].set_title(f'{split_name}划分 (F1={results[split_key]["metrics"]["f1"]:.3f})', fontsize=12)  # 设置子图标题，包含F1分数
        axes[idx].set_xlabel('预测标签', fontsize=10)  # 设置x轴标签
        axes[idx].set_ylabel('真实标签', fontsize=10)  # 设置y轴标签

    plt.suptitle('自编算法在不同数据集划分的混淆矩阵对比', fontsize=14)  # 设置总标题
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_混淆矩阵对比_自编.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # 3. 时间对比图
    plt.figure(figsize=(12, 6))  # 创建新的图形，尺寸12x6英寸

    split_names_display = [results[split]['original_name'] for split in results.keys()]  # 获取划分名称
    train_times = [results[split]['metrics']['train_time'] for split in results.keys()]  # 获取训练时间
    predict_times = [results[split]['metrics']['predict_time'] for split in results.keys()]  # 获取预测时间

    x = np.arange(len(split_names_display))  # 生成x轴的位置数组
    width = 0.35  # 设置柱状图的宽度

    plt.bar(x - width / 2, train_times, width, label='训练时间', color='#FF8C00')  # 绘制训练时间柱状图，橙色
    plt.bar(x + width / 2, predict_times, width, label='预测时间', color='#00CED1')  # 绘制预测时间柱状图，青色

    plt.xlabel('数据集划分', fontsize=12)  # 设置x轴标签
    plt.ylabel('时间（秒）', fontsize=12)  # 设置y轴标签
    plt.title('自编算法在不同划分下的训练和预测时间对比', fontsize=14)  # 设置标题
    plt.xticks(x, split_names_display)  # 设置x轴刻度标签
    plt.legend()  # 显示图例
    plt.grid(alpha=0.3, axis='y')  # 添加y轴方向的网格线，透明度0.3

    # 添加数值标签
    for i, (train_time, predict_time) in enumerate(zip(train_times, predict_times)):
        plt.text(i - width / 2, train_time + max(train_times) * 0.01, f'{train_time:.2f}s',
                 ha='center', va='bottom', fontsize=9)  # 在训练时间柱子上方添加数值
        plt.text(i + width / 2, predict_time + max(predict_times) * 0.01, f'{predict_time:.2f}s',
                 ha='center', va='bottom', fontsize=9)  # 在预测时间柱子上方添加数值

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_时间对比_自编.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # 4. 特征重要性对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 创建1行3列的子图

    for idx, split_key in enumerate(results.keys()):  # 遍历每个划分
        split_name = results[split_key]['original_name']  # 获取划分名称
        model = results[split_key]['model']  # 获取模型

        importances = model.feature_importances_  # 获取特征重要性
        indices = np.argsort(importances)[::-1]  # 按重要性降序排序，获取索引

        axes[idx].barh(range(len(features)), importances[indices], color='#8B4513')  # 绘制水平条形图，棕色
        axes[idx].set_yticks(range(len(features)))  # 设置y轴刻度
        axes[idx].set_yticklabels([features[i] for i in indices], fontsize=9)  # 设置y轴刻度标签为特征名称，按重要性排序
        axes[idx].set_xlabel('特征重要性', fontsize=10)  # 设置x轴标签
        axes[idx].set_title(f'{split_name}划分 - 特征重要性（自编）', fontsize=11)  # 设置子图标题

        # 添加重要性数值
        for i, (feature_idx, importance) in enumerate(zip(indices, importances[indices])):
            axes[idx].text(importance + 0.001, i, f'{importance:.3f}',
                           va='center', fontsize=8)  # 在条形图右侧添加重要性数值

    plt.suptitle('自编算法在不同数据集划分的特征重要性对比', fontsize=14)  # 设置总标题
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_特征重要性对比_自编.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # 5. ROC曲线对比
    plt.figure(figsize=(10, 8))  # 创建新图形

    for split_key in results.keys():  # 遍历每个划分
        split_name = results[split_key]['original_name']  # 获取划分名称
        y_test = results[split_key]['y_test']  # 获取测试标签
        y_pred_proba = results[split_key]['metrics']['y_pred_proba']  # 获取预测概率

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # 计算ROC曲线的假正率（FPR）和真正率（TPR）
        roc_auc = auc(fpr, tpr)  # 计算ROC曲线下面积（AUC）

        plt.plot(fpr, tpr, lw=2, label=f'{split_name} (AUC = {roc_auc:.3f})')  # 绘制ROC曲线，线宽为2，并添加图例

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 绘制对角线，表示随机分类器的性能
    plt.xlim([0.0, 1.0])  # 设置x轴范围
    plt.ylim([0.0, 1.05])  # 设置y轴范围，留一点空间
    plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)  # 设置x轴标签
    plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)  # 设置y轴标签
    plt.title('自编算法在不同数据集划分的ROC曲线对比', fontsize=14)  # 设置标题
    plt.legend(loc="lower right")  # 将图例放在右下角
    plt.grid(alpha=0.3)  # 添加网格线

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '5_ROC曲线对比_自编.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # 6. 精确率-召回率曲线对比
    plt.figure(figsize=(10, 8))

    for split_key in results.keys():  # 遍历每个划分
        split_name = results[split_key]['original_name']  # 获取划分名称
        y_test = results[split_key]['y_test']  # 获取测试标签
        y_pred_proba = results[split_key]['metrics']['y_pred_proba']  # 获取预测概率

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)  # 计算精确率和召回率
        pr_auc = auc(recall, precision)  # 计算精确率-召回率曲线下面积（PR AUC）

        plt.plot(recall, precision, lw=2, label=f'{split_name} (AUC = {pr_auc:.3f})')  # 绘制PR曲线

    plt.xlabel('召回率 (Recall)', fontsize=12)  # 设置x轴标签
    plt.ylabel('精确率 (Precision)', fontsize=12)  # 设置y轴标签
    plt.title('自编算法在不同数据集划分的精确率-召回率曲线对比', fontsize=14)  # 设置标题
    plt.legend(loc="upper right")  # 将图例放在右上角
    plt.grid(alpha=0.3)  # 添加网格线

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '6_精确率召回率曲线对比_自编.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # 7. 阈值影响分析图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 创建1行3列的子图

    for idx, split_key in enumerate(results.keys()):  # 遍历每个划分
        split_name = results[split_key]['original_name']  # 获取划分名称
        y_test = results[split_key]['y_test']  # 获取测试标签
        y_pred_proba = results[split_key]['metrics']['y_pred_proba']  # 获取预测概率

        thresholds = np.linspace(0.1, 0.9, 50)  # 生成从0.1到0.9的50个等间距阈值
        precisions = []  # 存储每个阈值下的精确率
        recalls = []  # 存储每个阈值下的召回率
        f1s = []  # 存储每个阈值下的F1分数

        for threshold in thresholds:  # 遍历每个阈值
            y_pred = (y_pred_proba >= threshold).astype(int)  # 根据阈值将概率转换为类别
            precisions.append(precision_score(y_test, y_pred, zero_division=0))  # 计算精确率
            recalls.append(recall_score(y_test, y_pred, zero_division=0))  # 计算召回率
            f1s.append(f1_score(y_test, y_pred, zero_division=0))  # 计算F1分数

        axes[idx].plot(thresholds, precisions, 'b-', label='精确率', linewidth=2)  # 绘制精确率曲线，蓝色
        axes[idx].plot(thresholds, recalls, 'g-', label='召回率', linewidth=2)  # 绘制召回率曲线，绿色
        axes[idx].plot(thresholds, f1s, 'r-', label='F1分数', linewidth=2)  # 绘制F1分数曲线，红色

        best_threshold = results[split_key]['metrics']['threshold']  # 获取最佳阈值
        best_f1 = results[split_key]['metrics']['f1']  # 获取最佳F1分数
        best_idx = np.argmin(np.abs(thresholds - best_threshold))  # 找到最佳阈值在阈值数组中的索引

        axes[idx].axvline(x=best_threshold, color='k', linestyle='--', alpha=0.5,
                          label=f'最佳阈值: {best_threshold:.2f}')  # 在最佳阈值处绘制垂直虚线
        axes[idx].scatter([best_threshold], [f1s[best_idx]], color='k', s=50,
                          label=f'F1={best_f1:.3f}')  # 在最佳阈值处绘制点，表示最佳F1分数

        axes[idx].set_xlabel('阈值', fontsize=10)  # 设置x轴标签
        axes[idx].set_ylabel('分数', fontsize=10)  # 设置y轴标签
        axes[idx].set_title(f'{split_name}划分 - 阈值影响分析（自编）', fontsize=11)  # 设置子图标题
        axes[idx].legend(loc='best', fontsize=9)  # 显示图例，自动选择最佳位置
        axes[idx].grid(alpha=0.3)  # 添加网格线
        axes[idx].set_ylim(0, 1)  # 设置y轴范围从0到1

    plt.suptitle('阈值对自编算法性能的影响分析', fontsize=14)  # 设置总标题
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '7_阈值影响分析_自编.png'), dpi=100, bbox_inches='tight')
    plt.close()


def visualize_comparison_results(results_custom, results_sklearn, features, save_dir):
    if not os.path.exists(save_dir):  # 如果目录不存在
        os.makedirs(save_dir)  # 创建目录

    # 获取两个结果中都有的划分
    common_splits = set(results_custom.keys()) & set(results_sklearn.keys())  # 取自编算法和sklearn算法结果的交集

    if not common_splits:  # 如果没有共同的划分
        print("没有共同的划分可用于对比")
        return

    # 按字母顺序排序
    common_splits = sorted(list(common_splits))

    # 获取显示名称
    display_names = [results_custom[split]['original_name'] for split in common_splits]  # 获取划分的显示名称

    # 定义颜色
    colors_custom = '#FF6B6B'  # 自编算法的颜色，红色系
    colors_sklearn = '#4ECDC4'  # sklearn算法的颜色，青色系

    # 定义指标
    metrics_en = ['accuracy', 'precision', 'recall', 'f1']  # 指标英文名称
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']  # 指标中文名称

    # 综合性能对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 创建2行2列的子图
    axes = axes.flatten()

    for idx, (metric, name) in enumerate(zip(metrics_en, metric_names)):  # 遍历每个指标
        # 只获取共有划分的指标值
        custom_vals = [results_custom[s]['metrics'][metric] for s in common_splits]  # 自编算法的指标值
        sklearn_vals = [results_sklearn[s]['metrics'][metric] for s in common_splits]  # sklearn算法的指标值

        x = np.arange(len(display_names))  # 生成x轴位置
        width = 0.35  # 柱状图宽度

        axes[idx].bar(x - width / 2, custom_vals, width, label='自编算法', color=colors_custom, alpha=0.8)  # 绘制自编算法柱状图
        axes[idx].bar(x + width / 2, sklearn_vals, width, label='sklearn', color=colors_sklearn, alpha=0.8)  # 绘制sklearn算法柱状图
        axes[idx].set_title(f'{name}对比', fontsize=12)  # 设置子图标题
        axes[idx].set_xticks(x)  # 设置x轴刻度
        axes[idx].set_xticklabels(display_names)  # 设置x轴刻度标签
        axes[idx].legend()  # 显示图例
        axes[idx].grid(alpha=0.3, axis='y')  # 添加y轴网格线

        # 添加数值标签
        for i, (cv, sv) in enumerate(zip(custom_vals, sklearn_vals)):
            axes[idx].text(i - width / 2, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=8)  # 自编算法数值
            axes[idx].text(i + width / 2, sv + 0.01, f'{sv:.3f}', ha='center', va='bottom', fontsize=8)  # sklearn算法数值

    plt.suptitle('自编与sklearn算法性能综合对比', fontsize=14)  # 设置总标题
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '算法综合性能对比.png'), dpi=150, bbox_inches='tight')  # 保存图表，DPI更高
    plt.close()


def save_final_results(results, features, save_dir):
    # 1. 将结果保存为CSV文件
    results_data = []  # 存储结果数据的列表
    for split_key, result in results.items():  # 遍历每个划分的结果
        metrics = result['metrics']  # 获取指标
        results_data.append({  # 将结果添加到列表
            '划分': result['original_name'],  # 划分名称
            '准确率': metrics['accuracy'],  # 准确率
            '精确率': metrics['precision'],  # 精确率
            '召回率': metrics['recall'],  # 召回率
            'F1分数': metrics['f1'],  # F1分数
            '阈值': metrics['threshold'],  # 最佳阈值
            '训练时间': metrics['train_time'],  # 训练时间
            '预测时间': metrics['predict_time'],  # 预测时间
            '真负例(TN)': metrics['tn'],  # 真负例
            '假正例(FP)': metrics['fp'],  # 假正例
            '假负例(FN)': metrics['fn'],  # 假负例
            '真正例(TP)': metrics['tp']  # 真正例
        })

    # 创建DataFrame并保存为CSV
    results_df = pd.DataFrame(results_data)  # 将列表转换为DataFrame
    csv_path = os.path.join(save_dir, '性能结果汇总_自编.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 保存为CSV文件，支持中文编码
    print(f"性能结果汇总已保存到: {csv_path}")

    # 2. 分析特征重要性（找出F1分数最高的模型）
    best_f1 = 0  # 初始化最佳F1分数
    best_split_key = ""  # 初始化最佳划分键

    # 遍历所有划分结果，找到F1分数最高的模型
    for split_key, result in results.items():
        if result['metrics']['f1'] > best_f1:  # 如果当前划分的F1分数更高
            best_f1 = result['metrics']['f1']  # 更新最佳F1分数
            best_split_key = split_key  # 更新最佳划分键

    # 获取最佳模型
    best_model = results[best_split_key]['model']  # 获取最佳模型
    feature_importance = best_model.feature_importances_  # 获取特征重要性

    # 创建特征重要性DataFrame并按重要性降序排序
    importance_df = pd.DataFrame({
        '特征': features,  # 特征名称
        '重要性': feature_importance  # 特征重要性
    }).sort_values('重要性', ascending=False)  # 按重要性降序排序

    # 特征重要性
    print("\n自编算法特征重要性排序:")
    for idx, row in importance_df.iterrows():  # 遍历每行
        print(f"{row['特征']}: {row['重要性']:.4f}")  # 打印特征和重要性

    # 保存特征重要性到CSV
    importance_path = os.path.join(save_dir, '特征重要性_自编.csv')
    importance_df.to_csv(importance_path, index=False, encoding='utf-8')  # 保存为CSV

    # 3. 保存最佳模型（使用joblib序列化）
    safe_split_name = results[best_split_key]["original_name"].replace(":", "_")  # 将划分名称中的冒号替换为下划线，避免文件命名问题
    model_filename = f'最佳模型_自编_{safe_split_name}.joblib'  # 模型文件名
    model_path = os.path.join(save_dir, model_filename)  # 模型文件路径

    # 保存模型
    joblib.dump(best_model, model_path)  # 使用joblib将模型序列化并保存到文件


def main_custom():
    data_path = r"客户销售退定单.xlsx"  # 数据文件路径
    save_dir = "自编算法结果"  # 结果保存目录

    # 1. 加载和预处理数据
    X, y, features, df = load_and_preprocess_data(data_path)  # 加载数据并进行预处理
    print(f"数据集大小: {X.shape}")  # 打印特征矩阵的形状
    print(f"特征数量: {len(features)}")  # 打印特征数量
    print(f"标签分布-退单: {sum(y == 1)}, 非退单: {sum(y == 0)}")  # 打印正负样本数量
    print(f"退单比例: {sum(y == 1) / len(y) * 100:.2f}%\n")  # 打印正样本比例

    # 2. 创建多种数据划分
    data_dict = create_multiple_data_splits(X, y, splits=[0.3, 0.2, 0.4])  # 创建三种划分：70-30, 80-20, 60-40

    for split_key, data in data_dict.items():  # 打印每个划分的信息
        print(f"{data['original_split']}划分: 训练集={data['X_train'].shape}, 测试集={data['X_test'].shape}")

    # 3. 训练和评估自编模型
    results_custom = train_and_evaluate_custom_models(data_dict)  # 训练并评估自编模型

    # 4. 生成自编算法可视化图表
    visualize_custom_model_results(results_custom, features, save_dir)  # 生成可视化图表

    # 5. 保存最终结果
    save_final_results(results_custom, features, save_dir)  # 保存结果到文件

    # 6. 与sklearn算法对比
    results_sklearn = {}  # 存储sklearn算法结果

    # 训练sklearn模型（使用较小的参数以加快训练速度）
    for split_name, data in data_dict.items():  # 遍历每个数据划分
        try:
            print(f"训练 {data['original_split']} 划分的sklearn模型...")

            start_time = time.time()
            # 使用与自编算法类似的参数
            model_sklearn = RandomForestClassifier(
                n_estimators=50,  # 树的数量设为50，比自编算法的100少，以加快训练
                max_depth=15,  # 最大深度为15
                random_state=42,  # 随机种子
                n_jobs=-1  # 使用所有CPU核心并行训练
            )
            model_sklearn.fit(data['X_train'], data['y_train'])  # 训练模型
            train_time = time.time() - start_time

            start_time = time.time()
            y_pred_proba = model_sklearn.predict_proba(data['X_test'])[:, 1]  # 预测概率
            predict_time = time.time() - start_time

            # 寻找最佳阈值
            best_threshold = 0.5  # 初始化最佳阈值
            best_f1 = 0  # 初始化最佳F1分数
            y_pred = None

            for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:  # 尝试不同阈值
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(data['y_test'], y_pred_thresh, zero_division=0)

                if f1 > best_f1:  # 如果当前阈值下的F1分数更高
                    best_f1 = f1
                    best_threshold = threshold
                    y_pred = y_pred_thresh

            # 计算混淆矩阵
            cm = confusion_matrix(data['y_test'], y_pred)

            # 收集评估指标
            metrics_sklearn = {
                'accuracy': accuracy_score(data['y_test'], y_pred),
                'precision': precision_score(data['y_test'], y_pred, zero_division=0),
                'recall': recall_score(data['y_test'], y_pred, zero_division=0),
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

            # 存储sklearn结果
            results_sklearn[split_name] = {
                'model': model_sklearn,
                'metrics': metrics_sklearn,
                'original_name': data['original_split']
            }

            print(f"sklearn模型训练完成: F1={best_f1:.4f}, 训练时间={train_time:.2f}秒\n")

        except Exception as e:  # 捕获异常
            print(f"训练{data['original_split']}划分的sklearn模型时出错: {e}")
            print("跳过该划分的对比\n")

    # 生成对比图表
    if results_sklearn:  # 如果sklearn结果非空
        comparison_dir = os.path.join(save_dir, "算法对比")  # 对比图表保存目录
        visualize_comparison_results(results_custom, results_sklearn, features, comparison_dir)  # 生成对比图表
    else:
        print("没有成功训练任何sklearn模型，跳过对比")

    # 7. 结果分析总结
    print(f"\n性能对比汇总:")
    print(f"{'划分':<8} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'阈值':<8} {'训练时间':<10}")  # 打印表头

    best_f1 = 0  # 初始化最佳F1分数
    best_split_name = ""  # 初始化最佳划分名称
    for split_key, result in results_custom.items():  # 遍历每个划分的结果
        metrics = result['metrics']
        split_name = result['original_name']
        print(f"{split_name:<8}{metrics['accuracy']:.4f}{metrics['precision']:.4f}"
              f"{metrics['recall']:.4f}{metrics['f1']:.4f}{metrics['threshold']:.2f}"
              f"{metrics['train_time']:.2f}秒")  # 打印每行的性能指标

        if metrics['f1'] > best_f1:  # 如果当前划分的F1分数更高
            best_f1 = metrics['f1']
            best_split_name = split_name

    print(f"\n最优划分: {best_split_name} (F1分数: {best_f1:.4f})")  # 打印最优划分

    # 计算平均性能
    avg_accuracy = np.mean([results_custom[s]['metrics']['accuracy'] for s in results_custom.keys()])  # 平均准确率
    avg_f1 = np.mean([results_custom[s]['metrics']['f1'] for s in results_custom.keys()])  # 平均F1分数
    avg_train_time = np.mean([results_custom[s]['metrics']['train_time'] for s in results_custom.keys()])  # 平均训练时间

    print(f"\n自编算法平均表现:")
    print(f"平均准确率: {avg_accuracy:.4f}")
    print(f" 平均F1分数: {avg_f1:.4f}")
    print(f"平均训练时间: {avg_train_time:.2f}秒")


if __name__ == "__main__":
    main_custom()