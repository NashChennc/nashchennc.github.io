from typing import List, Optional, Any
from numpy.typing import NDArray
import numpy as np
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def Calculate_F1(y_true: NDArray[Any], y_pred: NDArray[Any], output_file: Optional[str] = None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1
    )
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if output_file is not None:
        mode = 'w' 
        with open(output_file, mode) as f:
            print(f"Recall: {recall:.4f}", file=f)
            print(f"Precision: {precision:.4f}", file=f)
            print(f"F1 Score: {f1:.4f}", file=f)
            
        print(f"Results saved to {output_file}")
    return precision, recall, f1

def Calculate_F1_threshold(y_true: NDArray[Any], y_scores: NDArray[Any], fixed_threshold: Optional[float] = None, output_file: Optional[str] = None):
    if fixed_threshold is not None:
        best_threshold = fixed_threshold
        y_pred = (y_scores > fixed_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1
        )
        thresholds = [fixed_threshold]
        precisions = [precision]
        recalls = [recall]
        f1_scores = [f1]
            
    else:
        print("⚡ Searching for optimal threshold...")
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)
            
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        f1 = f1_scores[best_idx]
        precision = precisions[best_idx]
        recall = recalls[best_idx]
        print(f"-"*30)
        print(f"Threshold Used: {best_threshold:.4f}")

    print(f"-"*30)
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"-"*30)

    if output_file is not None:
        mode = 'w' 
        with open(output_file, mode) as f:
            print(f"Threshold Used: {best_threshold:.4f}", file=f)
            print(f"Recall: {recall:.4f}", file=f)
            print(f"Precision: {precision:.4f}", file=f)
            print(f"F1 Score: {f1:.4f}", file=f)
            
        print(f"Results saved to {output_file}")

    return thresholds, precisions, recalls, f1_scores

def Visualize_F1(y_true: NDArray[Any], y_scores: NDArray[Any], target_thresholds: List[float], output_file: Optional[str] = None):

    class ScoreWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, scores):
            self.scores = scores
            self.classes_ = np.array([0, 1])
            self._estimator_type = "classifier"
        def fit(self, X, y=None): return self
        def predict(self, X): return (self.scores >= 0.5).astype(int)
        def predict_proba(self, X): return np.column_stack((1 - self.scores, self.scores))
    
    print("⚡ Generating PR Curve with Multiple Threshold Markers...")
    
    # --- 2. 绘制 Yellowbrick 底图 ---
    # 我们保留曲线作为背景，alpha设为0.4让它不要太抢眼
    model = ScoreWrapper(y_scores)
    viz = PrecisionRecallCurve(
        model, 
        iso_f1_curves=True, 
        micro=False, 
        per_class=True,
        classes=[0, 1],
        title="PR Curve with Selected Thresholds",
        line_kws={'alpha': 0.5, 'linewidth': 2} 
    )

    X_dummy = np.zeros((len(y_true), 1))
    viz.fit(X_dummy, y_true)
    viz.score(X_dummy, y_true)
    
    # ==========================================
    # 🎯 核心逻辑：寻找并标记多个阈值点
    # ==========================================
    
    # 1. 获取全量数据 (用于查找)
    # precision, recall 长度是 N+1, thresholds 长度是 N
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores, pos_label=1)
    
    # 2. 定义颜色表 (让不同的阈值显示不同颜色)
    # 使用 viridis 颜色映射，根据阈值列表的索引分配颜色
    colors = cm.get_cmap('plasma')(np.linspace(0, 0.85, len(target_thresholds)))
    
    # 3. 遍历用户输入的阈值列表
    for i, target in enumerate(target_thresholds):
        
        # --- 关键步骤：寻找最近邻 ---
        # 用户的 target (比如 0.5) 可能不在 thresholds 数组里 (全是浮点数)
        # 我们使用 abs(diff).argmin() 找到数值最接近的那个索引
        closest_idx = np.abs(thresholds - target).argmin()
        
        # 提取对应的 P 和 R
        p_point = precisions[closest_idx]
        r_point = recalls[closest_idx]
        actual_t = thresholds[closest_idx]
        
        # 绘制散点
        viz.ax.scatter(
            x=r_point, 
            y=p_point, 
            s=50,            # 点的大小
            color=colors[i],  # 使用分配好的颜色
            edgecolor='black',
            marker='+',       # 圆点，也可以换成 '*'
            zorder=20,        # 保证图层最靠上
            label=f'Th={target} (P={p_point:.2f}, R={r_point:.2f})'
        )
        
        # 添加文字注释 (Annotate)
        # 为了防止文字重叠，我们可以交替调整偏移量
        # offset_sign = 1 if i % 2 == 0 else -1
        # viz.ax.annotate(
        #     text=f'T={target}',
        #     xy=(r_point, p_point),
        #     xytext=(r_point + 0.05 * offset_sign, p_point + 0.05),
        #     arrowprops=dict(facecolor=colors[i], arrowstyle='->', lw=1.5),
        #     fontsize=9,
        #     color='black',
        #     fontweight='bold'
        # )

    # 重新设置图例 (放在合适的位置)
    viz.ax.legend(loc='lower left', frameon=True, fancybox=True, framealpha=0.9)
    viz.ax.set_aspect('equal', 'box')

    if output_file:
        viz.show(outpath=output_file)
    else:
        viz.show()

def Visualize_Position_Distribution(positions: NDArray[Any], bins=200, title="Token Position Distribution", output_file: Optional[str] = None):
    """
    绘制 Token 位置的分布直方图。
    
    Args:
        positions: 包含 Token 位置索引的列表 (e.g., [1, 5, 12, 10, ...])
        bins: 直方图的箱子数量，或者是 'auto'
        title: 图表标题
    """
    # --- 1. 环境配置：修复字体 ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # --- 2. 创建画布 ---
    # 分布图通常宽一点比较好看，方便观察 X 轴的跨度
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # --- 3. 绘制直方图 (Histogram) ---
    # alpha: 透明度
    # edgecolor: 柱子边缘颜色，能够让柱子之间区分更明显
    # density: 是否归一化 (True则显示概率密度，False显示频次)
    n, bins_edges, patches = ax.hist(
        positions, 
        bins=bins, 
        color='#4c72b0', # 经典的深蓝色
        edgecolor='white', 
        alpha=0.8,
        rwidth=0.9 # 让柱子之间稍微留点空隙，视觉上更美观
    )
    
    # --- 4. 添加统计线 (可选：平均值/中位数) ---
    mean_pos = np.mean(positions)
    median_pos = np.median(positions)
    
    ax.axvline(mean_pos, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_pos:.1f}')
    ax.axvline(median_pos, color='orange', linestyle='-', linewidth=1.5, label=f'Median: {median_pos:.1f}')

    # --- 5. 装饰图表 ---
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Token Position (Index)", fontsize=12)
    ax.set_ylabel("Frequency / Count", fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if output_file is not None:
        plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    # 模拟数据
    y_true_test = [0, 1, 0, 1, 1, 0]
    y_scores_test = [0.1, 0.8, 0.3, 0.6, 0.9, 0.4]
    Visualize_F1(y_true_test, y_scores_test)