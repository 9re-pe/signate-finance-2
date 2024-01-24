from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def macro_f1_score(y_truth, y_hat):
    """マクロF1スコアの算出"""

    score = f1_score(y_truth, y_hat, average="macro")

    return score


def show_confusion_matrix(y_truth, y_hat):
    """混同行列の表示"""

    cm = confusion_matrix(y_truth, y_hat)

    TN, FP, FN, TP = cm.ravel()
    TNR = TN / (TN + FP)
    TPR = TP / (TP + FN)

    # 正解率の表示
    print(f"True Negative Rate (0の正解率): {TNR:.2f}")
    print(f"True Positive Rate (1の正解率): {TPR:.2f}")

    # 混同行列の描画
    _, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()
