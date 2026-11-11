from __future__ import annotations


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    matrix = [[0, 0], [0, 0]]
    for truth, pred in zip(y_true, y_pred):
        if truth not in (0, 1) or pred not in (0, 1):
            raise ValueError(f"Expected binary labels, got true={truth} pred={pred}.")
        matrix[truth][pred] += 1
    real_p, real_r, real_f = _prf(matrix[0][0], matrix[1][0], matrix[0][1])
    fake_p, fake_r, fake_f = _prf(matrix[1][1], matrix[0][1], matrix[1][0])
    correct = matrix[0][0] + matrix[1][1]
    return {
        "accuracy": correct / len(y_true),
        "precision_fake": fake_p,
        "recall_fake": fake_r,
        "f1_fake": fake_f,
        "precision_real": real_p,
        "recall_real": real_r,
        "f1_real": real_f,
        "macro_f1": (real_f + fake_f) / 2.0,
    }


def report_predictions(y_true: list[int], y_pred: list[int]) -> str:
    scores = score_predictions(y_true, y_pred)
    support_real = sum(label == 0 for label in y_true)
    support_fake = sum(label == 1 for label in y_true)
    macro_p = (scores["precision_real"] + scores["precision_fake"]) / 2.0
    macro_r = (scores["recall_real"] + scores["recall_fake"]) / 2.0

    def row(name: str, precision: float, recall: float, f1: float, support: int) -> str:
        return f"{name:>12} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}"

    lines = [
        f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
        "",
        row("real", scores["precision_real"], scores["recall_real"], scores["f1_real"], support_real),
        row("fake", scores["precision_fake"], scores["recall_fake"], scores["f1_fake"], support_fake),
        "",
        f"{'accuracy':>12} {'':>10} {'':>10} {scores['accuracy']:10.4f} {len(y_true):10d}",
        row("macro avg", macro_p, macro_r, scores["macro_f1"], len(y_true)),
    ]
    return "\n".join(lines)
