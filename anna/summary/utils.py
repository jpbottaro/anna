"""Utilities to process metrics from model summaries"""

import os


def keep_step(metrics, step):
    """
    Only keeps the values of the metrics for the step `step`.

    Default to None for each metric that does not have the given `step`.

    Args:
        metrics (dict): keys are metric names, values are lists of
          (step, value) pairs
          e.g.
          {
            "loss": [(1, 0.1), (100, 0.01), ...],
            "acc": [(1, 0.9), (100, 0.91), ...]
          }
        step (int): the step we want to keep

    Returns:
        metrics (dict): keys are metric names, values have the flatten result
          for the given `step`
          e.g.
          {
            "loss": 0.01,
            "acc": 0.91
          }
    """
    result = {}
    for metric, values in metrics.items():
        result[metric] = None
        for s, v in values:
            if s == step:
                result[metric] = v
                break

    return result


def best_step(metrics, target_metric="loss", target_run="dev", lowest=True):
    """
    Finds the best step that optimizes `target_metric` on the validation set.

    Args:
        metrics (dict): the train/val/test metrics for a model
          e.g.
          {
            "train": {"loss": {(1, 0.1), (100, 0.01), ...], "acc": ...},
            "val":   {"loss": [(1, 0.2), (100, 0.03), ...], "acc": ...},
            "test":  {"loss": [(1, 0.3), (100, 0.14), ...], "acc": ...},
           }
        target_metric (str): the metric we want to optimize (e.g. "acc")
        lowest (bool): find the lowest value of `target_metric` (use False
          increasing metrics like 'accuracy')

    Returns:
        best (dict): the test metrics, keeping only the best step according to
          the validation, e.g. {"loss": 0.2, "acc": 0.5}
    """
    values = metrics[target_run][target_metric]

    if len(values) == 0:
        raise ValueError("No metrics to analyze for: {}".format(target_metric))

    best_s = values[0][0]
    best_v = values[0][1]
    for s, v in values:
        # Pick last best value
        if (lowest and v <= best_v) or (not lowest and v >= best_v):
            best_s = s
            best_v = v

    return {k: keep_step(v, best_s) for k, v in metrics.items()}


def latex_name(model_name):
    """
    Replaces model names with latex-friendly names.

    E.g. "avg_br" becomes "$E_{avg}|D_{br}$".

    Args:
        model_name (str): original name of the model

    Returns:
        latex_name (str): latex-friendly translation of the model name
    """
    if model_name == "enc_dec":
        return "$seq2seq$"
    elif "enc_dec_" in model_name:
        return "$seq2seq_{{{}}}$".format(model_name[len("enc_dec_"):])
    elif "_" not in model_name:
        return model_name

    i = model_name.index("_")

    return "$E_{{{}}}|D_{{{}}}$".format(model_name[:i], model_name[i+1:])


if __name__ == "__main__":
    import sys
    import json
    import tabulate

    if len(sys.argv) < 2:
        print("Usage: utils.py EXTRACTED_METRICS")
        exit(1)

    headers = ["ACC", "HA", "ebF1", "miF1", "maF1"]
    keys = ["perf/accuracy",
            "perf/hamming",
            "perf/ebF1",
            "perf/miF1",
            "perf/maF1"]

    # Resolve metrics path
    metrics_path = os.path.abspath(sys.argv[1])

    # Read metrics
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)

    # Pick best step using validation, and display results in test
    table = []
    for model, metrics in all_metrics.items():
        best = best_step(metrics, "perf/accuracy", False)["test"]
        table.append([latex_name(model)] + [best[k] for k in keys])

    # Highlight best result of each column
    for i in range(1, len(keys) + 1):
        max_score = max([t[i] for t in table])
        for row in table:
            if row[i] == max_score:
                row[i] = "\\textbf{{{:.4f}}}".format(row[i])
            else:
                row[i] = "{:.4f}".format(row[i])

    print(tabulate.tabulate(table, headers, tablefmt="latex_raw"))
