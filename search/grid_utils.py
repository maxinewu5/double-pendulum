from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable
import csv


@dataclass
class GridSearchResult:
    params: dict[str, Any]
    metrics: dict[str, Any]


def iter_param_grid(param_grid: dict[str, list[Any]]):
    keys = list(param_grid.keys())
    for values in product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, values))


def run_grid_search(
    param_grid: dict[str, list[Any]],
    evaluate_fn: Callable[[dict[str, Any]], dict[str, Any]],
    progress_label: str = "grid",
) -> list[GridSearchResult]:
    total_runs = 1
    for values in param_grid.values():
        total_runs *= len(values)

    results: list[GridSearchResult] = []
    for run_index, params in enumerate(iter_param_grid(param_grid), start=1):
        metrics = evaluate_fn(params)
        results.append(GridSearchResult(params=params, metrics=metrics))
        print(f"[{run_index:03d}/{total_runs}] {progress_label} params={params} metrics={metrics}")

    return results


def write_results_csv(
    output_path: str,
    results: list[GridSearchResult],
    metric_keys: list[str],
) -> None:
    if not results:
        return

    param_keys = list(results[0].params.keys())
    fieldnames = param_keys + metric_keys
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {**result.params}
            row.update({key: result.metrics.get(key) for key in metric_keys})
            writer.writerow(row)
