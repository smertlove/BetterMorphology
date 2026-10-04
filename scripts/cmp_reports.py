from argparse import ArgumentParser
from pathlib import Path
import json


def compare_reports(*reports, metric="macro_f1", greater_is_better=True):
    if not reports:
        return {}

    categories = set()
    for report in reports:
        categories.update(report.keys())

    results = {}

    for category in categories:
        scores = []
        for idx, report in enumerate(reports):
            entry = report.get(category)
            if not entry or metric not in entry:
                continue
            scores.append((idx, entry[metric]))

        if not scores:
            continue

        scores.sort(key=lambda x: x[1], reverse=greater_is_better)

        results[category] = {
            "best": scores[0][0],
            "score": scores[0][1],
            "ranking": scores,
        }

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()

    report_paths = [Path(p) for p in args.reports]
    all_reports = []
    for p in report_paths:
        with open(p, "r", encoding="utf-8") as file:
            all_reports.append(json.load(file))

    result = compare_reports(*all_reports)

    for cat, info in result.items():
        best = report_paths[info['best']].name
        print(f"{cat}: best = {best} ({info['score']:.4f})")
        for idx, score in info["ranking"]:
            print(f"    {report_paths[idx].name}: {score:.4f}")


if __name__ == "__main__":
    main()