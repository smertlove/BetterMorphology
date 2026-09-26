import matplotlib.pyplot as plt
from collections import defaultdict


def parse_columns(columns, sep="::"):

    main_losses = []
    per_category_losses = defaultdict(list)
    per_category_metrics = defaultdict(list)

    for col in columns:
        parts = col.split(sep)
        if len(parts) == 3 and "loss" in parts:
            _, metric, _ = parts
            main_losses.append(col)
        else:
            _, category, metric, _ = parts
            if metric == "loss":
                per_category_losses[category].append(col)
            else:
                per_category_metrics[category].append(col)


    return main_losses, per_category_losses, per_category_metrics


def plot_all(main_losses, per_category_losses, per_category_metrics, names_order, df):
    def _pretty(name):
        parts = name.split("::")
        return parts[3] + "_" + parts[2] if len(parts) > 1 else name

    # x-axis shifted so the first point is at 1 instead of 0
    x = df.index + 1

    n = len(names_order)
    height_ratios = [2] + [1] * n

    fig = plt.figure(figsize=(18, 10 * (n + 1)))
    gs = fig.add_gridspec(
        n + 1, 2,
        height_ratios=height_ratios,
        hspace=0.4, wspace=0.25,
    )

    # --- Main losses: keep raw names, mark best val ---
    ax_main = fig.add_subplot(gs[0, :])
    for loss in main_losses:
        ax_main.plot(x, df[loss].values, label=loss)

        if "val" in loss.lower():
            y = df[loss]
            if y.notna().any():
                best_idx = y.idxmin()
                best_x, best_y = best_idx + 1, y.loc[best_idx]   # +1 shift

                ax_main.axhline(best_y, color="gray", linestyle="--",
                                linewidth=1, alpha=0.7)

                ax_main.scatter([best_x], [best_y], color="red", zorder=5,
                                marker="x", s=60)

                ax_main.annotate(
                    f"best {loss} = {best_y:.4g} @ {best_x}",
                    xy=(best_x, best_y),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=8, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1),
                )

    ax_main.set_title("Main Losses")
    ax_main.set_xlabel("epoch"); ax_main.set_ylabel("loss")
    ax_main.legend(loc="best"); ax_main.grid(True, alpha=0.3)

    # --- Per-category rows ---
    for i, name in enumerate(names_order, start=1):
        # --- Left: per-category losses, mark best val ---
        ax_left = fig.add_subplot(gs[i, 0])
        for loss in per_category_losses[name]:
            ax_left.plot(x, df[loss].values, label=_pretty(loss))

            if "val" in loss.lower():
                y = df[loss]
                if y.notna().any():
                    best_idx = y.idxmin()
                    best_x, best_y = best_idx + 1, y.loc[best_idx]   # +1 shift

                    ax_left.axhline(best_y, color="gray", linestyle="--",
                                    linewidth=1, alpha=0.7)

                    ax_left.scatter([best_x], [best_y], color="red", zorder=5,
                                    marker="x", s=60)

                    ax_left.annotate(
                        f"best {_pretty(loss)} = {best_y:.4g} @ {best_x}",
                        xy=(best_x, best_y),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=1),
                    )

        ax_left.set_title(f"[{name}] losses")
        ax_left.set_xlabel("epoch"); ax_left.set_ylabel("loss")
        ax_left.legend(loc="best", fontsize=8); ax_left.grid(True, alpha=0.3)

        ax_right = fig.add_subplot(gs[i, 1])
        for metric in per_category_metrics[name]:
            if "f1" not in metric: continue
            ax_right.plot(x, df[metric].values, label=_pretty(metric))
        ax_right.set_title(f"[{name}] metrics")
        ax_right.set_xlabel("epoch"); ax_right.set_ylabel("value")
        ax_right.legend(loc="best", fontsize=8); ax_right.grid(True, alpha=0.3)

    plt.show()