from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "reports" / "figures" / "paper"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = ROOT / "data" / "processed" / "tardis" / "phase2_outputs"
LLM_PANEL = BASE / "phase2_canonical_tuned_legacy_500runs" / "lob_full_simulation_llm_tuned_legacy.csv"
UNIFORM_PANEL = BASE / "phase2_uniform_tuned_legacy_500runs" / "lob_full_simulation_phase2_uniform_tuned_legacy.csv"
LITERATURE_PANEL = BASE / "phase2_literature_tuned_legacy_500runs" / "lob_full_simulation_phase2_literature_tuned_legacy.csv"
RUN_PANEL = BASE / "phase2_canonical_tuned_legacy_500runs" / "run_level_causal_panel_trailing200.csv"

COLORS = {
    "llm": "#0f4c5c",
    "uniform": "#9a6d38",
    "literature": "#7d2e3a",
    "empirical": "#1f1f1f",
    "crashed": "#b23a48",
    "stable": "#2f6f4f",
    "baseline": "#6c757d",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str, tight_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.965)) -> None:
    fig.tight_layout(rect=tight_rect)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def rolling_max_drawdown_pct(close: pd.Series, window_ticks: int = 10) -> float:
    rolling_peak = close.rolling(window_ticks, min_periods=1).max()
    drawdown_pct = (rolling_peak - close) / rolling_peak.replace(0, pd.NA) * 100.0
    return float(drawdown_pct.max())


def build_temporal_panel(path: Path, window_before_ticks: int = 200, window_after_ticks: int = 100) -> pd.DataFrame:
    cols = [
        "run_id",
        "tick_ms",
        "mid_price",
        "ofi",
        "spread_bps",
        "depth_imbalance",
        "trade_intensity",
        "flash_crash_flag",
    ]
    df = pd.read_csv(path, usecols=cols)
    crash_ticks = (
        df.loc[df["flash_crash_flag"] == 1, ["run_id", "tick_ms"]]
        .groupby("run_id", sort=False)["tick_ms"]
        .min()
        .rename("first_crash_tick")
    )
    crash_prices = (
        df.merge(crash_ticks, on="run_id", how="inner")
        .loc[lambda x: x["tick_ms"] == x["first_crash_tick"], ["run_id", "mid_price"]]
        .drop_duplicates("run_id")
        .set_index("run_id")["mid_price"]
        .rename("crash_mid_price")
    )
    windowed = df.merge(crash_ticks, on="run_id", how="inner").merge(crash_prices, on="run_id", how="inner")
    windowed["rel_tick"] = ((windowed["tick_ms"] - windowed["first_crash_tick"]) / 100.0).round().astype(int)
    windowed = windowed.loc[
        (windowed["rel_tick"] >= -window_before_ticks) & (windowed["rel_tick"] <= window_after_ticks)
    ].copy()
    windowed["rel_seconds"] = windowed["rel_tick"] / 10.0
    windowed["mid_price_change_pct"] = (
        (windowed["mid_price"] / windowed["crash_mid_price"] - 1.0) * 100.0
    )
    aggregated = (
        windowed.groupby("rel_tick", sort=True)
        .agg(
            rel_seconds=("rel_seconds", "mean"),
            mid_price_change_pct=("mid_price_change_pct", "mean"),
            ofi=("ofi", "mean"),
            spread_bps=("spread_bps", "mean"),
            depth_imbalance=("depth_imbalance", "mean"),
            trade_intensity=("trade_intensity", "mean"),
        )
        .reset_index(drop=True)
    )
    return aggregated


def plot_temporal_anatomy() -> None:
    summary = build_temporal_panel(LLM_PANEL)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=True)
    x = summary["rel_seconds"]

    axes[0, 0].plot(x, summary["mid_price_change_pct"], color=COLORS["llm"], linewidth=2.2)
    axes[0, 0].axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("A. Mid-price change")
    axes[0, 0].set_ylabel("Percent from crash onset")

    axes[0, 1].plot(x, summary["ofi"], color=COLORS["crashed"], linewidth=2.2)
    axes[0, 1].axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("B. Directional flow")
    axes[0, 1].set_ylabel("Mean signed flow")

    axes[1, 0].plot(x, summary["spread_bps"], color=COLORS["llm"], linewidth=2.2, label="Trading cost")
    axes[1, 0].axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax2 = axes[1, 0].twinx()
    ax2.plot(x, summary["depth_imbalance"], color=COLORS["stable"], linewidth=1.8, label="Book balance")
    ax2.spines["top"].set_visible(False)
    axes[1, 0].set_title("C. Trading cost and book balance")
    axes[1, 0].set_ylabel("Spread (bps)")
    ax2.set_ylabel("Depth imbalance")
    lines = [line for line in axes[1, 0].get_lines() + ax2.get_lines() if not line.get_label().startswith("_")]
    labels = [line.get_label() for line in lines]
    axes[1, 0].legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        frameon=False,
        ncol=2,
        columnspacing=1.0,
        handlelength=2.2,
        borderaxespad=0.0,
    )

    axes[1, 1].plot(x, summary["trade_intensity"], color=COLORS["literature"], linewidth=2.2)
    axes[1, 1].axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("D. Trading activity")
    axes[1, 1].set_ylabel("Trades per step")

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.grid(alpha=0.18, linewidth=0.8)
        ax.set_xlabel("Seconds relative to first crash")

    fig.suptitle("Temporal anatomy of the locked LLM crash panel", y=0.99, fontsize=12.5)
    save_figure(fig, "figure_02_temporal_anatomy_llm", tight_rect=(0.0, 0.0, 1.0, 0.91))


def draw_group_boxplot(ax: plt.Axes, df: pd.DataFrame, column: str, title: str, ylabel: str) -> None:
    groups = [df.loc[df["crashed"] == 0, column].to_numpy(), df.loc[df["crashed"] == 1, column].to_numpy()]
    box = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#1f1f1f", "linewidth": 1.4},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    fills = ["#c7d3dd", "#e7b8bf"]
    for patch, fill in zip(box["boxes"], fills):
        patch.set_facecolor(fill)
        patch.set_edgecolor("#4a4a4a")
    rng = np.random.default_rng(20260512)
    for idx, values in enumerate(groups, start=1):
        x = rng.normal(loc=idx, scale=0.06, size=len(values))
        ax.scatter(x, values, s=10, alpha=0.18, color="#4a4a4a", linewidths=0)
    ax.set_xticks([1, 2], labels=["Non-crash runs", "Crash runs"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)


def plot_precrash_and_interventions() -> None:
    run_df = pd.read_csv(RUN_PANEL)
    intervention = {
        "Baseline": 0.1760,
        "Local OFI\nmitigation": 0.1743,
        "Local leverage\nmitigation": 0.1358,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 6.0), gridspec_kw={"width_ratios": [1.05, 1.05, 1.15]})
    draw_group_boxplot(axes[0], run_df, "ofi_pre_mean", "A. Pre-crash directional pressure", "Mean signed flow")
    draw_group_boxplot(axes[1], run_df, "leverage_pre_max", "B. Pre-crash leverage", "Peak leverage proxy")

    labels = list(intervention.keys())
    values = list(intervention.values())
    colors = [COLORS["baseline"], COLORS["crashed"], COLORS["stable"]]
    bars = axes[2].bar(labels, values, color=colors, width=0.62)
    axes[2].axhline(0.1760, color="#444444", linestyle="--", linewidth=1.0)
    axes[2].set_title("C. Local mitigation crash-rate changes")
    axes[2].set_ylabel("Predicted crash rate")
    axes[2].set_ylim(0.0, 0.198)
    axes[2].grid(axis="y", alpha=0.18, linewidth=0.8)
    axes[2].tick_params(axis="x", labelsize=9)
    for bar, value in zip(bars, values):
        y_text = min(value + 0.005, 0.186)
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, y_text, f"{value:.4f}", ha="center", va="bottom")
    axes[2].annotate(
        "-0.98%",
        xy=(1, values[1]),
        xytext=(1, 0.189),
        ha="center",
        va="bottom",
        color=COLORS["crashed"],
        fontsize=9,
    )
    axes[2].annotate(
        "-22.83%",
        xy=(2, values[2]),
        xytext=(2, 0.165),
        ha="center",
        va="bottom",
        color=COLORS["stable"],
        fontsize=9,
    )

    fig.suptitle("Pre-crash separation and local mitigation effects", y=0.99, fontsize=12.5)
    save_figure(fig, "figure_03_precrash_and_interventions", tight_rect=(0.0, 0.0, 1.0, 0.93))


def comparison_metrics() -> pd.DataFrame:
    metrics = pd.DataFrame(
        [
            {"condition": "Empirical benchmark", "tail": 8581.0, "crash_rate": np.nan, "pre_flow": 0.26, "drop_flow": -2.70},
            {"condition": "LLM priors", "tail": 29.08417701780441, "crash_rate": 0.1760, "pre_flow": 0.18690711279890576, "drop_flow": -5.039080724277167},
            {"condition": "Broad priors", "tail": 26.492039467122257, "crash_rate": 0.0080, "pre_flow": 0.11542590303617163, "drop_flow": -2.9119870910923527},
            {"condition": "Literature priors", "tail": 77.19552670535295, "crash_rate": 0.1520, "pre_flow": 0.18475693260798706, "drop_flow": -2.88101141626842},
        ]
    )
    return metrics


def plot_locked_comparison() -> None:
    metrics = comparison_metrics()
    drawdown = {}
    for label, path in {
        "LLM priors": LLM_PANEL,
        "Broad priors": UNIFORM_PANEL,
        "Literature priors": LITERATURE_PANEL,
    }.items():
        df = pd.read_csv(path, usecols=["run_id", "close"])
        drawdown[label] = df.groupby("run_id", sort=False)["close"].apply(rolling_max_drawdown_pct).mean()

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))

    tail_order = ["Empirical benchmark", "LLM priors", "Broad priors", "Literature priors"]
    tail_values = metrics.set_index("condition").loc[tail_order, "tail"]
    tail_colors = [COLORS["empirical"], COLORS["llm"], COLORS["uniform"], COLORS["literature"]]
    axes[0].bar(tail_order, tail_values, color=tail_colors, width=0.65)
    axes[0].set_yscale("log")
    axes[0].set_title("A. Tail thickness")
    axes[0].set_ylabel("Excess kurtosis (log scale)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.18, linewidth=0.8)

    crash_order = ["LLM priors", "Broad priors", "Literature priors"]
    crash_values = metrics.set_index("condition").loc[crash_order, "crash_rate"]
    crash_colors = [COLORS["llm"], COLORS["uniform"], COLORS["literature"]]
    axes[1].axhspan(0.05, 0.40, color="#dbeadf", alpha=0.7)
    bars = axes[1].bar(crash_order, crash_values, color=crash_colors, width=0.65)
    axes[1].set_title("B. Crash incidence")
    axes[1].set_ylabel("Crash rate")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.18, linewidth=0.8)
    for bar, value in zip(bars, crash_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + 0.008, f"{value:.3f}", ha="center", va="bottom")

    x_positions = np.array([0, 1])
    for condition, color in zip(crash_order, crash_colors):
        row = metrics.set_index("condition").loc[condition]
        axes[2].plot(x_positions, [row["pre_flow"], row["drop_flow"]], marker="o", linewidth=2.0, color=color, label=condition)
    axes[2].axhline(0.0, color="#444444", linestyle=":", linewidth=1.0)
    axes[2].set_xticks([0, 1], labels=["Pre-crash phase", "Drop phase"])
    axes[2].set_title("C. Directional flow shift")
    axes[2].set_ylabel("Mean signed flow")
    axes[2].grid(axis="y", alpha=0.18, linewidth=0.8)
    axes[2].legend(frameon=False, loc="upper right")

    fig.suptitle("Locked comparison across the three simulation conditions", y=1.03, fontsize=13)
    save_figure(fig, "appendix_figure_a1_locked_comparison")

    summary = pd.DataFrame(
        {
            "condition": ["LLM priors", "Broad priors", "Literature priors"],
            "mean_short_horizon_drawdown_pct": [
                float(drawdown["LLM priors"]),
                float(drawdown["Broad priors"]),
                float(drawdown["Literature priors"]),
            ],
        }
    )
    summary.to_csv(OUTPUT_DIR / "appendix_figure_a1_drawdown_summary.csv", index=False)


def main() -> None:
    configure_style()
    plot_temporal_anatomy()
    plot_precrash_and_interventions()
    plot_locked_comparison()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()