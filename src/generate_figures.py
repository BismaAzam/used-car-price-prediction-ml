"""Generate the 14 portfolio figures from results/model_results.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results" / "model_results.csv"
FIGURE_DIR = ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

YELLOW = ["gold", "goldenrod", "khaki", "darkgoldenrod", "palegoldenrod"]


def label_vertical(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )


def label_horizontal(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        value = bar.get_width()
        ax.annotate(
            fmt.format(value),
            (value, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(RESULTS_CSV)
    models = df["Model"].tolist()
    x = np.arange(len(models))

    # 1. MAE
    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, df["MAE"], color=YELLOW)
    plt.title("Average Absolute Error Across Price Prediction Models")
    plt.xlabel("Compared Models")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=20, ha="right")
    label_vertical(plt.gca(), bars, "{:,.0f}")
    save("mae_comparison.png")

    # 2. RMSE
    temp = df.sort_values("RMSE")
    plt.figure(figsize=(9, 5))
    bars = plt.barh(temp["Model"], temp["RMSE"], color=YELLOW)
    plt.title("Root Mean Squared Error Ranking")
    plt.xlabel("RMSE (Lower is Better)")
    plt.ylabel("Model Name")
    label_horizontal(plt.gca(), bars, "{:,.0f}")
    save("rmse_comparison.png")

    # 3. R²
    plt.figure(figsize=(9, 5))
    plt.plot(x, df["R2"], marker="o", linewidth=2.5, color="goldenrod")
    plt.fill_between(x, df["R2"], color="khaki", alpha=0.5)
    for i, value in enumerate(df["R2"]):
        plt.annotate(f"{value:.4f}", (i, value), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)
    plt.xticks(x, models, rotation=20, ha="right")
    plt.title("Explained Variance (R²) by Model")
    plt.xlabel("Model")
    plt.ylabel("R² Score")
    save("r2_comparison.png")

    # 4. Training time
    plt.figure(figsize=(9, 5))
    markerline, stemlines, baseline = plt.stem(x, df["Training Time (s)"])
    plt.setp(markerline, color="goldenrod", markersize=8)
    plt.setp(stemlines, color="goldenrod", linewidth=2)
    plt.setp(baseline, color="black")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.title("Model Training Duration")
    plt.xlabel("Model")
    plt.ylabel("Training Time in Seconds")
    save("training_time_comparison.png")

    # 5. Inference time
    values = df["Inference Time (s)"].to_numpy()
    plt.figure(figsize=(9, 5))
    plt.fill_between(x, values, color="yellow", alpha=0.7)
    plt.plot(x, values, marker="D", color="darkgoldenrod", linewidth=2)
    plt.xticks(x, models, rotation=20, ha="right")
    plt.title("Prediction Speed on the Test Set")
    plt.xlabel("Model")
    plt.ylabel("Inference Time in Seconds")
    save("inference_time_comparison.png")

    # 6. MAE ranking
    temp = df.sort_values("MAE").reset_index(drop=True)
    y = np.arange(len(temp))
    plt.figure(figsize=(9, 5))
    plt.hlines(y, 0, temp["MAE"], color="goldenrod", linewidth=2)
    plt.plot(temp["MAE"], y, "o", color="yellow", markersize=8)
    plt.yticks(y, temp["Model"])
    plt.title("MAE Leaderboard")
    plt.xlabel("Absolute Error")
    plt.ylabel("Model")
    save("mae_ranking.png")

    # 7. RMSE ranking
    temp = df.sort_values("RMSE").reset_index(drop=True)
    y = np.arange(len(temp))
    plt.figure(figsize=(9, 5))
    plt.hlines(y, 0, temp["RMSE"], color="darkgoldenrod", linewidth=2)
    plt.plot(temp["RMSE"], y, "o", color="khaki", markersize=8)
    plt.yticks(y, temp["Model"])
    plt.title("RMSE Leaderboard")
    plt.xlabel("Squared Error Penalty")
    plt.ylabel("Model")
    save("rmse_ranking.png")

    # 8. R² ranking
    temp = df.sort_values("R2", ascending=False)
    plt.figure(figsize=(9, 5))
    bars = plt.barh(temp["Model"], temp["R2"], color=YELLOW)
    plt.gca().invert_yaxis()
    plt.title("R² Performance Ranking")
    plt.xlabel("R² Value (Higher is Better)")
    plt.ylabel("Model")
    label_horizontal(plt.gca(), bars, "{:.4f}")
    save("r2_ranking.png")

    # 9. MAE / RMSE grouped
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, df["MAE"], width, label="MAE", color="gold")
    plt.bar(x + width / 2, df["RMSE"], width, label="RMSE", color="goldenrod")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.title("Direct Comparison of MAE and RMSE")
    plt.xlabel("Model")
    plt.ylabel("Error Value")
    plt.legend()
    save("mae_rmse_comparison.png")

    # 10. Efficiency
    plt.figure(figsize=(9, 5))
    plt.plot(models, df["Training Time (s)"], marker="o",
             linewidth=2, label="Training Time", color="goldenrod")
    plt.plot(models, df["Inference Time (s)"], marker="s",
             linewidth=2, label="Inference Time", color="gold")
    plt.title("Computation Cost: Training vs Prediction")
    plt.xlabel("Model")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    save("efficiency_comparison.png")

    # 11. Training vs R²
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Training Time (s)"], df["R2"], s=120, color="goldenrod")
    for _, row in df.iterrows():
        plt.annotate(row["Model"], (row["Training Time (s)"], row["R2"]),
                     xytext=(6, 6), textcoords="offset points", fontsize=8)
    plt.title("Accuracy Versus Training Cost")
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("R² Score")
    save("training_vs_r2_scatter.png")

    # 12. MAE vs training
    sizes = df["Inference Time (s)"] * 1500 + 100
    plt.figure(figsize=(8, 5))
    plt.scatter(df["MAE"], df["Training Time (s)"], s=sizes,
                color="khaki", edgecolors="darkgoldenrod", alpha=0.9)
    for _, row in df.iterrows():
        plt.annotate(row["Model"], (row["MAE"], row["Training Time (s)"]),
                     xytext=(6, 6), textcoords="offset points", fontsize=8)
    plt.title("Error Magnitude Versus Training Duration")
    plt.xlabel("MAE")
    plt.ylabel("Training Time (seconds)")
    save("mae_vs_training_scatter.png")

    # 13. Normalized score heatmap
    metrics = df.set_index("Model")[
        ["MAE", "RMSE", "R2", "Training Time (s)", "Inference Time (s)"]
    ]
    scores = pd.DataFrame(index=metrics.index)
    for column in metrics.columns:
        values = metrics[column].astype(float)
        if column == "R2":
            scores[column] = (values - values.min()) / (values.max() - values.min())
        else:
            scores[column] = 1 - (
                (values - values.min()) / (values.max() - values.min())
            )

    plt.figure(figsize=(10, 4.8))
    plt.imshow(scores.values, aspect="auto", cmap="YlOrBr")
    plt.colorbar(label="Normalized Score")
    plt.xticks(
        range(len(scores.columns)),
        ["MAE", "RMSE", "R²", "Train Time", "Infer Time"],
        rotation=20,
        ha="right",
    )
    plt.yticks(range(len(scores.index)), scores.index)
    plt.title("Normalized Scorecard of Model Quality")
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            plt.text(j, i, f"{scores.iloc[i, j]:.2f}",
                     ha="center", va="center", fontsize=8)
    save("normalized_performance_heatmap.png")

    # 14. Overall rank
    ranks = pd.DataFrame(index=metrics.index)
    ranks["MAE"] = metrics["MAE"].rank(ascending=True)
    ranks["RMSE"] = metrics["RMSE"].rank(ascending=True)
    ranks["R2"] = metrics["R2"].rank(ascending=False)
    ranks["Training"] = metrics["Training Time (s)"].rank(ascending=True)
    ranks["Inference"] = metrics["Inference Time (s)"].rank(ascending=True)
    ranks["Average Rank"] = ranks.mean(axis=1)
    temp = ranks.sort_values("Average Rank")

    plt.figure(figsize=(9, 5))
    bars = plt.bar(temp.index, temp["Average Rank"], color=YELLOW)
    plt.title("Overall Rank Across All Evaluation Criteria")
    plt.xlabel("Model")
    plt.ylabel("Average Rank (Lower is Better)")
    plt.xticks(rotation=20, ha="right")
    label_vertical(plt.gca(), bars, "{:.2f}")
    save("overall_model_ranking.png")


if __name__ == "__main__":
    main()
