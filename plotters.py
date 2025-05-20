import pandas as pd
import seaborn as sns
from matplotlib import patches
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_presence_score(score: float, title: str = "Presence Score Grading", scale_type: str = "general"):
    """
    Plots a color-coded grading bar for a given presence score with type-specific grading thresholds.

    Parameters:
        score (float): The score to mark on the grading bar.
        title (str): Title of the plot.
        scale_type (str): Type of grading scale ('general', 'spatial', 'involvement').
    """
    if scale_type == "general":
        thresholds = {"F": 0, "E": 3.47, "D": 3.65, "C": 3.86, "B": 4.07, "A": 4.41}
    elif scale_type == "spatial":
        thresholds = {"F": 0, "E": 4.01, "D": 4.25, "C": 4.50, "B": 4.76, "A": 5.25}
    elif scale_type == "involvement":
        thresholds = {"F": 0, "E": 3.38, "D": 3.75, "C": 4.00, "B": 4.50, "A": 4.87}
    elif scale_type == "experienced_realism":
        thresholds = {"F": 0, "E": 2.63, "D": 3.0, "C": 3.38, "B": 3.75, "A": 4.5}
    else:
        raise ValueError(f"Unsupported scale_type: {scale_type}")

    if not title:
        title_map = {
            "general": "General Presence Score Grading",
            "spatial": "Spatial Presence Score Grading",
            "involvement": "Involvement Score Grading",
            "experienced_realism": "Experienced Realism Score Grading"
        }
        title = title_map.get(scale_type, "Presence Score Grading")

    colors = {
        "F": "darkred",
        "E": "red",
        "D": "orange",
        "C": "gold",
        "B": "yellowgreen",
        "A": "green"
    }

    fig, ax = plt.subplots(figsize=(10, 2.5))
    grade_labels = list(thresholds.keys())

    # Draw color bars and grade labels
    for i in range(len(grade_labels) - 1):
        grade = grade_labels[i + 1]
        x_start = thresholds[grade_labels[i]]
        x_end = thresholds[grade]
        ax.add_patch(patches.Rectangle((x_start, 0), x_end - x_start, 1,
                                       color=colors[grade_labels[i]], label=grade_labels[i]))

        label_x = x_end - 0.15 if grade_labels[i] == "F" else (x_start + x_end) / 2
        ax.text(label_x, -0.4, grade_labels[i], ha='center', va='bottom', fontsize=12, color='black')

    # Add last band (A)
    ax.add_patch(patches.Rectangle((thresholds["A"], 0), 6 - thresholds["A"], 1,
                                   color=colors["A"], label="A"))
    ax.text(thresholds["A"] + 0.15, -0.4, "A", ha='center', va='bottom', fontsize=12, color='black')

    # Score line and label
    ax.axvline(score, color='black', linewidth=2)
    grade = "F"
    for g in reversed(grade_labels):
        if score >= thresholds[g]:
            grade = g
            break
    label_color = colors[grade]
    if score <= 5.5:
        ax.text(score + 0.06, 1.15, f"{score:.2f}",
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor=label_color, boxstyle='round,pad=0.2', linewidth=2))
    else:
        ax.text(score - 0.06, 1.15, f"{score:.2f}",
                fontsize=12, fontweight='bold', color='black', ha='right',
                bbox=dict(facecolor='white', edgecolor=label_color, boxstyle='round,pad=0.2', linewidth=2))

    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks([])
    ax.set_title(title)

    # Labels for Acceptable/Unacceptable
    ax.text(0.2, -0.4, "Unacceptable", fontsize=12, color="black", va="bottom")
    ax.text(5.85, -0.4, "Acceptable", fontsize=12, color="black", va="bottom", ha='right')

    plt.tight_layout()
    plt.show()


def plot_all_presence_scores(scores: dict):
    """
    Draws all presence score grading scales in a single stacked plot for comparison.

    Parameters:
        scores (dict): Dictionary with keys 'general', 'spatial', 'involvement', 'experienced_realism',
                       each mapped to a float score.
    """
    scale_configs = {
        "general": {
            "thresholds": {"F": 0, "E": 3.47, "D": 3.65, "C": 3.86, "B": 4.07, "A": 4.41},
            "title": "General Precence"
        },
        "spatial": {
            "thresholds": {"F": 0, "E": 4.01, "D": 4.25, "C": 4.50, "B": 4.76, "A": 5.25},
            "title": "Spatial Presence"
        },
        "involvement": {
            "thresholds": {"F": 0, "E": 3.38, "D": 3.75, "C": 4.00, "B": 4.50, "A": 4.87},
            "title": "Involvement"
        },
        "experienced_realism": {
            "thresholds": {"F": 0, "E": 2.63, "D": 3.0, "C": 3.38, "B": 3.75, "A": 4.5},
            "title": "Experienced Realism"
        },
    }

    colors = {
        "F": "darkred",
        "E": "red",
        "D": "orange",
        "C": "gold",
        "B": "yellowgreen",
        "A": "green"
    }

    fig, axes = plt.subplots(len(scale_configs), 1, figsize=(12, 2.7 * len(scale_configs)), sharex=True)

    if len(scale_configs) == 1:
        axes = [axes]

    for ax, (scale_type, config) in zip(axes, scale_configs.items()):
        score = scores.get(scale_type, None)
        if score is None:
            continue  # skip if score for this scale not provided

        thresholds = config["thresholds"]
        grade_labels = list(thresholds.keys())

        # Draw colored grading bands
        for i in range(len(grade_labels) - 1):
            grade = grade_labels[i + 1]
            x_start = thresholds[grade_labels[i]]
            x_end = thresholds[grade]
            ax.add_patch(patches.Rectangle((x_start, 0), x_end - x_start, 1.1,
                                           color=colors[grade_labels[i]]))

            label_x = x_end - 0.15 if grade_labels[i] == "F" else (x_start + x_end) / 2
            ax.text(label_x, -0.3, grade_labels[i], ha='center', va='bottom', fontsize=12)

        # Last segment (A)
        ax.add_patch(patches.Rectangle((thresholds["A"], 0), 6 - thresholds["A"], 1.1,
                                       color=colors["A"]))
        ax.text(thresholds["A"] + 0.15, -0.3, "A", ha='center', va='bottom', fontsize=12)

        # Determine grade
        current_grade = "F"
        for g in reversed(grade_labels):
            if score >= thresholds[g]:
                current_grade = g
                break
        label_color = colors[current_grade]

        # Score line
        ax.axvline(score, color='black', linewidth=2)
        ax.text(score + 0.06, 1.25, f"{score:.2f}",
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor=label_color, boxstyle='round,pad=0.2', linewidth=2))

        # Decorations
        ax.set_xlim(0, 6)
        ax.set_ylim(-0.5, 1.6)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, 7, 1))
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_title(f"{config['title']} Scale → Grade: {current_grade}", loc='left', fontsize=12, weight='bold')
        ax.text(0.2, -0.3, "Unacceptable", fontsize=12, color="black", va="bottom")
        ax.text(5.85, -0.3, "Acceptable", fontsize=12, color="black", va="bottom", ha='right')

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_boxplots(df, learning_cols, design_cols, engagement_cols):
    for group_name, cols in [("Learning", learning_cols), ("Design", design_cols), ("Engagement", engagement_cols)]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[cols])
        plt.title(f"{group_name} Ratings Boxplot")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_score_histograms(df):
    # If no SCORE columns exist, create a general score column
    score_cols = [col for col in df.columns if "SCORE" in col]
    if not score_cols:
        df['WBLT_SCORE'] = df.mean(axis=1)
        score_cols = ['WBLT_SCORE']

    df[score_cols].hist(figsize=(10, 6), bins=10)
    plt.suptitle("Score Distributions")
    plt.tight_layout()
    plt.show()


def perform_pca_and_plot(df, learning_cols, design_cols, engagement_cols):
    all_items = df[learning_cols + design_cols + engagement_cols]
    pca = PCA(n_components=2)
    components = pca.fit_transform(all_items)
    df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2")
    plt.title("PCA: All Question Items")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.tight_layout()
    plt.show()


def plot_wblt_subscales(df, learning_cols, design_cols, engagement_cols):
    """
    Analyze and visualize WBLT subscale scores.
    - Computes average score for each subscale per respondent
    - Plots histograms, boxplots, and bar chart with error bars
    - Prints descriptive stats
    """
    # Compute per-respondent average scores
    df["Learning Score"] = df[learning_cols].mean(axis=1)
    df["Design Score"] = df[design_cols].mean(axis=1)
    df["Engagement Score"] = df[engagement_cols].mean(axis=1)

    score_cols = ["Learning Score", "Design Score", "Engagement Score"]

    # === Histogram of subscale scores ===
    df[score_cols].hist(bins=10, figsize=(10, 6))
    plt.suptitle("Distribution of WBLT Subscale Scores")
    plt.tight_layout()
    plt.show()

    # === Boxplot comparison ===
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[score_cols])
    plt.title("Comparison of WBLT Subscale Scores")
    plt.ylabel("Score (1–5)")
    plt.tight_layout()
    plt.show()

    # === Bar chart with standard deviation error bars ===
    means = df[score_cols].mean()
    stds = df[score_cols].std()
    x = np.arange(len(score_cols))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, means.values, yerr=stds.values, capsize=10)

    # Add value labels above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                 f"{height:.2f}", ha='center', va='bottom', fontsize=10)

    plt.xticks(x, score_cols)
    plt.ylim(0, 5.5)
    plt.ylabel("Average Score")
    plt.title("Average WBLT Subscale Scores with Std. Deviation")
    plt.tight_layout()
    plt.show()

    # === Descriptive statistics ===
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print("\n--- Subscale Descriptive Statistics ---")
    print(df[score_cols].describe())


def plot_sus_results(results):
    """
    Calculate, describe, and visualize SUS results
    """

    # === Descriptive Statistics ===
    print("\n--- SUS Score Descriptive Statistics ---")
    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)
    print(results["SUS Score"].describe())

    # === Histogram of SUS Scores ===
    plt.figure(figsize=(8, 5))
    sns.histplot(results["SUS Score"], bins=10, kde=True)
    plt.title("Distribution of SUS Scores")
    plt.xlabel("SUS Score (0–100)")
    plt.ylabel("Number of Participants")
    plt.tight_layout()
    plt.show()

    # === Bar plot of grades ===
    plt.figure(figsize=(6, 4))
    sns.countplot(data=results, x="Grade", order=["A", "B", "C", "D", "F"])
    plt.title("SUS Grades")
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_sus_percentile_curve_with_grades(sus_score):
    """
    Plot SUS percentile curve with grade bands, and highlight the given SUS score
    with both vertical (score) and horizontal (percentile) reference lines.

    Parameters:
        sus_score (float): The SUS score to highlight (0–100)
    """
    # Simulate SUS curve
    sus_scores = np.linspace(0, 100, 500)
    percentile_curve = 100 / (1 + np.exp(-(sus_scores - 68) / 7))

    # Calculate percentile for the given SUS score
    sus_percentile = 100 / (1 + np.exp(-(sus_score - 68) / 7))

    # Start plot
    plt.figure(figsize=(10, 6))
    plt.plot(sus_scores, percentile_curve, color='royalblue', linewidth=2)

    # Grade bands (percentile bands)
    grade_bands = [
        ('A', 80, 100),
        ('B', 70, 80),
        ('C', 50, 70),
        ('D', 20, 50),
        ('F', 0, 20)
    ]
    for label, y_min, y_max in grade_bands:
        plt.axhspan(y_min, y_max, color='lightgray', alpha=0.4)
        plt.text(2, (y_min + y_max) / 2, label,
                 va='center', ha='left', fontsize=14, weight='bold')

    # Vertical line: SUS score
    plt.axvline(x=sus_score, linestyle='--', color='crimson', linewidth=1.5)
    plt.plot(sus_score, sus_percentile, 'o', color='crimson')
    plt.text(sus_score - 1, sus_percentile + 3, f"{sus_score:.1f}", ha='right', color='crimson', fontsize=12)

    # Horizontal line: percentile rank
    plt.axhline(y=sus_percentile, linestyle='--', color='crimson', linewidth=1.2)
    plt.text(99, sus_percentile + 3, f"{sus_percentile:.0f}%", va='center', ha='right', color='crimson', fontsize=12)

    # Formatting
    # plt.title("SUS on a Curve with Percentile Ranks and Grades", fontsize=14)
    plt.xlabel("SUS Score")
    plt.ylabel("Percentile Rank")
    plt.yticks(np.arange(0, 110, 20), [f"{p}%" for p in np.arange(0, 110, 20)])
    plt.xticks(np.arange(0, 110, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_response_distribution(df, column_name, title=None, xlabel=None, ylabel="Respondent Count", round_values=False):
    """
    Plot a bar chart of response distribution for a single column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis. Default: "Respondent Count".
        round_values (bool): If True, round values to nearest int before plotting.
    """
    data = df[column_name].dropna()

    if round_values:
        data = data.round()

    plt.figure(figsize=(8, 5))
    sns.countplot(x=data, hue=data, palette="Blues", legend=False)

    plt.title(title or f"Distribution of {column_name}")
    plt.xlabel(xlabel or column_name)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()