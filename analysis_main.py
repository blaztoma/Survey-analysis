import readers
import plotters
import matplotlib

matplotlib.use('TkAgg')


def cronbach_alpha(df):
    df = df.dropna(axis=0, how='any')  # Drop rows with missing values
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    n_items = df.shape[1]
    if n_items <= 1:
        return float('nan')
    return (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)


def cronbach_alpha_sus(df):
    """
    Computes Cronbach's alpha for SUS questionnaire by first applying correct SUS scoring rules.

    Parameters:
        df (pd.DataFrame): DataFrame with columns SUS1 through SUS10 (numeric 1–5)

    Returns:
        float: Cronbach’s alpha
    """
    df = df.dropna(axis=0, how='any')  # Drop incomplete responses

    # Adjust SUS scoring: odd = score - 1; even = 5 - score
    adjusted_df = df.copy()
    for i in range(1, 11):
        col = f"SUS{i}"
        if i % 2 == 1:
            adjusted_df[col] = adjusted_df[col] - 1
        else:
            adjusted_df[col] = 5 - adjusted_df[col]

    item_vars = adjusted_df.var(axis=0, ddof=1)
    total_var = adjusted_df.sum(axis=1).var(ddof=1)
    n_items = adjusted_df.shape[1]

    if n_items <= 1:
        return float('nan')

    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha


def analyze_cronbach(df, learning_cols, design_cols, engagement_cols):
    print("\n--- Cronbach's Alpha ---")
    print("Learning α:", round(cronbach_alpha(df[learning_cols]), 3))
    print("Design α:", round(cronbach_alpha(df[design_cols]), 3))
    print("Engagement α:", round(cronbach_alpha(df[engagement_cols]), 3))


def calculate_sus_scores_and_grades(sus_df):
    """
    Calculate SUS scores, grades, adjectives, and acceptability levels for each respondent.

    Parameters:
        sus_df (pd.DataFrame): DataFrame with SUS1–SUS10 (already cleaned and numeric)

    Returns:
        pd.DataFrame: Original DataFrame + SUS calculation columns
    """
    # Identify odd and even item columns
    odd_cols = [f"SUS{i}" for i in [1, 3, 5, 7, 9]]
    even_cols = [f"SUS{i}" for i in [2, 4, 6, 8, 10]]

    # Compute SUS score per respondent
    odd_sum = sus_df[odd_cols].apply(lambda row: sum(row - 1), axis=1)
    even_sum = sus_df[even_cols].apply(lambda row: sum(5 - row), axis=1)
    sus_score = (odd_sum + even_sum) * 2.5

    result = sus_df.copy()
    result["SUS Score"] = sus_score

    # Grade scale thresholds
    def get_grade(score):
        if score >= 78.9:
            return "A"
        elif score >= 72.6:
            return "B"
        elif score >= 62.7:
            return "C"
        elif score >= 51.7:
            return "D"
        else:
            return "F"

    def get_adjective(score):
        if score >= 84.1:
            return "Best imaginable"
        elif score >= 72.6:
            return "Excellent"
        elif score >= 71.1:
            return "Good"
        elif score >= 51.7:
            return "OK/Fair"
        elif score >= 25.1:
            return "Poor"
        else:
            return "Worst imaginable"

    def get_acceptability(score):
        if score >= 51.7:
            return "Acceptable"
        elif score >= 25.1:
            return "Marginal"
        else:
            return "Not Acceptable"

    result["Grade"] = result["SUS Score"].apply(get_grade)
    result["Adjective"] = result["SUS Score"].apply(get_adjective)
    result["Acceptability"] = result["SUS Score"].apply(get_acceptability)

    return result


def apply_ipq_reverse_coding(ipq_df, short_labels):
    """
    Reverses values for inverse-coded IPQ items (1–5 scale).

    Parameters:
        ipq_df (pd.DataFrame): Cleaned IPQ DataFrame with short column names.
        short_labels (Dict[str, str]): Mapping from original column names to short labels.

    Returns:
        pd.DataFrame: IPQ DataFrame with reversed items corrected.
    """
    inverse_originals = ["SP2U", "INV3U", "REAL1U"]
    inverse_cols = [short_labels[orig] for orig in inverse_originals if orig in short_labels]

    for col in inverse_cols:
        ipq_df[col] = 6 - ipq_df[col]

    return ipq_df


def scale_ipq_to_0_6(df):
    """
    Scales all numeric values in a DataFrame from the 1–5 range to the 0–6 range.

    Parameters:
        df (pd.DataFrame): DataFrame with values in range 1–5.

    Returns:
        pd.DataFrame: Scaled DataFrame with values in range 0–6.
    """
    return ((df - 1) / 4) * 6


wblt_df, short_labels, learning_cols, design_cols, engagement_cols = readers.load_wblt_data_from_wblt_sus_ai("survey2.csv")

# analyze_cronbach(wblt_df, learning_cols, design_cols, engagement_cols)
# plotters.plot_correlation_heatmap(wblt_df)
# plotters.plot_boxplots(wblt_df, learning_cols, design_cols, engagement_cols)
# plotters.plot_score_histograms(wblt_df)
# plotters.perform_pca_and_plot(wblt_df, learning_cols, design_cols, engagement_cols)
# plotters.plot_wblt_subscales(wblt_df, learning_cols, design_cols, engagement_cols)
# print("\n--- Subscale Descriptive Statistics ---")
# print("\n--- Learning describe ---")
# print(wblt_df[learning_cols].describe())
# print("\n--- Design describe ---")
# print(wblt_df[design_cols].describe())
# print("\n--- Engagement describe ---")
# print(wblt_df[engagement_cols].describe())

# sus_df, sus_labels = readers.load_sus_data_from_survey("survey2.csv")
# sus_df, sus_labels = readers.load_sus_data_from_Egle("SUS_skale.csv")
#
# sus_with_scores = calculate_sus_scores_and_grades(sus_df)
# alpha = cronbach_alpha_sus(sus_df)
# print(f"Cronbach's alpha for SUS: {alpha:.3f}")
#
# # print(sus_with_scores[["SUS Score", "Grade", "Adjective", "Acceptability"]])
# plotters.plot_sus_results(sus_with_scores)
# avg_sus_score = sus_with_scores["SUS Score"].mean()
# print(avg_sus_score)
#
# plotters.plot_sus_percentile_curve_with_grades(avg_sus_score)

# ia_df, ia_labels, interpreter_cols, ai_cols = readers.load_interpreter_ai_data_from_survey("survey.csv")
#
# print(ia_df.head())
# print("Interpreter questions:", interpreter_cols)
# print("AI questions:", ai_cols)
#
# print("Average Q1:", ia_df["INT4"].mean() / 5)
# plotters.plot_response_distribution(ia_df, "INT1", title="Programming tasks cover the school curriculum", xlabel="Score (0–10)", round_values=True)

filename = "ipq_drivers.csv"
real_ipq_df, short_labels, general_col, spatial_cols, involvement_cols, realism_cols = readers.load_ipq_data_from_csv(filename)
scaled_ipq_df = scale_ipq_to_0_6(real_ipq_df)
ipq_df = apply_ipq_reverse_coding(scaled_ipq_df, short_labels)

# Print Cronbach's alpha for each subscale
print("\n--- Cronbach's Alpha ---")
print("General presence α:", round(cronbach_alpha(ipq_df[general_col]), 3))
print("Spatial presence α:", round(cronbach_alpha(ipq_df[spatial_cols]), 3))
print("Involvement α:", round(cronbach_alpha(ipq_df[involvement_cols]), 3))
print("Realism α:", round(cronbach_alpha(ipq_df[realism_cols]), 3))

all_cols = general_col + spatial_cols + involvement_cols + realism_cols
print("Total IPQ α:", round(cronbach_alpha(ipq_df[all_cols]), 3))

general_mean = ipq_df[general_col].mean().iloc[0]
spatial_mean = ipq_df[spatial_cols].mean(axis=1).mean()
involvement_mean = ipq_df[involvement_cols].mean(axis=1).mean()
realism_mean = ipq_df[realism_cols].mean(axis=1).mean()

# Print results
print(f"General Presence: {general_mean:.2f}")
print(f"Spatial Presence: {spatial_mean:.2f}")
print(f"Involvement: {involvement_mean:.2f}")
print(f"Experienced Realism: {realism_mean:.2f}")

# plotters.plot_presence_score(general_mean, scale_type="general")
# plotters.plot_presence_score(spatial_mean, scale_type="spatial")
# plotters.plot_presence_score(involvement_mean, scale_type="involvement")
# plotters.plot_presence_score(realism_mean, scale_type="experienced_realism")
plotters.plot_all_presence_scores({
    "general": general_mean,
    "spatial": spatial_mean,
    "involvement": involvement_mean,
    "experienced_realism": realism_mean
})


