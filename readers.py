import pandas as pd


def load_sus_data_from_survey(filename: str):
    """
    Load SUS data (questions 14–23) from survey CSV, clean it, and return structured data.

    Returns:
        sus_df (pd.DataFrame): Cleaned SUS responses with short labels SUS1–SUS10.
        sus_labels (Dict[str, str]): Mapping of full SUS question texts to SUS1–SUS10.
    """
    df = pd.read_csv(filename, quotechar='"', sep=',', encoding='utf-8')
    df.columns = df.columns.str.strip()

    # SUS is assumed to be columns 14–23 (i.e., columns[14:24])
    sus_columns = df.columns[14:24]
    sus_df = df[sus_columns].copy()

    # Clean and convert to float
    sus_df = sus_df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x).astype(float)

    # Create short labels SUS1–SUS10
    sus_labels = {col: f"SUS{i+1}" for i, col in enumerate(sus_columns)}
    sus_df.rename(columns=sus_labels, inplace=True)

    return sus_df, sus_labels


def load_wblt_data_from_wblt_sus_ai(filename: str):
    """
    Load WBLT data (first 13 survey questions) from CSV, clean it, and return structured data.

    Returns:
        wblt_df (pd.DataFrame): Cleaned DataFrame for WBLT.
        wblt_columns (List[str])
        learning_cols (List[str])
        design_cols (List[str])
        engagement_cols (List[str])
    """
    df = pd.read_csv(filename, quotechar='"', sep=',', encoding='utf-8')
    df.columns = df.columns.str.strip()

    wblt_columns = df.columns[1:14]
    wblt_df = df[wblt_columns].copy()
    wblt_df = wblt_df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x).astype(float)

    short_labels = {col: f"Q{i+1}" for i, col in enumerate(wblt_columns)}
    wblt_df.rename(columns=short_labels, inplace=True)

    clean_cols = list(wblt_df.columns)
    learning_cols = clean_cols[:5]
    design_cols = clean_cols[5:9]
    engagement_cols = clean_cols[9:13]

    return wblt_df, short_labels, learning_cols, design_cols, engagement_cols


def load_interpreter_ai_data_from_survey(filename: str):
    """
    Load Interpreter and AI subscale data from the survey.
    Handles mixed values: numbers and 'Taip'/'Ne' (Yes/No).

    Returns:
        df_clean (pd.DataFrame): Cleaned DataFrame with short codes (INT1–INT4, AI1–AI5)
        labels (Dict[str, str]): Mapping of full original question to short code
        interpreter_cols (List[str])
        ai_cols (List[str])
    """
    import pandas as pd

    df = pd.read_csv(filename, quotechar='"', sep=',', encoding='utf-8')
    df.columns = df.columns.str.strip()

    # Column ranges based on your file structure
    interpreter_raw = df.columns[24:28]
    ai_raw = df.columns[28:33]

    selected = df[interpreter_raw.tolist() + ai_raw.tolist()].copy()

    # Map 'Taip' to 1, 'Ne' to 0
    def convert_mixed(x):
        if isinstance(x, str):
            x = x.strip()
            if x.lower() == 'taip':
                return 1
            elif x.lower() == 'ne':
                return 0
            x = x.replace(',', '.')  # handle decimal commas
        try:
            return float(x)
        except:
            return pd.NA

    selected = selected.applymap(convert_mixed)

    # Rename columns
    labels = {}
    for i, col in enumerate(interpreter_raw):
        labels[col] = f"INT{i+1}"
    for i, col in enumerate(ai_raw):
        labels[col] = f"AI{i+1}"

    df_clean = selected.rename(columns=labels)
    interpreter_cols = [f"INT{i+1}" for i in range(len(interpreter_raw))]
    ai_cols = [f"AI{i+1}" for i in range(len(ai_raw))]

    return df_clean, labels, interpreter_cols, ai_cols
