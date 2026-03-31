import pandas as pd
from sklearn.feature_extraction import DictVectorizer


BLUE_COLS = ["blue_1", "blue_2", "blue_3", "blue_4", "blue_5"]
RED_COLS = ["red_1", "red_2", "red_3", "red_4", "red_5"]
ALL_COLS = BLUE_COLS + RED_COLS


def normalize_hero_name(name: str) -> str:
    if pd.isna(name):
        return ""
    return str(name).strip().lower().replace("-", " ").replace("_", " ")


def validate_dataframe(df: pd.DataFrame) -> None:
    required = ALL_COLS + ["winner"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    bad_winners = df[~df["winner"].isin(["blue", "red"])]
    if not bad_winners.empty:
        raise ValueError("Column 'winner' must contain only 'blue' or 'red'.")


def row_to_features(row: pd.Series) -> dict:
    features = {}

    for col in BLUE_COLS:
        hero = normalize_hero_name(row[col])
        if hero:
            features[f"blue_{hero}"] = 1

    for col in RED_COLS:
        hero = normalize_hero_name(row[col])
        if hero:
            features[f"red_{hero}"] = 1

    return features


def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    validate_dataframe(df)

    for col in ALL_COLS:
        df[col] = df[col].apply(normalize_hero_name)

    df["winner"] = df["winner"].str.strip().str.lower()

    feature_dicts = df.apply(row_to_features, axis=1).tolist()
    labels = df["winner"].map({"red": 0, "blue": 1}).values

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(feature_dicts)

    return X, labels, vectorizer, df


def prepare_single_draft(blue_team: list, red_team: list, vectorizer: DictVectorizer):
    if len(blue_team) != 5 or len(red_team) != 5:
        raise ValueError("Both blue_team and red_team must contain exactly 5 heroes.")

    features = {}

    for hero in blue_team:
        hero = normalize_hero_name(hero)
        if hero:
            features[f"blue_{hero}"] = 1

    for hero in red_team:
        hero = normalize_hero_name(hero)
        if hero:
            features[f"red_{hero}"] = 1

    X = vectorizer.transform([features])
    return X
