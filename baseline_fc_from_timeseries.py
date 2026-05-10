from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

ANATOMICAL_DIR = Path(
    r"C:\Users\mayag\Desktop\MAYA\University\Year 3\סדנת מחקר מדעי הנתונים\מודל\timeseries\anatomical"
)

TIMESERIES_DIR = ANATOMICAL_DIR.parent
GLOBAL_DIR = TIMESERIES_DIR / "global"

CAPS_CSV = Path(__file__).resolve().parent / "CAPS_scores.csv"


FEATURE_TYPES = [
    "global_schaefer400",
    "global_tian",
    "anatomical_schaefer400",
    "anatomical_tian",
]


def load_caps(path: Path = CAPS_CSV) -> pd.Series:
    df = pd.read_csv(path).dropna()

    return df.set_index("subject_id")["CAPS_score"]


def fc_vector(csv_path: Path) -> np.ndarray:
    timeseries = pd.read_csv(csv_path)

    correlation_matrix = timeseries.corr(numeric_only=True).values

    i, j = np.triu_indices_from(correlation_matrix, k=1)

    return correlation_matrix[i, j]


def feature_type(folder_name: str, fname: str) -> Optional[str]:
    low = fname.lower()

    if "schaefer400" in low:
        atlas = "schaefer400"
    elif "tian" in low:
        atlas = "tian"
    else:
        return None

    return f"{folder_name}_{atlas}"


def get_subject_id(file_path: Path) -> str:
    return file_path.name.split("_")[0]


def build_datasets(
    caps_path: Path = CAPS_CSV,
    timeseries_dir: Path = TIMESERIES_DIR,
) -> Tuple[Dict, Dict]:

    caps = load_caps(caps_path)

    data = {ft: [] for ft in FEATURE_TYPES}
    mri_subjects = set()

    for mapping_type in ["global", "anatomical"]:
        folder = timeseries_dir / mapping_type

        for file_path in sorted(folder.glob("*ses-MRI1*")):
            subject_id = get_subject_id(file_path)
            mri_subjects.add(subject_id)

            ft = feature_type(mapping_type, file_path.name)

            if ft not in data:
                continue

            if subject_id not in caps.index:
                continue

            data[ft].append({
                "subject_id": subject_id,
                "X": fc_vector(file_path),
                "y": float(caps[subject_id]),
            })

    caps_subjects = set(caps.index.astype(str))

    report = {
        "in_caps_no_mri": sorted(caps_subjects - mri_subjects),
        "mri_no_caps": sorted(mri_subjects - caps_subjects),
    }

    datasets = {}

    for ft, rows in data.items():
        datasets[ft] = {
            "X": np.array([row["X"] for row in rows]),
            "y": np.array([row["y"] for row in rows]),
            "subject_ids": [row["subject_id"] for row in rows],
        }

    return datasets, report


def run_cv(X: np.ndarray, y: np.ndarray, n_features: int) -> Dict:
    pipeline = Pipeline([
        ("select", SelectKBest(score_func=f_regression, k=n_features)),
        ("model", GradientBoostingRegressor(random_state=42)),
    ])

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42,
    )

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in cv.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    pearson_r = pearsonr(y_true_all, y_pred_all).statistic
    r2 = r2_score(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    return {
        "pearson_r": pearson_r,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }

K_VALUES = [5, 10, 15, 20, 50, 100,200]

def run_all_cv(datasets: Dict) -> pd.DataFrame:
    results = []

    for feature_type, dataset in datasets.items():
        X = dataset["X"]
        y = dataset["y"]
        n_subjects = len(y)

        for k in K_VALUES:
            scores = run_cv(X, y, n_features=k)

            results.append({
                "feature_type": feature_type,
                "n_subjects": n_subjects,
                "n_features": k,
                "pearson_r": scores["pearson_r"],
                "r2": scores["r2"],
                "mae": scores["mae"],
                "rmse": scores["rmse"],
            })

    return pd.DataFrame(results)

def main():
    datasets, report = build_datasets()

    print("Subjects in CAPS but missing MRI:")
    print(report["in_caps_no_mri"])

    print("\nSubjects with MRI but missing CAPS:")
    print(report["mri_no_caps"])

    print("\nDataset sizes:")
    for feature_type, dataset in datasets.items():
        print(
            feature_type,
            "X:", dataset["X"].shape,
            "y:", dataset["y"].shape,
        )

    results = run_all_cv(datasets)

    output_path = Path(__file__).resolve().parent / "cv_results.csv"
    results.to_csv(output_path, index=False)

    print("\nResults:")
    print(results)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()