from flask import Flask, request, jsonify, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import numpy as np
import os
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RESULT_FOLDER = 'filled'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -------- CUSTOM KNN IMPUTATION FUNCTION --------
def knn_impute_manual(df, k=3):
    df = df.copy()
    for row_idx in df.index:
        row = df.loc[row_idx]
        if row.isnull().any():
            for col in df.columns:
                if pd.isnull(row[col]):
                    candidate_rows = df[df[col].notnull()]
                    distances = []
                    for idx, candidate in candidate_rows.iterrows():
                        common_cols = row.notnull() & candidate.notnull()
                        if common_cols.sum() == 0:
                            continue
                        dist = np.linalg.norm(row[common_cols] - candidate[common_cols])
                        distances.append((idx, dist))
                    neighbors = sorted(distances, key=lambda x: x[1])[:k]
                    if neighbors:
                        values = [df.loc[n[0], col] for n in neighbors]
                        df.at[row_idx, col] = np.mean(values)
                    else:
                        df.at[row_idx, col] = df[col].mean()
    return df

# -------- FAST KNN IMPUTATION FUNCTION --------
def knn_impute_fast(df, k=3):
    imputer = KNNImputer(n_neighbors=k)
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns)

# -------- MICE IMPUTATION FUNCTION --------
def mice_impute(df, max_iter=5):
    df = df.copy()
    df_initial = df.fillna(df.mean())
    for iteration in range(max_iter):
        for col in df.columns:
            missing_rows = df[col].isnull()
            if missing_rows.sum() == 0:
                continue
            not_missing = ~missing_rows
            X_train = df_initial.loc[not_missing, df.columns != col]
            y_train = df_initial.loc[not_missing, col]
            X_test = df_initial.loc[missing_rows, df.columns != col]
            if X_train.isnull().any().any() or y_train.isnull().any():
                continue
            model = LinearRegression()
            model.fit(X_train, y_train)
            df_initial.loc[missing_rows, col] = model.predict(X_test.fillna(df_initial.mean()))
    for col in df.columns:
        if df_initial[col].isnull().any():
            df_initial[col].fillna(df[col].mean(), inplace=True)
    return df_initial

# -------- FILE UPLOAD ROUTE --------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded."})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."})

    try:
        df = pd.read_csv(file)
        df_numeric = df.select_dtypes(include=["number"])
        df_non_numeric = df.select_dtypes(exclude=["number"])

        # Fast KNN Imputation
        start_knn = time.time()
        df_knn_result = knn_impute_fast(df_numeric, k=3)
        print("Fast KNN completed in", time.time() - start_knn, "seconds")

        # MICE Imputation
        start_mice = time.time()
        df_mice_result = mice_impute(df_numeric, max_iter=5)
        print("MICE completed in", time.time() - start_mice, "seconds")

        df_knn_full = pd.concat([df_non_numeric.reset_index(drop=True), df_knn_result.reset_index(drop=True)], axis=1)
        df_mice_full = pd.concat([df_non_numeric.reset_index(drop=True), df_mice_result.reset_index(drop=True)], axis=1)

        knn_path = os.path.join(RESULT_FOLDER, "filled_knn.csv")
        mice_path = os.path.join(RESULT_FOLDER, "filled_mice.csv")
        df_knn_full.to_csv(knn_path, index=False)
        df_mice_full.to_csv(mice_path, index=False)

        return jsonify({
            "success": True,
            "message": "File processed successfully."
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# -------- FILE DOWNLOAD ROUTE --------
@app.route("/download/<method>")
def download_file(method):
    if method == "knn":
        path = os.path.join(RESULT_FOLDER, "filled_knn_manual.csv")
    elif method == "mice":
        path = os.path.join(RESULT_FOLDER, "filled_mice_manual.csv")
    else:
        return jsonify({"success": False, "message": "Invalid method."})
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)