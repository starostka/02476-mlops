import pandas as pd
from google.cloud import bigquery


def load_train_data(train_file) -> pd.DataFrame:

    pass


def load_last_predictions(samples: int = 10):
    client = bigquery.Client()
    query_job = client.query(
        f"""
    SELECT *
    FROM hybrid-essence-236114.model_prediction_log.model_prediction_log
    ORDER BY TIMESTAMP DESC
    LIMIT {samples}
    """
    )
    results = query_job.result()

    return results


def to_df(table):
    """Change from sql table structure to pd where each feature has its own column"""
    list_of_dicts = []
    for r in table:
        d = {
            # "timestamp": r["TIMESTAMP"],
            "output": r["OUTPUT"],
            "label": r["LABEL"],
        }

        for i, f in pd.read_json(r["INPUT"], orient="values").iterrows():
            d[f"feature_{i}"] = f[0]

        list_of_dicts.append(d)

    return pd.DataFrame.from_records(list_of_dicts)


if __name__ == "__main__":
    table = load_last_predictions(10)
    to_df(table)
