import logging
import time

import pandas as pd
import torch
from google.cloud import bigquery


def save_to_db(data, pred, index) -> None:
    logger = logging.getLogger(__name__)
    data_x = [data.x[index]]
    data_y = [data.y[index]]
    data_y_hat = [pred]

    client = bigquery.Client()
    table_id = "hybrid-essence-236114.model_prediction_log.model_prediction_log"

    row_to_insert = make_data_sql_friendly(data_x, data_y, data_y_hat)

    errors = client.insert_rows_json(table_id, row_to_insert)
    if errors:
        logger.info(
            f"Error: {str(errors)}"
        )  # I am a bit unsure how exactly logger works with FastAPI


def make_data_sql_friendly(data_x, data_y, data_y_hat):
    """This is not the best way but it works for now.
    As we are currently storing the >1400 features as a json string in one column
    in the sql table this is necessary."""

    row_to_insert = []
    for x, y, y_hat in zip(data_x, data_y, data_y_hat):
        row_to_insert.append(
            {
                "TIMESTAMP": time.time(),
                "INPUT": pd.DataFrame(x.numpy()).to_json(orient="values"),
                "OUTPUT": y.numpy().item(),
                "LABEL": y_hat.numpy().item(),
            }
        )

    return row_to_insert


def load_last_predictions(samples: int = 10) -> pd.DataFrame:
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

    return to_df(results)


def to_df(table: dict) -> pd.DataFrame:
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


def load_train_data(path_to_dataset: str) -> pd.DataFrame:
    data = torch.load(path_to_dataset)[0]
    data_x = data.x[data.train_mask]
    data_y = data.y[data.train_mask]
    data_y_hat = data.y[data.train_mask]

    return to_df(make_data_sql_friendly(data_x, data_y, data_y_hat))
