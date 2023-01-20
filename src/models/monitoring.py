import numpy as np
import omegaconf
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.utilities.helpers import load_last_predictions, load_train_data

cfg = omegaconf.OmegaConf.load("conf/config.yaml")


def main() -> None:
    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    last_predictions = load_last_predictions(samples=10)
    train_dataset = load_train_data(cfg.dataset)

    with np.errstate(divide="ignore", invalid="ignore"):
        # as some of the features only consists of 0 we get a "divide by 0" warning
        # I believe evidently is handling this error, so I have muted the warning here.
        data_drift_report.run(
            current_data=last_predictions,
            reference_data=train_dataset.iloc[
                :10
            ],  # only 10 test samples to reduce running time
            column_mapping=None,
        )
    data_drift_report.save_html("static/drift.html")


if __name__ == "__main__":
    main()
