import os

import hydra
import pandas as pd
import jsonlines
from omegaconf import DictConfig
from qcs import QCS_DIRECTORY_PATH

from qcs.utils.decorators import main_decorator
from dataset_maker import QCSDatasetMaker
from qcs.dataset.custom_dataclasses import XmlCodeSpanInterviewSplit


def interview_split_to_json(
    cfg: DictConfig, x: dict[str, list], y: pd.DataFrame
) -> dict:
    data_features = {k: x[k] for k, v in cfg["features_to_write"].items() if v}
    y_ = {"y": y.to_dict(orient="list")}
    return {**data_features, **y_}


@hydra.main(version_base=None)
@main_decorator
def main(run_name: str, cfg: DictConfig) -> None:
    # Make dataset from xml interview files
    if cfg["dataset_maker"]["type"] == "qcs":
        dataset_maker = QCSDatasetMaker(
            xml_directory=os.path.join(
                QCS_DIRECTORY_PATH, cfg["dataset_maker"]["xml_directory"]
            ),
            coder=cfg["dataset_maker"]["coder"],
            interviews=cfg["dataset_maker"]["interviews"],
            left_context=cfg["dataset_maker"]["left_context"],
            right_context=cfg["dataset_maker"]["right_context"],
            previous_question=cfg["dataset_maker"]["previous_question"],
            train_val_split=cfg["dataset_maker"]["train_val_split"],
        )
    else:
        raise ValueError("Dataset maker type not supported.")
    xml_codespan_ds = dataset_maker.make_dataset()
    # Prepare each interview split
    prepared_interview_splits: list[XmlCodeSpanInterviewSplit] = []
    for interview_split in xml_codespan_ds.enumerate():
        interview_split.prepare(
            span_text=cfg["prepare_dataset"]["span_text"],
            left_context=cfg["prepare_dataset"]["left_context"],
            right_context=cfg["prepare_dataset"]["right_context"],
            previous_question=cfg["prepare_dataset"]["previous_question"],
            previous_question_number=cfg["prepare_dataset"]["previous_question_number"],
        )
        prepared_interview_splits.append(interview_split)
    # Place prepared interview splits into json file
    json_list = []
    for xml_interview_split in prepared_interview_splits:
        x_train, y_train = xml_interview_split.get_training_data()
        x_val, y_val = xml_interview_split.get_validation_data()
        x_test, y_test = xml_interview_split.get_test_data()
        json_dict = {
            "interview_id": xml_interview_split.interview_id,
            "train": interview_split_to_json(cfg, x_train, y_train),
            "val": interview_split_to_json(cfg, x_val, y_val),
            "test": interview_split_to_json(cfg, x_test, y_test),
        }
        json_list.append(json_dict)
    with jsonlines.open(
        os.path.join(QCS_DIRECTORY_PATH, cfg["output_directory"], f"{run_name}.jsonl"),
        "w",
    ) as writer:
        writer.write_all(json_list)


if __name__ == "__main__":
    main()
