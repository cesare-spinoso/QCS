import os
from datetime import datetime

import hydra
import pandas as pd
import jsonlines
from omegaconf import DictConfig, OmegaConf

from qcs import SRC_DIRECTORY
from qcs.utils.decorators import main_decorator
from dataset_maker import QCSDatasetMaker
from datasets import ClassificationDataset
from extended_xml_code_span import ExtendedXmlCodeSpan


def dataset_to_json(cfg: DictConfig, x: dict[str, list[str]], y: pd.DataFrame) -> dict:
    data_features = {k: x[k] for k, v in cfg["write_dataset"].items() if v}
    y_ = {"y": y.to_dict(orient="list")}
    return {**data_features, **y_}


@hydra.main(version_base=None, config_path="./conf", config_name="config")
@main_decorator
def main(run_name: str, cfg: DictConfig) -> None:
    if cfg["dataset_maker"]["type"] == "qcs":
        dataset_maker = QCSDatasetMaker(
            xml_directory=cfg["dataset_maker"]["xml_directory"],
            coder=cfg["dataset_maker"]["coder"],
            interviews=cfg["dataset_maker"]["interviews"],
            left_context=cfg["dataset_maker"]["left_context"],
            right_context=cfg["dataset_maker"]["right_context"],
            previous_question=cfg["dataset_maker"]["previous_question"],
            train_val_split=cfg["dataset_maker"]["train_val_split"],
        )
    else:
        raise ValueError("Dataset maker type not supported.")
    extended_xml_codespan_ds = dataset_maker.prepare()
    prepared_datasets: list[ClassificationDataset] = []
    for dataset in extended_xml_codespan_ds.enumerate():
        ds = ClassificationDataset(
            interview_id=dataset.interview_id,
            train=dataset.train,
            val=dataset.val,
            test=dataset.test,
        )
        ds.prepare_neural(
            classifier_type=cfg["prepare_dataset"]["classifier_type"],
            span_text=cfg["prepare_dataset"]["span_text"],
            left_context=cfg["prepare_dataset"]["left_context"],
            right_context=cfg["prepare_dataset"]["right_context"],
            previous_question=cfg["prepare_dataset"]["previous_question"],
            code_description=cfg["prepare_dataset"]["code_description"],
            max_length=cfg["prepare_dataset"]["max_length"],
        )
        prepared_datasets.append(ds)
    json_list = []
    for dataset in prepared_datasets:
        x_train, y_train = dataset.get_training_data()
        x_val, y_val = dataset.get_validation_data()
        x_test, y_test = dataset.get_test_data()
        json_dict = {
            "interview_id": dataset.interview_id,
            "train": dataset_to_json(cfg, x_train, y_train),
            "val": dataset_to_json(cfg, x_val, y_val),
            "test": dataset_to_json(cfg, x_test, y_test),
        }
        json_list.append(json_dict)
    with jsonlines.open(
        os.path.join(cfg["output_directory"], f"{run_name}.jsonl"), "w"
    ) as writer:
        writer.write_all(json_list)


if __name__ == "__main__":
    main()
