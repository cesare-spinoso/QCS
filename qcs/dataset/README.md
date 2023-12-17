Once you have made a request for our dataset and we have approved it, we will send you a folder with 15 `.xml` files. You can use our script, `write_dataset.py`, to extract the annotations and place them into a `jsonlines` file. To do so, you must have `hydra` installed as `write_dataset` works using a config file. The config file should have the following format. *Note: All paths must be relative to the `QCS` repository path*.

```
dataset_maker:
  type: qcs
  xml_directory: data/raw_xml/
  coder: 2 # Coder 1 or 2
  interviews:
    # This is the order which the coders used to annotate the interview transcripts
    # Interview 15 was never annotated as the data had to be dropped
    - 1
    - 2
    - 3
    - 4
    - 5
    - 6
    - 7
    - 8
    - 9
    - 10
    - 11
    - 12
    - 13
    - 14
    - 16
  left_context: 1000 # How much text (space-separated tokens) to the left of the code span should be kept
  right_context: 1000 # How much text (space-separated tokens) to the right of the code span should be kept
  previous_question: True # Should the most recently asked question be extracted?
  previous_question_number: True # Should the question number also be extracted?
  train_val_split: 0.8 # What proportion of the data in the first 1 ... K interviews should be reserved for training
# You can set all the features below to True, this ensures that data preprocessing is done on all the features
prepare_dataset:
  span_text: True
  left_context: True
  right_context: True
  previous_question: True
  previous_question_number: True
features_to_write:
  span_text: True
  left_context: True
  right_context: True
  previous_question: True
  previous_question_number: True
output_directory: data/jsonl_data/
# This will log your config file to the directory below using the same name as the created dataset
# for ease of traceability
write_config_directory: qcs/dataset/config_instances/
```

To run the dataset maker, run the `run_dataset_maker.py` script with the following 
```
python run_dataset_maker.py --config-path=<full path to config file directory> --config-name=<name of config file>
```