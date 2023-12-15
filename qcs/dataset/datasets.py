import copy
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from src.task_definition.extended_xml_code_span import ExtendedXmlCodeSpan

from src.utils.interview import INTERVIEW_IDS

NUM_QUESTIONS = 20


class Dataset:
    pass


@dataclass
class ExtendedXmlCodeSpanDataset(Dataset):
    interview_ids: list[str]
    train: list[list[ExtendedXmlCodeSpan]]
    val: list[list[ExtendedXmlCodeSpan]]
    test: list[list[ExtendedXmlCodeSpan]]

    def __post_init__(self) -> None:
        assert (
            (len(self.interview_ids) - 1)
            == len(self.train)
            == len(self.val)
            == len(self.test)
        ), print(
            f"{len(self.interview_ids)} != {len(self.train)} != {len(self.val)} != {len(self.test)}"
        )

    def __len__(self):
        return len(self.interview_ids[:-1])

    def enumerate(self):
        @dataclass
        class ExtendedXmlCodeSpanElement:
            interview_id: str
            train: list[ExtendedXmlCodeSpan]
            val: list[ExtendedXmlCodeSpan]
            test: list[ExtendedXmlCodeSpan]

        for i, interview_id in enumerate(self.interview_ids[:-1]):
            yield ExtendedXmlCodeSpanElement(
                interview_id=interview_id,
                train=self.train[i],
                val=self.val[i],
                test=self.test[i],
            )


@dataclass
class ClassificationDataset(Dataset):
    interview_id: str
    """All transcripts up to interview_id are part of the training set."""
    train: list[ExtendedXmlCodeSpan]
    val: list[ExtendedXmlCodeSpan]
    test: list[ExtendedXmlCodeSpan]
    is_prepared: bool = False

    def __post_init__(self) -> None:
        assert self.interview_id in INTERVIEW_IDS
        # TODO: Assert that everything in the train is up to interview_id
        # TODO: Assert that everything in val and test is after interview_id
        # assert all()

    def __get_state__(self):
        state = self.__dict__.copy()
        return state

    def __set_state__(self, state):
        self.__dict__.update(state)

    def _convert_train_codes_to_labels(self) -> None:
        """Converts singleton code to novel label."""
        # Count the codes in the training set
        code_counts = Counter([code_span.code for code_span in self.train])
        # Convert codes that appear only once to "novel" bucket
        for code_span in self.train:
            if code_counts[code_span.code] == 1:
                code_span.label = "novel"
            else:
                code_span.label = code_span.code
        self.converted_train_codes_to_labels = True

    def _convert_test_codes_to_labels(self) -> None:
        """Convert unseen codes in test to novel label."""
        assert (
            hasattr(self, "converted_train_codes_to_labels")
            and self.converted_train_codes_to_labels
        )
        # Gather the training codes
        train_labels = set()
        for code_span in self.train:
            train_labels.add(code_span.label)
        # Convert codes outside of train to "novel" bucket
        for code_span in self.val + self.test:
            if code_span.code not in train_labels:
                code_span.label = "novel"
            else:
                code_span.label = code_span.code

    def _get_index(
        self,
        code_span: ExtendedXmlCodeSpan,
        code_spans: list[ExtendedXmlCodeSpan],
        ignore_code: bool = False,
        ignore_label: bool = False,
    ) -> int:
        for idx, code_span_ in enumerate(code_spans):
            if code_span.__eq__(
                code_span_, ignore_code=ignore_code, ignore_label=ignore_label
            ):
                return idx
        return -1

    def _merge_code_spans(
        self, code_spans: list[ExtendedXmlCodeSpan]
    ) -> list[ExtendedXmlCodeSpan]:
        merged_code_spans = []
        for code_span in code_spans:
            idx = self._get_index(
                code_span, merged_code_spans, ignore_label=True, ignore_code=True
            )
            if idx == -1:
                span_copy = copy.deepcopy(code_span)
                span_copy.code = [code_span.code]
                span_copy.label = [code_span.label]
                merged_code_spans.append(span_copy)
            else:
                if code_span.label not in merged_code_spans[idx].label:
                    merged_code_spans[idx].label.append(code_span.label)
                if code_span.code not in merged_code_spans[idx].code:
                    merged_code_spans[idx].code.append(code_span.code)
        return merged_code_spans

    def _merge_duplicate_data_points(self) -> None:
        """Merges data points with the same text but different codes and/or labels."""
        self.train = self._merge_code_spans(self.train)
        self.val = self._merge_code_spans(self.val)
        self.test = self._merge_code_spans(self.test)

    def get_num_labels(self, exclude_novel: bool = False) -> int:
        assert self.is_prepared, "Notion of labels DNE for unprepared dataset"
        return len(
            {
                v
                for v in self.labeler.idx2label.values()
                if not exclude_novel or v != "novel"
            }
        )

    def prepare_non_neural(
        self,
        classifier_type: str,
        span_text: bool,
        left_context: bool,
        right_context: bool,
        previous_question: bool,
        previous_question_number: bool,
    ) -> None:
        assert not self.is_prepared, "Dataset is already prepared"
        assert classifier_type in [
            "binary",
            "multiclass",
        ], "classifier_type must be binary or multiclass"
        # Convert codes to labels
        self._convert_train_codes_to_labels()
        self._convert_test_codes_to_labels()
        # Merge duplicate data points
        if classifier_type == "binary":
            self._merge_duplicate_data_points()
        # Vectorize text
        vectorizer = NonNeuralVectorizer(
            span_text=span_text,
            left_context=left_context,
            right_context=right_context,
            previous_question=previous_question,
            previous_question_number=previous_question_number,
        )
        # Vectorize text
        vectorizer.fit(self.train)
        self.vectorizer = vectorizer
        # Create labeler
        labeler = CustomLabeler(label_type=classifier_type)
        labeler.fit(self.train)
        self.labeler = labeler
        self.is_prepared = True
        self.data_type = "non-neural"

    def prepare_neural(
        self,
        classifier_type: str,
        span_text: bool,
        left_context: bool,
        right_context: bool,
        previous_question: bool,
        code_description: bool,
        max_length: int,
    ) -> None:
        assert not self.is_prepared, "Dataset is already prepared"
        assert classifier_type in [
            "binary",
            "multiclass",
        ], "classifier_type must be binary or multiclass"
        # Convert codes to labels
        self._convert_train_codes_to_labels()
        self._convert_test_codes_to_labels()
        # Merge duplicate data points
        if classifier_type == "binary":
            self._merge_duplicate_data_points()
        # Create labeler
        labeler = CustomLabeler(label_type=classifier_type)
        labeler.fit(self.train)
        self.labeler = labeler
        # Create neural vectorize
        vectorizer = NeuralVectorizer(
            span_text=span_text,
            left_context=left_context,
            right_context=right_context,
            previous_question=previous_question,
            code_description=code_description,
            max_length=max_length,
        )
        self.vectorizer = vectorizer
        self.is_prepared = True
        self.data_type = "neural"

    def prepare_retriever(
        self,
        classifier_type: str,
        span_text: bool,
        left_context: bool,
        right_context: bool,
        previous_question: bool,
        tokenization_method: str,
        lowercase: bool,
    ) -> None:
        assert not self.is_prepared, "Dataset is already prepared"
        assert classifier_type == "binary"
        # Convert codes to labels
        self._convert_train_codes_to_labels()
        self._convert_test_codes_to_labels()
        # Merge duplicate data points
        self._merge_duplicate_data_points()
        # Create labeler
        labeler = CustomLabeler(label_type=classifier_type)
        labeler.fit(self.train)
        self.labeler = labeler
        # Create neural vectorize
        vectorizer = RetrieverVectorizer(
            span_text=span_text,
            left_context=left_context,
            right_context=right_context,
            previous_question=previous_question,
            tokenization_method=tokenization_method,
            lowercase=lowercase,
        )
        self.vectorizer = vectorizer
        self.is_prepared = True
        self.data_type = "retriever"

    def _get_data_non_neural(
        self, code_spans: list[ExtendedXmlCodeSpan]
    ) -> tuple[np.ndarray, pd.DataFrame]:
        assert (
            hasattr(self, "data_type") and self.data_type == "non-neural"
        ), "Dataset must be prepared using the non-neural method before use"
        X = self.vectorizer.transform(code_spans)
        Y = self.labeler.transform(code_spans)
        if "novel" in Y.columns:
            Y = Y[[col for col in Y.columns if col != "novel"] + ["novel"]]
        return X, Y

    def _get_data_neural(
        self, code_spans: list[ExtendedXmlCodeSpan]
    ) -> tuple[list[str], pd.DataFrame]:
        assert (
            hasattr(self, "data_type") and self.data_type == "neural"
        ), "Dataset must be prepared using the neural method before use"
        X = self.vectorizer.fit_transform(code_spans)
        Y = self.labeler.transform(code_spans)
        return X, Y

    def _get_data_retriever(
        self, code_spans: list[ExtendedXmlCodeSpan]
    ) -> tuple[list[list[str]], pd.DataFrame]:
        assert (
            hasattr(self, "data_type") and self.data_type == "retriever"
        ), "Dataset must be prepared using the retriever method before use"
        X = self.vectorizer.fit_transform(code_spans)
        Y = self.labeler.transform(code_spans)
        return X, Y

    def get_training_data(self) -> tuple[np.ndarray, pd.DataFrame]:
        assert hasattr(self, "data_type"), "Dataset must be prepared before use"
        if self.data_type == "non-neural":
            return self._get_data_non_neural(self.train)
        elif self.data_type == "neural":
            return self._get_data_neural(self.train)
        elif self.data_type == "retriever":
            return self._get_data_retriever(self.train)

    def get_validation_data(self) -> tuple[np.ndarray, pd.DataFrame]:
        assert hasattr(self, "data_type"), "Dataset must be prepared before use"
        if all(isinstance(elt, list) and len(elt) == 0 for elt in self.val):
            return None, None
        if self.data_type == "non-neural":
            return self._get_data_non_neural(self.val)
        elif self.data_type == "neural":
            return self._get_data_neural(self.val)
        elif self.data_type == "retriever":
            return self._get_data_retriever(self.val)

    def get_test_data(self) -> tuple[np.ndarray, pd.DataFrame]:
        assert hasattr(self, "data_type"), "Dataset must be prepared before use"
        if self.data_type == "non-neural":
            return self._get_data_non_neural(self.test)
        elif self.data_type == "neural":
            return self._get_data_neural(self.test)
        elif self.data_type == "retriever":
            return self._get_data_retriever(self.test)


class NonNeuralVectorizer:
    def __init__(
        self,
        span_text: bool,
        left_context: bool,
        right_context: bool,
        previous_question: bool,
        previous_question_number: bool,
    ):
        assert isinstance(span_text, bool)
        assert isinstance(left_context, bool)
        assert isinstance(right_context, bool)
        assert isinstance(previous_question, bool)
        assert isinstance(previous_question_number, bool)
        assert any(
            feature
            for feature in [
                span_text,
                left_context,
                right_context,
                previous_question,
                previous_question_number,
            ]
        ), "Must have at least one feature to vectorize, otherwise X will be empty."
        self.span_text = span_text
        self.left_context = left_context
        self.right_context = right_context
        self.previous_question = previous_question
        self.previous_question_number = previous_question_number
        self.is_fit = False

    def __get_state__(self):
        state = self.__dict__.copy()
        return state

    def __set_state__(self, state):
        self.__dict__.update(state)

    def fit(self, train: list[ExtendedXmlCodeSpan]) -> None:
        if self.span_text:
            text_vectorizer = TfidfVectorizer()
            text_vectorizer.fit([code_span.text for code_span in train])
            self.text_vectorizer = text_vectorizer
        if self.left_context:
            left_context_vectorizer = TfidfVectorizer()
            left_context_vectorizer.fit([code_span.left_context for code_span in train])
            self.left_context_vectorizer = left_context_vectorizer
        if self.right_context:
            right_context_vectorizer = TfidfVectorizer()
            right_context_vectorizer.fit(
                [code_span.right_context for code_span in train]
            )
            self.right_context_vectorizer = right_context_vectorizer
        if self.previous_question:
            previous_question_vectorizer = TfidfVectorizer()
            previous_question_vectorizer.fit(
                [code_span.previous_question for code_span in train]
            )
            self.previous_question_vectorizer = previous_question_vectorizer
        if self.previous_question_number:
            previous_question_number_vectorizer = OneHotEncoder()
            previous_question_number_vectorizer.fit(
                np.array([i for i in range(1, NUM_QUESTIONS + 1)]).reshape(-1, 1)
            )
            self.previous_question_number_vectorizer = (
                previous_question_number_vectorizer
            )

        self.is_fit = True

    def transform(self, code_spans: list[ExtendedXmlCodeSpan]) -> np.ndarray:
        if not self.is_fit:
            raise ValueError("Vectorizer must be fit before transform")
        X = None
        if self.span_text:
            X_text = self.text_vectorizer.transform(
                [code_span.text for code_span in code_spans]
            )
            X = hstack((X, X_text)) if X is not None else X_text
        if self.left_context:
            X_left_context = self.left_context_vectorizer.transform(
                [code_span.left_context for code_span in code_spans]
            )
            X = hstack((X, X_left_context)) if X is not None else X_left_context
        if self.right_context:
            X_right_context = self.right_context_vectorizer.transform(
                [code_span.right_context for code_span in code_spans]
            )
            X = hstack((X, X_right_context)) if X is not None else X_right_context
        if self.previous_question:
            X_previous_question = self.previous_question_vectorizer.transform(
                [code_span.previous_question for code_span in code_spans]
            )
            X = (
                hstack((X, X_previous_question))
                if X is not None
                else X_previous_question
            )
        if self.previous_question_number:
            X_previous_question_number = (
                self.previous_question_number_vectorizer.transform(
                    np.array(
                        [code_span.previous_question_number for code_span in code_spans]
                    ).reshape(-1, 1)
                )
            )
            X = (
                hstack((X, X_previous_question_number))
                if X is not None
                else X_previous_question_number
            )
        assert X is not None
        return X


class RetrieverVectorizer:
    def __init__(
        self,
        span_text: bool,
        left_context: bool,
        right_context: bool,
        previous_question: bool,
        tokenization_method: str,
        lowercase: bool,
    ) -> None:
        assert isinstance(span_text, bool)
        assert isinstance(left_context, bool)
        assert isinstance(right_context, bool)
        assert isinstance(previous_question, bool)
        assert isinstance(tokenization_method, str) and tokenization_method in [
            "space separated"
        ]
        assert isinstance(lowercase, bool)
        assert any(
            feature
            for feature in [
                span_text,
                left_context,
                right_context,
                previous_question,
            ]
        ), "Must have at least one feature to vectorize."
        self.span_text = span_text
        self.left_context = left_context
        self.right_context = right_context
        self.previous_question = previous_question
        self.tokenization_method = tokenization_method
        self.lowercase = lowercase
        self.is_fit = False

    def fit_transform(self, code_spans: list[ExtendedXmlCodeSpan]) -> list[list[str]]:
        transformed_code_spans: list[str] = []
        for code_span in code_spans:
            text = ""
            if self.previous_question:
                text += f"{code_span.previous_question} "
            if self.left_context:
                text += f" {code_span.left_context} "
            if self.span_text:
                text += f" {code_span.text} "
            if self.right_context:
                text += f" {code_span.right_context} "
            assert len(text) > 0
            if self.lowercase:
                text = text.lower()
            if self.tokenization_method == "space separated":
                transformed_code_spans.append(text.split(" "))
        return transformed_code_spans


class NeuralVectorizer:
    def __init__(
        self,
        span_text: bool,
        left_context: bool,
        right_context: bool,
        previous_question: bool,
        max_length: int,
        code_description: bool,
    ) -> None:
        assert isinstance(span_text, bool)
        assert isinstance(left_context, bool)
        assert isinstance(right_context, bool)
        assert isinstance(previous_question, bool)
        assert isinstance(max_length, int)
        assert any(
            feature
            for feature in [
                span_text,
                left_context,
                right_context,
                previous_question,
                code_description,
            ]
        ), "Must have at least one feature to vectorize."
        self.span_text = span_text
        self.left_context = left_context
        self.right_context = right_context
        self.previous_question = previous_question
        self.code_description = code_description
        self.max_length = max_length
        self.is_fit = False

    def fit_transform(
        self, code_spans: list[ExtendedXmlCodeSpan]
    ) -> dict[str, list[str]]:
        transformed_code_spans: dict[str, list[str]] = {}
        transformed_code_spans = {
            "span_text": [code_span.text for code_span in code_spans]
            if self.span_text
            else None,
            "left_context": [code_span.left_context for code_span in code_spans]
            if self.left_context
            else None,
            "right_context": [code_span.right_context for code_span in code_spans]
            if self.right_context
            else None,
            "previous_question": [
                code_span.previous_question for code_span in code_spans
            ]
            if self.previous_question
            else None,
        }
        assert any(v is not None for v in transformed_code_spans.values())
        return transformed_code_spans


class CustomLabeler:
    def __init__(self, label_type: Literal["binary", "multiclass"]):
        assert label_type in ["binary", "multiclass"]
        self.label_type = label_type
        self.labeler = None
        self.label2idx = None
        self.idx2label = None
        self.is_fit = False

    def __get_state__(self):
        state = self.__dict__.copy()
        return state

    def __set_state__(self, state):
        self.__dict__.update(state)

    def _set_dicts(self, code_spans: list[ExtendedXmlCodeSpan]) -> None:
        all_labels = []
        for code_span in code_spans:
            if isinstance(code_span.label, list):
                all_labels.extend(code_span.label)
            else:
                all_labels.append(code_span.label)
        unique_labels = []
        for label in all_labels:
            if label not in unique_labels:
                unique_labels.append(label)
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for idx, label in enumerate(unique_labels)}

    def _binarize_labels(self, labels: list[Union[str, list[str]]]) -> pd.DataFrame:
        assert self.label2idx is not None
        assert self.idx2label is not None
        if any(isinstance(label, str) for label in labels):
            raise NotImplementedError(
                "Binarize does not support singleton strings as labels"
                + "They must be in a list"
            )
        else:
            Y = np.zeros((len(labels), len(self.idx2label)))
            for i, label in enumerate(labels):
                indices = [self.label2idx[label] for label in label]
                Y[i, indices] = 1
        # Convert to pandas dataframe to conserve label names
        Y = pd.DataFrame(
            Y, columns=[self.idx2label[idx] for idx in range(len(self.idx2label))]
        )
        return Y

    def _multiclass_labels(self, labels: list[Union[str, list[str]]]) -> np.ndarray:
        assert self.label2idx is not None
        assert self.idx2label is not None
        if all(isinstance(label, list) for label in labels):
            raise NotImplementedError("Multiclass does not support lists as labels")
        else:
            y = np.array([self.label2idx[label] for label in labels])
        y = pd.DataFrame(y, columns=["label"])
        return y

    def fit(self, train: list[ExtendedXmlCodeSpan]) -> None:
        assert all(isinstance(code_span.label, str) for code_span in train) or all(
            isinstance(code_span.label, list) for code_span in train
        ), "Cannot mix single and multi-labels"
        self._set_dicts(train)
        if self.label_type == "binary":
            self.labeler = self._binarize_labels
        elif self.label_type == "multiclass":
            self.labeler = self._multiclass_labels
        self.is_fit = True

    def transform(self, code_spans: list[ExtendedXmlCodeSpan]) -> pd.DataFrame:
        assert all(isinstance(code_span.label, str) for code_span in code_spans) or all(
            isinstance(code_span.label, list) for code_span in code_spans
        ), "Cannot mix single and multi-labels"
        assert self.is_fit
        labels = [code_span.label for code_span in code_spans]
        return self.labeler(labels)
