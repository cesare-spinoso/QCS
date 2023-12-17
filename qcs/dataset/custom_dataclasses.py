import copy
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qcs.utils.interview_xml import XmlCodeSpan
from qcs.utils.interview import INTERVIEW_IDS


@dataclass
class XmlCodeSpanInterviewSplit:
    interview_id: int
    """All transcripts up to interview_id are part of the training set."""
    train: list[XmlCodeSpan]
    val: list[XmlCodeSpan]
    test: list[XmlCodeSpan]
    is_prepared: bool = False

    def __post_init__(self) -> None:
        assert self.interview_id in INTERVIEW_IDS
        assert all(
            code_span.interview_id <= self.interview_id for code_span in self.train + self.val
        )
        assert all(code_span.interview_id > self.interview_id for code_span in self.test)

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
        code_span: XmlCodeSpan,
        code_spans: list[XmlCodeSpan],
        ignore_code: bool = False,
        ignore_label: bool = False,
    ) -> int:
        for idx, code_span_ in enumerate(code_spans):
            if code_span.__eq__(
                code_span_, ignore_code=ignore_code, ignore_label=ignore_label
            ):
                return idx
        return -1

    def _merge_code_spans(self, code_spans: list[XmlCodeSpan]) -> list[XmlCodeSpan]:
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

    def prepare(
        self,
        span_text: bool,
        left_context: bool,
        right_context: bool,
        previous_question: bool,
        previous_question_number: bool,
    ) -> None:
        assert not self.is_prepared, "Dataset is already prepared"
        # Convert codes to labels
        # NOTE: This takes care of assigning the "novel" code to spans
        # in train that only appear once and spans in val and test
        # which do not appear in train. This is done **before** merging
        # so that a code span may have several labels including "novel".
        self._convert_train_codes_to_labels()
        self._convert_test_codes_to_labels()
        # Merge duplicate data points
        self._merge_duplicate_data_points()
        # Create labeler
        labeler = CustomLabeler()
        labeler.fit(self.train)
        self.labeler = labeler
        # Create neural vectorize
        vectorizer = FeaturesVectorizer(
            span_text=span_text,
            left_context=left_context,
            right_context=right_context,
            previous_question=previous_question,
            previous_question_number=previous_question_number,
        )
        self.vectorizer = vectorizer
        self.is_prepared = True

    def _get_data(
        self, code_spans: list[XmlCodeSpan]
    ) -> tuple[dict[str, list], pd.DataFrame]:
        assert self.is_prepared, "Dataset must be prepared before use"
        X = self.vectorizer.fit_transform(code_spans)
        Y = self.labeler.transform(code_spans)
        assert all(len(v) == len(Y) for v in X.values())
        return X, Y

    def get_training_data(self) -> tuple[dict[str, list], pd.DataFrame]:
        assert self.is_prepared
        return self._get_data(self.train)

    def get_validation_data(self) -> tuple[dict[str, list], pd.DataFrame]:
        assert self.is_prepared
        # Allow empty validation set
        if all(isinstance(elt, list) and len(elt) == 0 for elt in self.val):
            return None, None
        return self._get_data(self.val)

    def get_test_data(self) -> tuple[np.ndarray, pd.DataFrame]:
        assert self.is_prepared
        return self._get_data(self.test)


class FeaturesVectorizer:
    def __init__(
        self,
        span_text: bool,
        left_context: bool,
        right_context: bool,
        previous_question: bool,
        previous_question_number: bool,
    ) -> None:
        assert isinstance(span_text, bool)
        assert isinstance(left_context, bool)
        assert isinstance(right_context, bool)
        assert isinstance(previous_question, bool)
        assert any(
            feature
            for feature in [
                span_text,
                left_context,
                right_context,
                previous_question,
                previous_question_number,
            ]
        ), "Must have at least one feature to vectorize."
        self.span_text = span_text
        self.left_context = left_context
        self.right_context = right_context
        self.previous_question = previous_question
        self.previous_question_number = previous_question_number
        self.is_fit = False

    def fit_transform(self, code_spans: list[XmlCodeSpan]) -> dict[str, list]:
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
            "previous_question_number": [
                code_span.previous_question_number for code_span in code_spans
            ]
            if self.previous_question_number
            else None,
        }
        assert any(v is not None for v in transformed_code_spans.values())
        return transformed_code_spans


class CustomLabeler:
    def __init__(self):
        self.labeler = None
        self.label2idx = None
        self.idx2label = None
        self.is_fit = False

    def __get_state__(self):
        state = self.__dict__.copy()
        return state

    def __set_state__(self, state):
        self.__dict__.update(state)

    def _set_dicts(self, code_spans: list[XmlCodeSpan]) -> None:
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

    def _binarize_labels(self, labels: list[list[str]]) -> pd.DataFrame:
        assert self.label2idx is not None
        assert self.idx2label is not None
        assert all(isinstance(label, list) for label in labels)
        Y = np.zeros((len(labels), len(self.idx2label)))
        for i, label in enumerate(labels):
            indices = [self.label2idx[label] for label in label]
            Y[i, indices] = 1
        # Convert to pandas dataframe to conserve label names
        Y = pd.DataFrame(
            Y, columns=[self.idx2label[idx] for idx in range(len(self.idx2label))]
        )
        return Y

    def fit(self, train: list[XmlCodeSpan]) -> None:
        assert all(isinstance(code_span.label, str) for code_span in train) or all(
            isinstance(code_span.label, list) for code_span in train
        ), "Cannot mix single and multi-labels"
        assert not self.is_fit
        self._set_dicts(train)
        self.labeler = self._binarize_labels
        self.is_fit = True

    def transform(self, code_spans: list[XmlCodeSpan]) -> pd.DataFrame:
        assert all(isinstance(code_span.label, str) for code_span in code_spans) or all(
            isinstance(code_span.label, list) for code_span in code_spans
        ), "Cannot mix single and multi-labels"
        assert self.is_fit
        labels = [code_span.label for code_span in code_spans]
        return self.labeler(labels)


@dataclass
class XmlCodeSpansDataset:
    """Contains ALL of splits for every corresponding interview id."""

    interview_ids: list[int]
    train: list[list[XmlCodeSpan]]
    val: list[list[XmlCodeSpan]]
    test: list[list[XmlCodeSpan]]

    def __post_init__(self) -> None:
        assert (
            (len(self.interview_ids) - 1)
            == len(self.train)
            == len(self.val)
            == len(self.test)
        ), print(
            f"{len(self.interview_ids)} != {len(self.train)} != {len(self.val)} != {len(self.test)}"
        )

    def enumerate(self):
        """Enumerates each xml code span split by interview id."""
        for i, interview_id in enumerate(self.interview_ids[:-1]):
            yield XmlCodeSpanInterviewSplit(
                interview_id=interview_id,
                train=self.train[i],
                val=self.val[i],
                test=self.test[i],
            )
