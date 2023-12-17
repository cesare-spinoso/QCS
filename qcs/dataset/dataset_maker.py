import os
from pathlib import Path
import random
import re
from typing import Literal, Union

from bs4 import BeautifulSoup, element
from qcs.dataset.custom_dataclasses import XmlCodeSpansDataset

from qcs.utils.interview_xml import (
    XmlCodeSpan,
    filter_code_spans,
    read_xml_interview_directory,
)


class DatasetMaker:
    def __init__(
        self,
        xml_directory: Union[str, Path],
        coder: Literal[1, 2],
        interviews: list[int],
        left_context: int,
        right_context: int,
        previous_question: bool,
    ) -> None:
        """
        xml_directory : Path to the directory containing the xml files.
        coder : The coder to filter for.
        interviews : The interviews to include. The order in the list is the order which will be preserved
        for training.
        left_context : The number of tokens to include on the left-side of the code span.
        right_context : The number of tokens to include on the right-side of the code span.
        previous_question : Whether to include the previous question as a feature.
        """
        assert isinstance(xml_directory, (str, Path)) and os.path.exists(xml_directory)
        assert isinstance(coder, int) and coder in [1, 2]
        assert isinstance(interviews, list) and all(
            isinstance(i, int) for i in interviews
        )
        assert isinstance(left_context, int) and left_context >= 0
        assert isinstance(right_context, int) and right_context >= 0
        assert isinstance(previous_question, bool)
        self.xml_directory = (
            xml_directory if isinstance(xml_directory, Path) else Path(xml_directory)
        )
        self.coder = coder
        self.interviews = interviews
        self.left_context = left_context
        self.right_context = right_context
        self.previous_question = previous_question

    @staticmethod
    def get_left_context(tag: element.Tag, tokens: int) -> str:
        """Returns the surrounding context of the {tag} on the
        left-side with {tokens} number of tokens using space-
        separation."""
        assert (
            isinstance(tokens, int) and tokens >= 0
        ), f"{tokens} is not of type int or is negative."
        surrounding_text = ""
        if tag is None:
            return surrounding_text
        while True:
            if tag.previous_sibling is not None:
                tag = tag.previous_sibling
            else:
                tag = tag.parent
                tag = tag.previous_sibling
            if tag is None:
                break
            surrounding_text = (
                " ".join(tag.get_text().split()[-tokens:]) + " " + surrounding_text
            )
            tokens -= len(tag.get_text().split())
            if tokens <= 0:
                break
        return surrounding_text.strip()

    @staticmethod
    def get_right_context(tag: element.Tag, tokens: int) -> str:
        """Returns the surrounding context of the {tag} on the
        right-side with {tokens} number of tokens using space-
        separation."""
        assert (
            isinstance(tokens, int) and tokens >= 0
        ), f"{tokens} is not of type int or is negative."
        surrounding_text = ""
        if tag is None:
            return surrounding_text
        while True:
            if tag.next_sibling is not None:
                tag = tag.next_sibling
            else:
                tag = tag.parent
                tag = tag.next_sibling
            if tag is None:
                break
            surrounding_text = (
                surrounding_text + " " + " ".join(tag.get_text().split()[:tokens])
            )
            tokens -= len(tag.get_text().split())
            if tokens <= 0:
                break
        return surrounding_text.strip()

    @staticmethod
    def get_previous_question(tag: element.Tag) -> tuple[int, str]:
        """Returns the most recent question before the code span.
        Returns the question number and the text."""
        assert isinstance(tag, element.Tag), f"{tag} is not of type element.Tag."
        question_tag = tag
        while True:
            if question_tag.previous_sibling is None:
                question_tag = question_tag.parent
            else:
                question_tag = question_tag.previous_sibling
            if not isinstance(question_tag, element.Tag):
                continue
            else:
                if question_tag.name.startswith("question_"):
                    question_text = question_tag.get_text()
                    question_number = int(question_tag.name.split("_")[1])
                    return question_number, question_text.strip()
                else:
                    continue

    @staticmethod
    def get_xml_code_span(
        interview_id: int,
        interview_soup: BeautifulSoup,
        right_context: int = 50,
        left_context: int = 50,
        previous_question: bool = True,
    ) -> list[XmlCodeSpan]:
        """Returns the interview as a list of xml code spans."""
        code_spans = []
        for code_span in interview_soup.find_all(re.compile("code_.*")):
            # Dynamically walk through xml tree
            if "value" not in code_span.attrs or "coder" not in code_span.attrs:
                print(
                    f"Code span {code_span} does not have a code attribute, add it manually."
                )
                continue
            extended_xml_span = XmlCodeSpan(
                text=code_span.get_text(),
                code=code_span["value"],
                coder=int(code_span["coder"]),
                interview_id=interview_id,
            )
            if left_context > 0:
                extended_xml_span.left_context = DatasetMaker.get_left_context(
                    tag=code_span, tokens=left_context
                )
            if right_context > 0:
                extended_xml_span.right_context = DatasetMaker.get_right_context(
                    tag=code_span, tokens=right_context
                )
            if previous_question:
                (
                    extended_xml_span.previous_question_number,
                    extended_xml_span.previous_question,
                ) = DatasetMaker.get_previous_question(tag=code_span)
            else:
                extended_xml_span.previous_question_number = None
                extended_xml_span.previous_question = None
            code_spans.append(extended_xml_span)
        return code_spans

    TrainingSpans = list[XmlCodeSpan]
    TestSpans = list[XmlCodeSpan]

    @staticmethod
    def create_train_test_span(
        index_split: int,
        interview_ids: list[int],
        code_spans: list[XmlCodeSpan],
    ) -> tuple[TrainingSpans, TestSpans]:
        """Create train/test spans based on interview_id as split."""
        train_spans = []
        for interview_id in interview_ids[: index_split + 1]:
            train_spans.extend(code_spans[interview_id])
        test_spans = []
        for interview_id in interview_ids[index_split + 1 :]:
            test_spans.extend(code_spans[interview_id])
        return train_spans, test_spans

    @staticmethod
    def create_train_test_spans(
        interview_ids: list[int], code_spans: dict[int, list[XmlCodeSpan]]
    ) -> list[tuple[TrainingSpans, TestSpans]]:
        """Create train/test spans based on interview_id as split."""
        # Return train/test spans
        train_test_spans = []
        for index_split in range(len(interview_ids[:-1])):
            train_test_spans.append(
                DatasetMaker.create_train_test_span(
                    index_split, interview_ids, code_spans
                )
            )
        return train_test_spans

    @staticmethod
    def split_code_spans(
        code_spans: list[XmlCodeSpan], ratio=0.5, shuffle=True, seed=42
    ) -> tuple[list[XmlCodeSpan], list[XmlCodeSpan]]:
        """Split code spans into 2 lists based on ratio.
        Returns list with length (ratio * len(code_spans), (1 - ratio) * len(code_spans)).
        """
        assert all(
            isinstance(elt, (XmlCodeSpan)) for elt in code_spans
        ), f"{code_spans} contains elements that are not of type XmlCodeSpan or XmlCodeSpan."
        assert (
            isinstance(ratio, float) and 0 <= ratio <= 1
        ), f"{ratio} is not of type float or is not between 0 and 1."
        assert isinstance(shuffle, bool), f"{shuffle} is not of type bool."
        indices = list(range(len(code_spans)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        if ratio == 0:
            return [], [code_spans[index] for index in indices]
        elif ratio == 1:
            return [code_spans[index] for index in indices], []
        ratio_point = int(len(indices) * ratio)
        code_span_1 = [code_spans[index] for index in indices[:ratio_point]]
        code_span_2 = [code_spans[index] for index in indices[ratio_point:]]
        return code_span_1, code_span_2

    @staticmethod
    def split_test_spans(
        test_spans: list[TestSpans],
        ratio=0.5,
        shuffle=True,
    ) -> tuple[list[TestSpans], list[TestSpans]]:
        """Split test spans into 2 test spans.
        Returns test spans with length (ratio * len(test_spans), (1 - ratio) * len(test_spans)).
        """
        test_spans_1 = []
        test_spans_2 = []
        for test_span in test_spans:
            test_span_1, test_span_2 = DatasetMaker.split_code_spans(
                test_span, ratio, shuffle
            )
            test_spans_1.append(test_span_1)
            test_spans_2.append(test_span_2)
        return test_spans_1, test_spans_2


class QCSDatasetMaker(DatasetMaker):
    """QCSDatasetMaker for the QCS paper. Splits at each interview,
    reserving the first K for training and the remaining N - K
    for testing. The validation set is created by taking a chunk
    of the training set. Only do this if request a validation set."""

    def __init__(
        self,
        xml_directory: Union[str, Path],
        coder: Literal[1, 2],
        interviews: list[int],
        left_context: int,
        right_context: int,
        previous_question: bool,
        train_val_split: float,
    ) -> None:
        """train_val_split = 1 => no validation set, train_val_split = 0.9
        => 10% of the training set reserved for validation."""
        super().__init__(
            xml_directory=xml_directory,
            coder=coder,
            interviews=interviews,
            left_context=left_context,
            right_context=right_context,
            previous_question=previous_question,
        )
        self.train_val_split = train_val_split

    def make_dataset(self) -> XmlCodeSpansDataset:
        # Get the transcripts xml
        xml_interviews = read_xml_interview_directory(self.xml_directory)
        assert set(xml_interviews.keys()) == set(self.interviews)

        # Add custom classification preprocessing to xml
        code_spans_dict = {
            interview_id: DatasetMaker.get_xml_code_span(
                interview_id,
                interview_soup,
                self.right_context,
                self.left_context,
                self.previous_question,
            )
            for interview_id, interview_soup in xml_interviews.items()
        }

        # Filter for the coder
        code_spans_dict = {
            interview_id: filter_code_spans(
                code_spans, filter_key="coder", filter_value=self.coder
            )
            for interview_id, code_spans in code_spans_dict.items()
        }

        # Create train, val, test based on self.interviews order
        train_test_spans = DatasetMaker.create_train_test_spans(
            interview_ids=self.interviews, code_spans=code_spans_dict
        )
        train_spans = [spans[0] for spans in train_test_spans]
        test_spans = [spans[1] for spans in train_test_spans]
        train_spans, val_spans = DatasetMaker.split_test_spans(
            train_spans, ratio=self.train_val_split, shuffle=True
        )
        return XmlCodeSpansDataset(
            interview_ids=self.interviews,
            train=train_spans,
            val=val_spans,
            test=test_spans,
        )
