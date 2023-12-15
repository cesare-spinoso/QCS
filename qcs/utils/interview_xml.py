import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

from bs4 import BeautifulSoup
from qcs.utils.interview import INTERVIEW_FILE_NAME_TO_ID_DICT, INTERVIEW_IDS


@dataclass
class XmlCodeSpan:
    """Represents a code span in an interview along"""

    text: str
    code: str
    coder: int
    interview_id: int
    """Label is of type list since many codes can be assigned to a single span."""
    label: list[str] = None
    left_context: str = None
    right_context: str = None
    previous_question: str = None
    previous_question_number: int = None

    def __post_init__(self):
        """Set the left and right context."""
        assert isinstance(self.text, str), f"{self.text} is not of type str."
        assert isinstance(self.code, str), f"{self.code} is not of type str."
        assert isinstance(self.coder, int), f"{self.coder} is not of type int."
        assert (
            isinstance(self.interview_id, int) and self.interview_id in INTERVIEW_IDS
        ), f"{self.interview_id} is not of type int."


def read_xml_interview_directory(
    directory_path: Path, parser="html.parser"
) -> dict[int, BeautifulSoup]:
    """Reads all the interviews in the directory and returns a dictionary of
    the interview id to the interview text"""
    assert isinstance(directory_path, Path), f"{directory_path} is not of type Path."
    interview_dict: dict[int, BeautifulSoup] = {}
    for interview_path in directory_path.glob("*.xml"):
        interview_id = INTERVIEW_FILE_NAME_TO_ID_DICT[interview_path.stem]
        with open(interview_path, "r") as f:
            soup = BeautifulSoup(f, parser)
            interview_dict[interview_id] = soup
    return interview_dict


def get_code_spans_from_interview_soup(
    interview_id: int, interview_soup: BeautifulSoup
) -> list[XmlCodeSpan]:
    """Returns the text from the interview soup as a list of Xml spans."""
    xml_code_span_list = []
    for code_span in interview_soup.find_all(re.compile("code_.*")):
        if "value" not in code_span.attrs or "coder" not in code_span.attrs:
            print(
                f"Code span {code_span} does not have a code attribute, add it manually."
            )
            continue
        xml_code_span_list.append(
            XmlCodeSpan(
                text=code_span.get_text(),
                code=code_span["value"],
                coder=int(code_span["coder"]),
                interview_id=interview_id,
            )
        )
    return xml_code_span_list


def filter_code_spans(
    list_code_spans: list[XmlCodeSpan],
    filter_key: Literal["code", "coder", "source"],
    filter_value: Union[str, int],
) -> list[XmlCodeSpan]:
    """Only keep code spans that match the filter key and value."""
    return [
        code_span
        for code_span in list_code_spans
        if getattr(code_span, filter_key) == filter_value
    ]
