from distutils.core import setup

# README file contents
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="QCS",
    version="1.0",
    packages=[
        "qcs",
    ],
    license="MIT",
    description="Qualitative Code Suggestion",
    long_description=long_description,
    author="Cesare Spinoso-Di Piano",
    author_email="cesare.spinoso-dipiano@mail.mcgill.ca",
    install_requires=[
        "numpy",
        "pandas",
        "tabulate",
        "matplotlib",
        "scikit-learn",
        "nltk",
        "spacy",
        "gensim",
        "torch",
        "rank_bm25",
        "bs4",
        "plotly",
        "omegaconf",
        "hydra-core",
        "jsonlines",
    ],
)
