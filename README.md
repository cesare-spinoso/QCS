# QCS
Repository for [Qualitative Code Suggestion: A Human-Centric Approach To Qualitative Coding](https://aclanthology.org/2023.findings-emnlp.993/) by [Cesare Spinoso-Di Piano](https://cesare-spinoso.github.io/), [Samira Abbasgholizadeh-Rahimi](https://rahimislab.ca/) and [Jackie Chi Kit Cheung](https://www.cs.mcgill.ca/~jcheung/index.html).

If you use our dataset, code or findings, please cite us:
```
@inproceedings{spinoso-di-piano-etal-2023-qualitative,
    title = "Qualitative Code Suggestion: A Human-Centric Approach to Qualitative Coding",
    author = "Spinoso-Di Piano, Cesare and Rahimi, Samira and Cheung, Jackie",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    year = "2023",
}
```

For any questions regarding the paper or the dataset, please open an issue on GitHub or email the corresponding author (email address in paper).

# Installation and environment setup
To install this repo, first clone it and then execute the following commands (assumes Anaconda is installed on your machine):
```
conda create -n my_env python=3.9
```
Activate your environment with `conda activate my_env`, and then execute the following to complete the installation of the repository and its packages:
```
make setup
```

# Dataset
The `CVDQuoding` dataset is available upon request. The current repository assumes that the 15 `xml` files we will send you will be placed in the `data/raw_xml` directory. To reproduce our results, you must convert the raw XML files to a `.jsonl` file format. To do so, please run our dataset maker script found in `qcs/dataset`. More instruction available there.

