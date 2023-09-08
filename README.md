# ServiceNow Generative AI Challenge - Senior Data Scientist (NLP)

Please develop solutions to the following two problems in Jupyter Notebook, and deliver a 1-hour technical presentation (incl. Q&A) to the interview committee.

Tips:

1. You are expected to develop your presentation in a PowerPoint format. However, you may be occasionally asked to show us your code in Jupyter Notebook when you and we deep dive into the technical details together.
2. The interview committee members are all technical, but not everyone specializes in NLP and GenAI. Please tailor your presentation to fit both NLP and non-NLP data scientists.
3. Be ready to present and/or answer questions about “what” and “why”:
a. What: the steps you take to tackle the problem.
b. Why: why you do things in such a way. For example, why you select those samples; why you choose to use that model; why this metric is important for this problem.

## Tasks

1. Paraphrase Detection
2. Content Summarization

### Problem 1: Paraphrase Detection

You are asked to build a model to classify if two sentences are paraphrases of each other. “1” = yes, “0” = no. You are expected to establish an end-to-end process, including pre-processing, modeling, validation, etc.

1. Here is the dataset you can use: https://huggingface.co/datasets/quora. Rather than using the entire dataset of ~404K datapoints, you should select (by yourself) a subset of 50K datapoints to work on.

2. Please explain the data pre-processing steps you take and why.

3. Please explain the model you use and why. Report all hyperparameters used if any.

4. Please report all relevant evaluation metrics.

### Problem 2: Content Summarization

You are asked to extract and summarize useful information from a 10-K report. Use the 10-K report of NextGen Healthcare as an example: https://www.sec.gov/Archives/edgar/data/708818/000095017023023499/nxgn-20230331.htm

1. Extract the “Business” section and the “Risk Factors” section from the report, using the following pipeline: https://github.com/Unstructured-IO/pipeline-sec-filings

2. Summarize each of the two sections using the Flan-T5 Large model (https://huggingface.co/google/flan-t5-large). If the Large version does not fit your machine or takes time to load or generate tokens, feel free to grab a smaller version of Flan-T5.

3. Please report all hyperparameters used if any.

### Requirements

**Critical Requirements:**

1. Python >=3.8
2. NVidia GPU - V100 or better
3. Ubuntu >= 18.04 with CuDNN >= 11.1

**Primary Requirements:**

1. numpy
2. tqdm
3. datasets
4. transformers
5. pandas
6. torch

#### References

- https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
- https://www.kaggle.com/competitions/quora-question-pairs/overview
- https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca
- https://gombru.github.io/2019/04/03/ranking_loss/
- https://huggingface.co/spaces/mteb/leaderboard
- https://towardsdatascience.com/the-quora-question-pair-similarity-problem-3598477af172
