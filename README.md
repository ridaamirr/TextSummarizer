# Text Summarizer

This project implements a text summarizer using machine learning techniques. The summarizer takes articles and generates their highlights.

## Dataset Description and Features

### Dataset Identification
The dataset used for this project is sourced from Kaggle:
[Newspaper Text Summarization (CNN/DailyMail)](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail?resource=download)

The dataset consists of articles and their corresponding highlights, which are used as the source text and target summaries, respectively. An ID is included for each entry but is not used in the summarization process.

### Features
- **Article**: The main text of the news article.
- **Highlights**: The summarized text or key points of the article.
- **ID**: A unique identifier for each entry (not used in the summarization process).

## Preprocessing with Pandas

### Import Libraries
The necessary libraries for this project are imported and used throughout the notebook. This includes libraries such as `pandas` for data manipulation and `kaggle` for dataset access.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


Here's the README.md file with the additional sections included in Markdown format:

markdown
Copy code
# Text Summarizer

This project implements a text summarizer using machine learning techniques. The summarizer takes articles and generates their highlights.

## Group Members
- Rida Amir (21L-5184)
- Haisem Naeem (21L-7528)
- Hamza Amer (21L-5368)

## Dataset Description and Features

### Dataset Identification
The dataset used for this project is sourced from Kaggle:
[Newspaper Text Summarization (CNN/DailyMail)](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail?resource=download)

The dataset consists of articles and their corresponding highlights, which are used as the source text and target summaries, respectively. An ID is included for each entry but is not used in the summarization process.

### Features
- **Article**: The main text of the news article.
- **Highlights**: The summarized text or key points of the article.
- **ID**: A unique identifier for each entry (not used in the summarization process).

## Preprocessing Steps
- **Loading the Dataset**: The dataset is loaded into a Pandas DataFrame.
- **Cleaning the Data**: Unnecessary columns, such as the ID, are removed. Any missing values are handled appropriately.
- **Tokenization**: The text is tokenized into individual words or subwords, depending on the summarization model requirements.
- **Padding/Truncating**: Ensuring uniform length for model input by padding shorter sequences and truncating longer ones.

## Model Training
The summarizer model is trained using the preprocessed dataset. Various machine learning models and techniques can be applied, such as:
- Transformer-based models (e.g., BERT)

## Results
The trained model's performance on the test set is presented, showing the effectiveness of the summarization. Examples of generated summaries versus reference summaries are provided for qualitative analysis.

