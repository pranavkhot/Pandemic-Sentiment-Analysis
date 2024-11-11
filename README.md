# 📊 Sentiment Analysis on COVID-19 Tweets Using Machine Learning

This project analyzes public sentiment on Twitter regarding COVID-19 using machine learning models. It classifies tweets into **Positive**, **Negative**, or **Neutral** sentiments, employing models like **BERT** . The project includes data loading, preprocessing, exploratory data analysis (EDA), and sentiment prediction through a graphical interface.

## 📜 Table of Contents
- [📁 Project Structure](#project-structure)
- [🚀 Usage](#usage)
- [🧹 Data Preprocessing](#data-preprocessing)
- [📊 Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [🤖 Model Training](#model-training)
- [🖥️ Real-Time Sentiment Analysis UI](#real-time-sentiment-analysis-ui)
- [🛠️ Technologies Used](#-technologies-used)




## Prerequisites
- Python 3.9 or higher
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`, `textblob`, `torch`, `transformers`, `sklearn`, `tabulate`, `tkinter`



## 📁 Project Structure

* DATA_LOADING.ipynb: Loads and combines tweet datasets.
* PREPROCESSING.ipynb: Cleans and prepares tweets for analysis.
* EDA.ipynb: Conducts exploratory data analysis, including word clouds and tweet distribution.
* BERT.ipynb: Implements BERT-based sentiment analysis.
* UI.ipynb: Provides a GUI for real-time sentiment analysis on new tweets.

## 🚀 Usage

1. Load and Combine Datasets
Run the DATA_LOADING.ipynb script to load and combine COVID-19 tweet data from multiple sources into a single dataset.

2. Preprocess Data
Execute PREPROCESSING.ipynb to clean and filter data. This step removes unnecessary information, filters for English tweets from India, and applies text preprocessing like removing mentions, special characters, URLs, and hashtags.

3. Perform Exploratory Data Analysis (EDA)
The EDA.ipynb script explores the dataset, including:

Top tweets by likes and retweets
Distribution of tweets by hour
Word cloud visualizations for common words
Bigram and trigram analysis

4. Train the Sentiment Analysis Model
Run BERT.ipynb to train the sentiment analysis model:

Tokenizes tweets using BERT tokenizer
Converts text to TF-IDF vectors
Splits data into training and test sets
Fine-tunes BERT for COVID-19 sentiment classification

5. Real-Time Sentiment Analysis UI
Run UI.ipynb to start a GUI where users can input a tweet to receive a sentiment prediction (Positive, Neutral, or Negative) with a bar chart visualization of prediction probabilities.
## 🧹 Data Preprocessing

The `PREPROCESSING.ipynb` script performs essential preprocessing steps to prepare the data for analysis:

- **Filtering**
  - Retains only English-language tweets originating from India for targeted analysis.

- **Text Cleaning**
  - Removes mentions (`@username`), hashtags (`#hashtag`), URLs, and special characters to normalize the text.

- **Handling Missing Values**
  - Identifies and addresses missing values to ensure data quality and reliability.

- **Preprocessing for Model Input**
  - Converts timestamps to a decimal format for consistency.
  - Normalizes the text data for better model compatibility.

## 📊 Exploratory Data Analysis (EDA)

The `EDA.ipynb` script explores and visualizes the data to reveal key patterns and trends:

- **Top Tweets Analysis**
  - Extracts the top tweets by **retweet count** and **like count** to identify the most popular content.

- **Tweet Distribution by Time**
  - Visualizes tweet frequency over a 24-hour period to analyze user activity trends.

- **Word Cloud Visualization**
  - Creates a word cloud to highlight the most frequently mentioned words in the tweets.

- **N-grams Analysis**
  - Identifies common **bigrams** (two-word sequences) and **trigrams** (three-word sequences) for deeper insights into prevalent phrases.

- **Tweet Length and Sentiment Distribution**
  - Examines the distribution of tweet lengths and sentiments to understand the data's overall structure.

## 🤖 Model Training

The `BERT.ipynb` file utilizes the BERT model to classify tweet sentiments. The primary steps include:

- **Tokenization**
  - Transforms tweets into BERT-compatible tokens, enabling the model to understand text context.

- **Training**
  - Fine-tunes BERT on the COVID-19 tweet data to improve sentiment prediction accuracy.

- **Evaluation**
  - Generates a detailed classification report, including **precision**, **recall**, and **F1-score**, to assess model performance.

## 🖥️ Real-Time Sentiment Analysis UI

The `UI.ipynb` script provides a user-friendly interface to predict tweet sentiments in real-time:

- **Tkinter-based GUI**
  - Offers a simple text input for users to enter a tweet or sentence.

- **Sentiment Prediction**
  - Displays sentiment predictions (Positive, Neutral, or Negative) with a visual bar chart of prediction probabilities for easy interpretation.

## 🛠️ Technologies Used

* Python: Core programming language for scripting and automation.
* Pandas: Data manipulation and analysis, handling CSVs, and dataset operations.
* Seaborn & Matplotlib: Data visualization for EDA and analysis insights.
* Sklearn: Support for machine learning model training and evaluation.
* BERT and Transformers: NLP using pre-trained language models for contextual understanding.
* Tkinter: User Interface for inputting tweets and displaying real-time sentiment predictions.
