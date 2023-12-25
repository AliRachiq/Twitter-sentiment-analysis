This repository hosts Jupyter notebooks for conducting sentiment analysis on Twitter data using cutting-edge natural language processing models like BERT, LSTM. The notebooks encompass the complete workflow, encompassing data preprocessing, exploration, model training, evaluation, and result visualization.

## Summary:

- **Objective:** Conduct sentiment analysis on Twitter data using BERT, LSTM, and GPT models.
- **Workflow:** Covers data preprocessing, exploratory data analysis, model implementation, training, evaluation, and visualizing results.
- **Prerequisites:** Python, Jupyter Notebook, TensorFlow, Hugging Face Transformers, Plotly, NLTK, Pandas, NumPy.

### Twitter Sentiment Analysis with LSTM

This notebook focuses on sentiment analysis using the LSTM architecture to classify tweets into positive or negative sentiment categories.

- **Dataset:** Sourced from Kaggle: [training.1600000.processed.noemoticon.csv](https://www.kaggle.com/datasets/ferno2/training1600000processednoemoticoncsv/data)
- **Steps:**
    1. **Data Preparation:** Clean and preprocess the dataset for sentiment analysis.
    2. **Text Processing:** Tokenization, sequence padding, and other text processing steps.
    3. **Model Implementation:** Utilizing LSTM for sentiment analysis.
    4. **Training and Evaluation:** Training the model and evaluating its performance on a test set.

### Twitter Sentiment Analysis with BERT

This notebook explores sentiment analysis using the BERT architecture for classifying tweets into positive or negative sentiment categories.

- **Dataset:** Sourced from Kaggle: [training.1600000.processed.noemoticon.csv](https://www.kaggle.com/datasets/ferno2/training1600000processednoemoticoncsv/data)
- **Steps:**
    1. **Data Preparation:** Clean and prepare the dataset for sentiment analysis.
    2. **Text Processing:** Tokenization and sequence padding of text data.
    3. **BERT Model:** Utilizing the powerful BERT model for sentiment classification.
    4. **Training and Evaluation:** Training the model and evaluating its performance on a test set.

### Results:

- **LSTM Model Performance:**
    - Precision: 0.74 (Class 0), 0.69 (Class 1)
    - Recall: 0.67 (Class 0), 0.77 (Class 1)
    - F1-score: 0.70 (Class 0), 0.73 (Class 1)
    - Accuracy: 0.72

- **BERT Model Performance:**
    - Precision: 0.75 (Class 0), 0.77 (Class 1)
    - Recall: 0.79 (Class 0), 0.73 (Class 1)
    - F1-score: 0.77 (Class 0), 0.75 (Class 1)
    - Accuracy: 0.76

To run these notebooks, click on the "Open in Colab" badge provided with each section and follow the instructions within the notebook cells for execution.: 
**BERT** : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11sfKeJ_hFtpw1DnJi4YgYjsU9eOFmoL-?usp=sharing)
**LSTM** : [Open In Colab](https://colab.research.google.com/drive/1V5a738HNZQjy120htn1p8qYoRse-OdOL?usp=sharing).


![image](https://github.com/AliRachiq/Twitter-sentiment-analysis/assets/85627949/e262fb59-cfa6-4547-970e-3a703418a48d)
