# Machine Learning Complements Project

[Course page](https://sigarra.up.pt/feup/en/UCURR_GERAL.FICHA_UC_VIEW?pv_ocorrencia_id=518813)

### First project - Recommender Systems + Social Network Analysis
RS + SNA branch

Community-Based Recommendations: used SNA to identify user communities and applied an RS to each community

See [presentation](https://github.com/DiogoSilva11/CAC/blob/RS_and_SNA/docs/presentation.pdf)

### Second project - Natural Language Processing + Time series
Main branch

Sentiment Analysis Over Time: implemented NLP to predict sentiment in reviews and tracked its evolution over time.

See [presentation](https://github.com/DiogoSilva11/CAC/blob/main/docs/CAC_groupC_proj2.pdf)

----
**Project developed by [Group C]:**

| Name | Student Number |
| :---- | --------------: |
| Bruna Marques | 202007191 |
| Diogo Silva | 202004288 |
| Lia Vieira | 202005042 |
| Pedro Fonseca | 202008307 |

## Dataset

[Yelp Dataset](https://www.yelp.com/dataset/download)

[Dataset Documentation](https://www.yelp.com/dataset/documentation/main)

## 2n project - Natural Language Processing + Time series

### Project Structure

#### Folders

- **data_exploration** - general data exploration and understanding processes performed on each dataset file

- **filtered_cities** - CSV files storing only the relevant data for the project (from the selected city or cities), to avoid loading the whole dataset each time

- **nlp** - notebooks dedicated to exploring Natural Language Processing
  - nlp/**data** - folder where the sentiment for each review is stored (review_sentiment.txt) as well as the corpus of each review (corpus.txt)
  - nlp/**embeddings** - folder where the reviews model and word vectors are stored for future use without retraining
  - nlp/**features** - folder where sparse and dense vectors are stored after transformations

- **time_series** - Notebooks dedicated to exploring Time Series
  - time_series/**data** - folder where we save the positive/neutral/negatives reviews per month, as well as, the total/Restaurant/Nightlife/Breakfast&Brunch mean sentiment per month
  - time_series/**models** - folder where the ARIMA models are stored
  - time_series/**features** - folder that saves the same information as data but in each dataset there are more features, such as, box_cox, trend, seasonal, residual, etc.

#### Files

- **filter_dataset.ipynb** - to store data from city or cities of interest in the filtered_cities folder

- **utils.py** - few helper methods for data loading

- nlp/**data.ipynb** - loads, processes, and analyzes review data to determine sentiment and visualize word usage patterns

- nlp/**embeddings.ipynb** - loads and preprocesses review sentiment data, trains a Word2Vec model to generate word embeddings, and visualizes the embeddings using t-SNE for semantic analysis

- nlp/**feature_engineering.ipynb** - performs text preprocessing, applies various feature engineering techniques including word embeddings and topic modeling, and visualizes the results

- nlp/**modelling.ipynb** - performs sentiment analysis and classification tasks, including feature extraction, model selection, hyperparameter tuning, evaluation, and visualization

- time_series/**data.ipynb** - analyzes Yelp data for St. Louis, including sentiment analysis of reviews, time series plots for user activity and check-ins, and exploration of top business categories

- time_series/**patterns.ipynb** - conducts comprehensive time series analysis on sentiment data, including loading, plotting lag and autocorrelation, testing stationarity, and visualizing rolling statistics

- time_series/**feature_engineering.ipynb** - conducts feature engineering, including Box-Cox transformations and additive decomposition, on time series data for sentiment analysis of reviews

- time_series/**arima.ipynb** - performs ARIMA modeling on various sentiment time series data, including negative, neutral, and positive reviews, along with sentiment analysis for different categories like restaurants, nightlife, and breakfast & brunch

- time_series/**modelling.ipynb** - performs time series forecasting using various models, including ARIMA and exponential smoothing, for different sentiment categories like negative, neutral, and positive reviews, with evaluations and predictions presented
