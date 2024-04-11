# Machine Learning Complements Project

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

## Recommender Systems + Social Network Analysis

### Project Structure

#### Folders

- **communities** - communities from each developed graph stored in JSON format

- **data_exploration** - general data exploration and understanding processes performed on each dataset file

- **filtered_cities** - CSV files storing only the relevant data for the project (from the selected city or cities), to avoid loading the whole dataset each time

- **nodes_and_edges** - nodes and edges from each type of connection computed between users stored in CSV format (to later create graphs)

#### Files

- **bulk_rs_run.py** - script with the objective of running several recommendation models across several formed communities (from different graphs), calculating the essential metrics for each case

- **connections.ipynb** - data preparation and creation of different types of connections between users, forming the corresponding nodes and edges to be used in graphs

- **filter_dataset.ipynb** - to store data from city or cities of interest in the filtered_cities folder

- **recsys.ipynb** - development of the recommender systems

- **results.csv** - results from the bulk_rs_run script stored

- **sna.ipynb** - graph and community formation; social network analysis

- **utils.py** - few helper methods for data loading