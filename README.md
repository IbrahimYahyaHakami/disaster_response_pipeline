# Disaster Response Pipeline Project

Building web interface which can take on an emergency message as input and classify to appropriate type for the emergency response team to plan to counter it accordingly.
## Getting Started

We will be using jupyter notebook in this project so we will need to Install anaconda 3 which will automatically install jupyter notebook , which is the main platform we will be using for this project.

### Prerequisites

Install anaconda 3 which will automatically install jupyter notebook with python 3  , which is the main platform we will be using for this project.

### Installing

You will want to install anaconda 3 latest version depend on your operating system through this [link](https://docs.anaconda.com/anaconda/install/hashes/win-3-64/).

## Built With

* [Jupyter notebook](https://docs.anaconda.com/anaconda/install/hashes/win-3-64/) - platform used
* [stackoverflow](https://insights.stackoverflow.com/survey) - data set source

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Authors

* **Ibrahim Hakami** - *Initial work* - [IbrahimHakami](https://github.com/IbrahimYahyaHakami)

## Acknowledgments

* MISK academy
* Udacity
* special thanks to our instructors for all the help they provided.