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

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## ScreenShots from web app

![web app user input](https://github.com/IbrahimYahyaHakami/disaster_response_pipeline/blob/master/img/web%20app%20user%20input.PNG)

This show web interface where user can input message and click classify for the model to output if the message reletive to any kind of emergency.

![Example of an input with results](https://github.com/IbrahimYahyaHakami/disaster_response_pipeline/blob/master/img/example%20of%20an%20input%20with%20results.PNG)

Here is an example of an input with results, if you keep scroling down you will see al related emergncy types.

![The count of different emergency type](https://github.com/IbrahimYahyaHakami/disaster_response_pipeline/blob/master/img/The%20count%20of%20different%20emergency%20type.PNG)
![Distribution of Message Genres](https://github.com/IbrahimYahyaHakami/disaster_response_pipeline/blob/master/img/Distribution%20of%20Message%20Genres.PNG)

Some visualization and overview of the training data set.

## Authors

* **Ibrahim Hakami** - *Initial work* - [IbrahimHakami](https://github.com/IbrahimYahyaHakami)

## Acknowledgments

* MISK academy
* Udacity
* special thanks to our instructors for all the help they provided.