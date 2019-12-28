import sys
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, precision_score,recall_score,f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pickle
from sklearn.model_selection import GridSearchCV



def load_data(database_filepath):
    '''
    input data base path
    
    output features, target and data frame
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filepath[:-3], engine )
    X = df.message
    Y = df.iloc[:,4:]
    return X, Y, df


def tokenize(text):
    '''
    input full messages 
    
    output tokens as important words in the messages 
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    output pipline model 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [10, 20],
    'clf__estimator__max_depth':[None, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters , verbose=True , n_jobs= -1 ,cv = 3 )
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    output accuracy of the model on the test set 
    '''
    acc = []
    for i1,i2 in zip(np.array(Y_test), model.predict(X_test)):
        acc.append(accuracy_score(i1 , i2))
    return np.mean(acc)


def save_model(model, model_filepath):
    '''
    input model and model path 
    
    save model in path given 
    '''
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()