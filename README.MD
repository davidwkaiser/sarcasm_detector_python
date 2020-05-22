# Sarcasm Detector Model
A model created to classify a text as sarcastic or not.

Creates a Document Text Matrix (DTM), on which a clasification model is trained to label a comment as sarcastic or not.

Trained and tested on a random sample drawn from a [Kaggle dataset](https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit) of 1M Reddit comments which were tagged (or not) by the author as sarcastic, evenly split.

This project was originally created in R, for my Data Science Fundamentals class. I am now recreating in Python so I can deploy for your amusement. 

The Python version of this now works. I tested it using Logistic Regression, Random Forest, and Naive Bayes classifiers. THe original project in R managed to get a 71% score using a Binomial classifier, but the best I have been able to manage in Python has been about 67%, and that's even with some tuning of hyperparameters and increasing the sample size four fold. 

You can see the original project [HERE](https://github.com/davidwkaiser/sarcasm_detector)

To Do:
- ~~Get it working in Python.~~. 
- ~~Test multiple classifiers.~~. 
- ~~Create a web app that takes a sentence and returns a prediction.~~.  
- ~~Deploy to the web (Heroku? Colab?)~~.   

You can see the deployed app [HERE](https://stormy-brushlands-12264.herokuapp.com/texts/)
You can see the repo for the deployed app [HERE](https://github.com/davidwkaiser/sarcasm_detector_python_app)
