Kaggle Otto Group Classification Challenge
==========================================

This was one of the most interesting challenges I've taken part in, for the following reasons:

    #. The small amount of data
    #. Features were already given, and no idea as to what they mean (which for me made feature engineering impossible)
    #. Lots of interesting ideas on the forum – I learned about Calibrated classifiers in scikit-learn, XGBoost, nolearn and Lasagne

So mostly the competition was about tuning classifiers, which I never really did before, at least in such large amounts, and this has been a great experience and opportunity to learn more about various parameters of various classification methods. If only I had more knowledge about Neural Nets, I think I would've scored higher, but I kind of got stuck with them, and couldn't get the extremely low scores some other forum users were getting (0.42 or so, if I recall correctly).
A full list of all my submits (with their scores) can be found here: https://github.com/Kunstmord/kaggle-otto/blob/master/descriptions.txt
A companion list describing the architecture of used Neural Nets can be found here: https://github.com/Kunstmord/kaggle-otto/blob/master/CVscoresLLoss

My best result was a mix of multiple models, namely, 3 averaged neural nets (each one was an average of from of 4 networks), 2 XGBoosts, 1 Gradient Boosting Classifier
and 1 Calibrated Classifier on top of a Random Forest. The three neural networks were with quite different architectures (4 layers and 2 dropout layers; 3 layers and 2 dropout layers; 3 layers and 1 dropout layer), and each one was an average of similar-sized nets (I changed around the sizes and some learn-rate related parameters and then just took a simple average, this improved the result by 0.01 as compared to using a single neural network of each architecture type).

Full description of the model can be found here: https://github.com/Kunstmord/kaggle-otto/blob/master/final.rst

And here's a list of best results (on the public leaderboard) obtained using single models (I could've gotten better results for Random Forests, but I started wrapping them in the Calibrated Classifier and stopped trying to improve the RF by itself):

    #. SVM - 7.15267 (I didn't use the version that can predict probabilities instead of labels, that's the reason for the low score)
    #. Logistic Regression – 0.66674
    #. Random Forest – 0.55300 (n_estimators=200, max_features=20)
    #. AdaBoost on 15 Random Forests 0.52698 (n_estimators=400, max_features=25)
    #. nolearn.DBN (a simple neural net) 0.47606
    #. Calibrated Random Forest 0.46996 (Random Forest settings: n_estimators=700, max_features=10; CalibratedClassifierCV settings: method='isotonic', cv=7)
    #. nolearn+Lasagne neural net 0.46408
    #. Gradient Boosting Classifier (700 estimators, max_depth=7, max_features = 20, learn_rate = 0.03) 0.45516
    #. XGBoost 0.43306 ('max_depth': 20, 'eta': 0.04, 'colsample_bytree': 0.5, 'min_child_weight': 2, 'gamma': 0.88 )

To find the best weights for mixing, I did 3-fold cross-validation using a fixed random seed (so that the train/test sets would always be the same) for the models I wanted to try out, and saved the results.
Then I loaded the results I wanted to try out and just checked all possible combinations of weights (first, with a large step of 0.05, then with a small step of 0.01 to refined the results).

Code
====
I wrote everything in Jupyter notebooks. Most of them are here (I did not upload some of the notebooks which are just copies of other ones with some different parameters):

A basic notebook for creation of CV files that I used is this: https://github.com/Kunstmord/kaggle-otto/blob/master/basic_clf_CV.ipynb
This is a notebook to find the optimal weights using the CV predictions: https://github.com/Kunstmord/kaggle-otto/blob/master/best-avgs-new.ipynb
This is a notebook for a basic submission: https://github.com/Kunstmord/kaggle-otto/blob/master/basic_clf.ipynb
This is a notebook for a submissions which is a weighted sum of other submissions: https://github.com/Kunstmord/kaggle-otto/blob/master/basic_clf-averaging.ipynb