This is a description of the final model that got me my best score.
It is a mix of 3 averaged neural networks, a Calibrated Random Forest, a Gradient Boost and 2 XGBoosts.

The formula is as follows:
0.08 * KATZ_S1 + 0.17 * LAS_S1 + 0.1 * CB RF + 0.11 * GB + 0.06 * XGB1 + 0.08 * KITTY_S1 + 0.4 * XGB2

CB RF
=====

This is a CalibratedClassifierCV wrapped around a Random Forest classifier (both from scikit-learn).
Parameters: Calibrated Classifier: isotonic, cv=7; Random Forest: n_estimators=700, max_features=10

GB
==

This is the GradientBoostingClassifier from scikit-learn.
Parameters: 600 estimators, max_depth=8, max_features = 25, learn_rate = 0.03

XGB1
====

This is the XGBoost classifier.
Parameters: 800 trees, 'max_depth': 10, 'eta': 0.04, 'colsample_bytree': 0.75, 'min_child_weight': 2, 'gamma': 0.25

XGB2
====

This is the XGBoost classifier.
Parameters: 1200 trees, 'max_depth': 20, 'eta': 0.04, 'colsample_bytree': 0.5, 'min_child_weight': 2, 'gamma': 0.88

KATZ_S1
=======

This is an average of 4 neural networks (using nolearn and Lasagne), the base architecture was::

    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('dense2', DenseLayer),
               ('output', DenseLayer)]

KATZ_S1 - network 1
-------------------

Parameters::

    learn_r = 0.14
    mom_r = 0.85
    learn_dec = 0.61
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    input_shape=(None, 93),
                    dense0_num_units=1000,
                    dropout0_p=0.2,
                    dense1_num_units=700,
                    dense2_num_units=600,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=18)

KATZ_S1 - network 2
-------------------
Parameters::

    learn_r = 0.09
    mom_r = 0.85
    learn_dec = 0.69
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    input_shape=(None, 93),
                    dense0_num_units=1100,
                    dropout0_p=0.22,
                    dense1_num_units=800,
                    dense2_num_units=600,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=18)

KATZ_S1 - network 3
-------------------
Parameters::

    learn_r = 0.12
    mom_r = 0.85
    learn_dec = 0.64
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    input_shape=(None, 93),
                    dense0_num_units=1300,
                    dropout0_p=0.25,
                    dense1_num_units=1000,
                    dense2_num_units=800,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=15)

KATZ_S1 - network 4
-------------------
Parameters::

    learn_r = 0.11
    mom_r = 0.85
    learn_dec = 0.64
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    batch_iterator_train=BatchIterator(batch_size=100),
                    batch_iterator_test=BatchIterator(batch_size=100),
                    input_shape=(None, 93),
                    dense0_num_units=1300,
                    dropout0_p=0.25,
                    dense1_num_units=1000,
                    dense2_num_units=800,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=17)

LAS_S1
======

This is an average of 4 neural networks (using nolearn and Lasagne), the base architecture was::

    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense2', DenseLayer),
               ('output', DenseLayer)]

LAS_S1 - network 1
------------------
Parameters::

    learn_r = 0.1
    mom_r = 0.9
    learn_dec = 0.65
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    input_shape=(None, 93),
                    dense0_num_units=900,
                    dropout0_p=0.26,
                    dense1_num_units=1300,
                    dropout1_p=0.23,
                    dense2_num_units=600,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=14)

LAS_S1 - network 2
------------------
Parameters::

    learn_r = 0.11
    mom_r = 0.85
    learn_dec = 0.66
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    batch_iterator_train=BatchIterator(batch_size=100),
                    batch_iterator_test=BatchIterator(batch_size=100),
                    input_shape=(None, 93),
                    dense0_num_units=1000,
                    dropout0_p=0.26,
                    dense1_num_units=1400,
                    dropout1_p=0.23,
                    dense2_num_units=700,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=14)

LAS_S1 - network 3
------------------
Parameters::

    learn_r = 0.1
    mom_r = 0.91
    learn_dec = 0.64
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    batch_iterator_train=BatchIterator(batch_size=100),
                    batch_iterator_test=BatchIterator(batch_size=100),
                    input_shape=(None, 93),
                    dense0_num_units=1200,
                    dropout0_p=0.3,
                    dense1_num_units=1600,
                    dropout1_p=0.21,
                    dense2_num_units=600,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=15)

LAS_S1 - network 4
------------------
Parameters::

    learn_r = 0.11
    mom_r = 0.88
    learn_dec = 0.66
    mom_max = 0.97
    clf = NeuralNet(layers=layers0,
                    batch_iterator_train=BatchIterator(batch_size=110),
                    batch_iterator_test=BatchIterator(batch_size=110),
                    input_shape=(None, 93),
                    dense0_num_units=1200,
                    dropout0_p=0.32,
                    dense1_num_units=1600,
                    dropout1_p=0.21,
                    dense2_num_units=800,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=17)

KITTY_S1
========

This is an average of 4 neural networks (using nolearn and Lasagne), the base architecture was::

    layers0 = [('input', InputLayer),
               ('dropout0', DropoutLayer),
               ('dense0', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense1', DenseLayer),
               ('dense2', DenseLayer),
               ('dense3', DenseLayer),
               ('output', DenseLayer)]

KITTY_S1 - network 1
--------------------
Parameters::

    learn_r = 0.11
    mom_r = 0.88
    learn_dec = 0.71
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    input_shape=(None, 93),
                    dropout0_p=0.17,
                    dense0_num_units=1000,
                    dropout1_p=0.08,
                    dense1_num_units=1000,
                    dense2_num_units=700,
                    dense3_num_units=700,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=20)

KITTY_S1 - network 2
--------------------
Parameters::

    learn_r = 0.115
    mom_r = 0.88
    learn_dec = 0.71
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    batch_iterator_train=BatchIterator(batch_size=100),
                    batch_iterator_test=BatchIterator(batch_size=100),
                    input_shape=(None, 93),
                    dropout0_p=0.18,
                    dense0_num_units=1000,
                    dropout1_p=0.09,
                    dense1_num_units=1000,
                    dense2_num_units=900,
                    dense3_num_units=900,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=20)

KITTY_S1 - network 3
--------------------
Parameters::

    learn_r = 0.115
    mom_r = 0.88
    learn_dec = 0.71
    mom_max = 0.95
    clf = NeuralNet(layers=layers0,
                    batch_iterator_train=BatchIterator(batch_size=100),
                    batch_iterator_test=BatchIterator(batch_size=100),
                    input_shape=(None, 93),
                    dropout0_p=0.25,
                    dense0_num_units=1500,
                    dropout1_p=0.13,
                    dense1_num_units=1300,
                    dense2_num_units=1000,
                    dense3_num_units=1000,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=20)

KITTY_S1 - network 4
--------------------
Parameters::

    learn_r = 0.12
    mom_r = 0.87
    learn_dec = 0.71
    mom_max = 0.99
    clf = NeuralNet(layers=layers0,
                    batch_iterator_train=BatchIterator(batch_size=100),
                    batch_iterator_test=BatchIterator(batch_size=100),
                    input_shape=(None, 93),
                    dropout0_p=0.22,
                    dense0_num_units=1200,
                    dropout1_p=0.12,
                    dense1_num_units=1200,
                    dense2_num_units=1100,
                    dense3_num_units=900,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(learn_r)),
                    update_momentum=theano.shared(float32(mom_r)),
                    on_epoch_finished=[AdjustVariable('update_learning_rate', start=learn_r, decay_rate=learn_dec),
                                      AdjustVariableLin('update_momentum', start=mom_r, stop=mom_max),],
                    eval_size=0.05,
                    verbose=1,
                    max_epochs=20)