# Multi-label Hoeffding Tree
## MultiLabel_Hoeffding_Tree.classification.BR_hoeffding
```py
    class MultiLabel_Hoeffding_Tree.classification.MultiLabelHoeffdingTree()
```
<b>BR_Hoeffding</b> is a class based on <b>Binary Relevance(BR)</b> method who is capable of performing multi-labels classification on a stream dataset.

The Hoeffding tree is the state-of-the-art classifier for single-label data streams, and performs prediction by choosing the majority class at each leaf by default. Predictive accuracy can be increased by adding naive Bayes models at the leaves of the trees. Here, we extend the Hoeffding Tree to deal with multi-label data: a BR Hoeffding Tree.

Same as other classifiers, <b>BR_Hoeffding</b> takes as input two array: an array X, sparse or dense, of size [n_samples, n_features] holding the training samples, and an array Y includes serval sub-array, and the sub-array store 1 or 0 to present yes or no:

```py
    >>> from MultiLabel_Hoeffding_Tree.classification.BR_hoeffding import BR_Hoeffding
    >>> X = [[0,0,1,0,0],[1,1,1,0,1]]
    >>> Y = [[0,1,0],[1,1,1]]
    >>> clf = BR_Hoeffding()
    >>> clf.fit(X,Y)
```
And also can use `partial_fit` to fit the online stream data, the format of Input is the same as 'fit':

```py
    >>> from MultiLabel_Hoeffding_Tree.classification.BR_hoeffding import BR_Hoeffding
    >>> X = [[0,0,1,0,0]]
    >>> Y = [[0,1,0]]
    >>> clf = BR_Hoeffding()
    >>> clf.partial_fit(X,Y)
```
  <b>note</b>

  Currently, BR_Hoeffding is only capable of binary classification (where the labels are [0,1]), because we only use 0 and 1 to present yes or no with this label.

After being fitted, the model can then be used to predict the class of samples:
```py
  clf.predict([0,0,1,0,1])
```
A online streaming test is as follows, using the IMDB datasets with prequential method( interleaved test-then-train method) to evaluate BR_Hoeffding:
```py
  from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
  from skmultiflow.options.file_option import FileOption
  from skmultiflow.data.file_stream import FileStream
  from MultiLabel_Hoeffding_Tree.classification.BR_hoeffding import BR_Hoeffding
  opt = FileOption("FILE", "OPT_NAME", "imdb.csv", "CSV", False)
  # The targets(labels) are in the first 27 columns
  stream = FileStream(opt, 0, 27)
  stream.prepare_for_use()
  eval = EvaluatePrequential(pretrain_size=1000, output_file='result_' + dataset + '.csv', max_instances=10000, batch_size=1,n_wait=500, max_time=1000000000, task_type='multi_output', show_plot=False)
  eval.eval(stream=stream, classifier=BR_Hoeffding())
```

## Tree algorithms: C4.5
C4.5 is the successor to ID3 and removed the restriction that features must be categorical by dynamically defining a discrete attribute (based on numerical variables) that partitions the continuous attribute value into a discrete set of intervals. C4.5 converts the trained trees (i.e. the output of the ID3 algorithm) into sets of if-then rules. These accuracy of each rule is then evaluated to determine the order in which they should be applied. Pruning is done by removing a rule’s precondition if the accuracy of the rule improves without it.

## Mathematical formulation
Information gain:
  - ![](https://raw.githubusercontent.com/shuopwang/MultiLabel_Hoeffding_Tree/master/01.png?token=AJ5Yeh9m5GJsqJLiS6RIayLZBCTgkKfEks5aiek3wA%3D%3D)

## OneVSRestClassifier
The similar method OneVsRest to binery relevance in sklearn:
> http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
