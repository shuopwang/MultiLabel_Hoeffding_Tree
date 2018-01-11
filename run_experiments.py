import matplotlib as mpl
mpl.use('TkAgg')
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.classification.lazy.knn_adwin import KNNAdwin, KNN
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.file_stream import FileStream
from my_classifier import MultiLabelHoeffdingTree
from multi_label import BR
from multi_label import LC
dataset = "elec"

# 1. Create a stream
opt = FileOption("FILE", "OPT_NAME", "./data/"+dataset+".csv", "CSV", False)
stream = FileStream(opt, -1, 1)
# 2. Prepare for use
stream.prepare_for_use()
# 2. Instantiate the HoeffdingTree classifier
h = [	BR(h=SGDClassifier(n_iter=1)),
        #KNN(k=10, max_window_size=100, leaf_size=30), 
        #HoeffdingTree(),
        MultiLabelHoeffdingTree(),
     ]
# 3. Setup the evaluator
eval = EvaluatePrequential(pretrain_size=1000, output_file='result_'+dataset+'.csv', max_instances=10000, batch_size=1, n_wait=500, max_time=1000000000, task_type='classification', show_plot=False, plot_options=['performance'])
# 4. Run
eval.eval(stream=stream, classifier=h)
