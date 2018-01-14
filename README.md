# MultiLabel_Hoeffding_Tree

Try to Create the hoeffding tree which could work with multi-label's problem and data stream.  

Here we extend the class Hoeffding tree from skmultiflow. 

Here we need to change the class info_gain_split_criterion, in order to adopt the multi-label situation. NominalAttributeClassObserver also needs to be changed, because in the hoeffding tree we use this class to store the distribution per attribute.  

Once those thing have done, the project is done.

