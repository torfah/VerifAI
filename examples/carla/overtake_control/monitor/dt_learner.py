from sklearn import tree
from sklearn.tree import _tree
import pandas as pd 
import numpy as np
#import graphviz
import os

def tree_to_code(tree, feature_names, file_name, out_dir="."):

    code_file = open(f"{out_dir}/{file_name}.py","w")

    tree_ = tree.tree_
    feature_name = [feature_names[i] 
                    if i != _tree.TREE_UNDEFINED else "undefined!" 
                    for i in tree_.feature]
    code_file.write("def execute(input_map):\n")

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            code_file.write("{}if input_map[\"{}\"] <= {}:\n".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            code_file.write("{}else:  # if input_map[\"{}\"] > {}\n".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            code_file.write("{}return {}\n".format(indent, np.argmax(tree_.value[node])))

    recurse(0, 1)
    code_file.close()

def learn_dt(csv_file_path, label, features_names, file_name, visualization=False,out_dir="."):

    data = pd.read_csv(csv_file_path)   

    X = data[features_names]
    Y = data[label]

    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(X,Y)

    # Export tree
    os.system(f"rm -r {out_dir}/dt")
    os.system(f"mkdir {out_dir}/dt")
    ## Visualization 
    if visualization == True:
        dot_data = tree.export_graphviz(dt, out_file=None, feature_names=features_names, class_names=["0","1"], filled=True)
        graph = graphviz.Source(dot_data)
        graph.render(f"{out_dir}/dt/{file_name}")
        os.system(f"dot -Tpng {out_dir}/dt/{file_name} -o {out_dir}/dt/{file_name}.png")
   
    ## Executable Code
    tree_to_code(dt,features_names, file_name, f"{out_dir}/dt")

    return dt

# feature_names = ['init_pos@0','init_head@0','day_time@0','clouds@0','pos@0']
# class_names = ['flag']
# dt = learn_dt("training_data/training_data.csv", class_names, feature_names, "dt",True)

