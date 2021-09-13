from sklearn import tree
from joblib import dump, load
import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--traces', help='file path to traces', default='training_data.pkl')
    parser.add_argument('-m', '--model_save', help='file path to save model (.pkl)', default='model.pkl')
    parser.add_argument('-s', '--static', help='file specifying static parameters (.txt)', 
                        default='static.txt')
    args = parser.parse_args()

    with open(args.static) as f:
        static_data = f.readlines()
    static_data = set([line.rstrip() for line in static_data])
    print(static_data)

    with open(args.traces, 'rb') as f:
        data = pickle.load(f)

    X, Y = [], []
    for trace, satisfy in data:
        new_trace = []
        static_data = []
        added_static = set()
        for param, value in trace:
            if type(value) == tuple: # keep this for now, for groundspeed data
                value = value[0]
            if param == 'time':
                continue
            if param == 'he' or param == 'cte':
                continue
            if param in static_data and param not in added_static:
                static_data.append(float(value))
                added_static.add(param)
            elif param not in static_data:
                new_trace.append(float(value))
        X.append(static_data + new_trace)
        Y.append(int(satisfy))

    print(len(X), len(Y))

    print('Solving decision tree...')
                
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, Y)

    # tree.plot_tree(clf)

    dump(clf, args.model_save)
