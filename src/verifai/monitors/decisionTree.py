from sklearn import tree
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_args('-t', '--traces', help='file path to traces', default='all.pkl')
    parser.add_args('-s', '--model_save', help='file path to save model (.pkl)', default='model.pkl')
    args = parser.parse_args()

    with open(args.traces, 'rb') as f:
        data = pickle.load(f)

    X = [d[0] for d in data]
    Y = [d[1] for d in data]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, Y)

    pickle.dumps(clf)

    decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
    pickle.dump(decision_tree_model, decision_tree_model_pkl)
    # Close the pickle instances
    decision_tree_model_pkl.close()
