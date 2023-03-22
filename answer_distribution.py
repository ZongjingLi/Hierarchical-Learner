from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from dtreeviz.trees import dtreeviz


rf = RandomForestClassifier(n_estimators=100,
                            max_depth=3,
                            max_features='auto',
                            min_samples_leaf=4,
                            bootstrap=True,
                            n_jobs=-1,
                            random_state=0)
rf.fit(X, y)

viz = dtreeviz(rf.estimators_[99], X, y,
               target_name="SizeClass",
               feature_names=X_train.columns,
               class_names=list(y_train.feature_names),
               title="100th decision tree")

viz.save("decision_tree.svg")