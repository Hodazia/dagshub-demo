import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

import dagshub
dagshub.init(repo_owner='Hodazia', repo_name='dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Hodazia/dagshub-demo.mlflow")

mlflow.set_experiment('nested-run')
# Start the parent run
with mlflow.start_run(run_name="Parent Run") as parent_run:
    # Log a parameter at the parent level
    mlflow.log_param("Parent Param", "Parent Value")
    
    for n_estimators in [10, 50, 100]:  # Experiment with different numbers of trees
        # Start a child run
        with mlflow.start_run(run_name=f"Child Run {n_estimators} estimators", nested=True):
            # Train a model
            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log parameters and metrics in the child run
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_metric("accuracy", accuracy)
            
            # Log the model
            mlflow.sklearn.log_model(clf, f"RandomForest-{n_estimators}")

    # Log a final metric at the parent level
    mlflow.log_metric("Parent Metric", 0.9)  # Example of aggregate performance

print("Nested runs created!")
