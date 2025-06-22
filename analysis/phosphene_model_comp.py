import pandas as pd
import os
from scipy.stats import mannwhitneyu
import numpy as np

human_trials = pd.read_csv(os.path.join("summary_datasets", "accuracy_by_task_implant_model_subject.csv"))
model_predictions = pd.read_csv(os.path.join("summary_datasets", "accuracy_by_task_implant_model_run.csv"))
results_human = []
results_model = []

print(human_trials[human_trials["model_type"] == "streaky"]["accuracy"].median() - human_trials[human_trials["model_type"] == "pointy"]["accuracy"].median())
print(mannwhitneyu(human_trials[human_trials["model_type"] == "streaky"]["accuracy"], human_trials[human_trials["model_type"] == "pointy"]["accuracy"]))
print(model_predictions[model_predictions["model_type"] == "streaky"]["accuracy"].median() - model_predictions[model_predictions["model_type"] == "pointy"]["accuracy"].median())
print(mannwhitneyu(model_predictions[model_predictions["model_type"] == "streaky"]["accuracy"], model_predictions[model_predictions["model_type"] == "pointy"]["accuracy"]))

for task in human_trials["task"].unique():
    for implant_type in human_trials["implant_type"].unique():
        df = human_trials[
            (human_trials["task"] == task) & 
            (human_trials["implant_type"] == implant_type)]
        if len(df) > 0:
            streaky_acc = df[df["model_type"] == "streaky"]["accuracy"]
            pointy_acc = df[df["model_type"] == "pointy"]["accuracy"]
            u_stat, p_value = mannwhitneyu(streaky_acc, pointy_acc)
            results_human.append({
                "task": task,
                "implant_type": implant_type,
                "p_value": p_value,
                "observed_difference": np.median(streaky_acc) - np.median(pointy_acc),
            })
            df = model_predictions[
                (model_predictions["task"] == task) & 
                (model_predictions["implant_type"] == implant_type)]
            if len(df) > 0:
                streaky_acc = df[df["model_type"] == "streaky"]["accuracy"]
                pointy_acc = df[df["model_type"] == "pointy"]["accuracy"]
                u_stat, p_value = mannwhitneyu(streaky_acc, pointy_acc)
                results_model.append({
                    "task": task,
                    "implant_type": implant_type,
                    "p_value": p_value,
                    "observed_difference": np.median(streaky_acc) - np.median(pointy_acc),
                })
results_human = pd.DataFrame(results_human)
results_model = pd.DataFrame(results_model)
print(results_human.to_markdown(index=False))
print(results_model.to_markdown(index=False))


