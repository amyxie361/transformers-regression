import csv
import json
import sys
import numpy as np
import datasets
import glob
import re

def regression_metrics(total, golden_labels, model1_labels, model2_labels):
    delta1 = 0  # abs(model1 - golden)
    delta2 = 0
    side_flip = 0
    good_flip = 0
    golden_labels = [float(label) for label in golden_labels]
    model1_labels = [float(label) for label in model1_labels]
    model2_labels = [float(label) for label in model2_labels]
    for i in range(total):
        delta_1 = abs(golden_labels[i] - model1_labels[i])
        delta1 += delta_1
        delta_2 = abs(golden_labels[i] - model2_labels[i])
        delta2 += delta_2
        if (golden_labels[i] - model1_labels[i]) * (golden_labels[i] - model2_labels[i]) < 0:
            side_flip += 1
            if delta_1 > delta_2:
                good_flip += 1
    total = float(total)
    # print(“{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}“.format(task,  delta1/total, delta2/total, side_flip/total, good_flip/total, total))
    return {"delta1": delta1, "delta2": delta2, "side_flip": side_flip, "good_flip": good_flip, "Total": total}

def classification_metrics(total, golden_labels, model1_labels, model2_labels, task=None, name1="", name2=""):
    m1t_m2t = 0
    m1t_m2f = 0
    m1f_m2t = 0
    m1f_m2f = 0
    if total != len(model1_labels):
        print(task, total, len(model1_labels), len(model2_labels))
        return 0
    if total != len(model2_labels):
        print(task, total, len(model1_labels), len(model2_labels))
        return 0
    for i in range(total):
        #print(golden_labels[i], model1_labels[i], model2_labels[i])
        #print(type(golden_labels[i]), type(model1_labels[i]), type(model2_labels[i]))
        #input()
        #if model1_labels[i]!=model2_labels[i]:
        #    print(golden_labels[i], model1_labels[i], model2_labels[i])
        if golden_labels[i] == model1_labels[i]:
            if golden_labels[i] == model2_labels[i]:
                m1t_m2t += 1
            else:
                m1t_m2f += 1
        else:
            if golden_labels[i] == model2_labels[i]:
                m1f_m2t += 1
            else:
                m1f_m2f += 1
    total = float(total)
    # print(“{}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}“.format(
    # print(“{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}“.format(
    print(name1, name2)
    print("{}\t{:.4f}\t{:.4f}".format(
        task, m1t_m2f / total,
              # m1f_m2t / total, m1f_m2f / total, m1t_m2t / total,
              #m1t_m2f / (m1t_m2f + float(m1t_m2t)),  # Negative Flip Rate
              #m1f_m2t / (m1f_m2t + float(m1t_m2t)),  # Positive Flip Rate
              #m1f_m2t / (m1t_m2f + float(m1f_m2t) + 0.00001),  # Correct Flip Rate
              #m1t_m2f / total + m1t_m2t / total,  # Old correct
              m1f_m2t / total + m1t_m2t / total,  # New correct
              #(m1t_m2f ) / (m1t_m2f + float(m1f_m2f)) #AN-NFR
    ))
    return {"Negative_Flip": m1t_m2f,
            "Positive_Flip": m1f_m2t,
            "Negative_NoFlip": m1f_m2f,
            "Positive_NoFlip": m1t_m2t,
            "Total": total}

def evaluate_model_pair_task_flip_rate(
        model1, model2, task,
        golden_path="/home/ec2-user/SageMaker/data",
        model_path="results/output",
        model_filename="eval_results_{}.txt"):
    '''
    We consider model1 as the old model and model2 as the new model.
    This function caculate the four matrices on the given task:
        Negative Flip - the old model gives the correct label while the new model don’t;
        Positive Flip - the new model gives the correct label while the old model don’t;
        Negative NoFlip - the old model and the new model both give wrong label;
        Postive NoFlip - the old model and the new model both give correct lable.'''
    if "MNLI" in task:
        golden_file = open("{}/{}/{}".format(golden_path, "MNLI", golden_filename),'r')
        golden_labels = [line.split("\t")[-1].strip() for line in golden_file][1:]
    else:
        dataset = datasets.load_dataset('glue', task.lower(), split='validation')
        #golden_file = open("{}/{}".format(golden_path, "dev.tsv"),'r')
        #golden_labels = [line.split("\t")[0].strip() for line in golden_file][1:]
        #golden_labels = [int(x) for x in golden_labels]
        golden_labels = [int(x["label"]) for x in dataset]
    if model1 == model2:
        #print(“New model is the same as the old model”)
        return 0
    try:
        model1_file = open(model1,'r')
    except FileNotFoundError:
        print("Old model not trained on ", task)
        return 0
    model1_labels = [line.split("\t")[-1].strip() for line in model1_file][1:]
    if task.lower() == "mrpc":
        model1_labels = [1 if t=="equivalent" else 0 for t in model1_labels]
    elif task.lower() == "qqp":
        model1_labels = [1 if t=="duplicate" else 0 for t in model1_labels]
    elif task.lower() == "rte":
        model1_labels = [0 if t == "entailment" else 1 for t in model1_labels]
    try:
        model2_file = open(model2,'r')
        
    except FileNotFoundError:
        print(model2)
        print("New model not trained on ", task)
        return 0
    model2_labels = [line.split("\t")[-1].strip() for line in model2_file][1:]
    if task.lower() == "mrpc":
        model2_labels = [1 if t == "equivalent" else 0 for t in model2_labels]
    elif task.lower() == "qqp":
        model2_labels = [1 if t=="duplicate" else 0 for t in model2_labels]
    elif task.lower() == "rte":
        model2_labels = [0 if t == "entailment" else 1 for t in model2_labels]
    total = len(golden_labels)
    if task == 'STS-B':
        return 0
        # return regression_metrics(total, golden_labels, model1_labels, model2_labels)
    else:
        return classification_metrics(total, golden_labels, model1_labels, model2_labels, task=task, name1=model1, name2=model2)

def evaluate_model_pair(
        model1, model2, tasks,
        golden_path="glue_data",
        model_path="results/output"):
    '''
    This function caculate the four matrices on all the glue tasks
    '''
    results = {}
    for task in tasks:
        if task == "MNLI":
            result1 = evaluate_model_pair_task_flip_rate(
                model1, model2, task,
                model_filename="eval_results_{}.txt")
            #result2 = evaluate_model_pair_task_flip_rate(
            #    model1, model2, “MNLI-mm”,
            #    golden_filename=“dev_mismatched-index_label.tsv”,
            #    model_filename=“test_results_{}-mm.txt”)
            results["MNLI"] = result1
            #results["MNLI-mm"] = result2
        else:
            result = evaluate_model_pair_task_flip_rate(
                model1, model2, task)
            results[task] = result
    #print(results)
    return results
#tasks = ["MRPC"]
tasks= ["RTE"]
import os
#model2 = [f.path for f in os.scandir(model2_dir) if f.is_dir()]
#for model_old in model1:
#    for model_new in model2:
        #print(model_old, model_new)
#model_old = "mrpc/baseline_base_seed0_BS64_lr9e-5_epoch3/predict_results_mrpc.txt"
#model_old = "qqp/baseline_base_seed0_BS64_lr9e-5_epoch3/predict_results_qqp.txt"
#model_old = "rte/baseline_base_seed0_BS32_lr9e-5_epoch5/predict_results_rte.txt"
#seed3_BS16_lr2e-5_initepoch1_continueepoch2_lr24e-6

model_old = "rte/baseline_bert_base_uncased_seed0_BS32_lr3e-5_epoch5/predict_results_rte.txt"
#model_old = "rte/baseline_base_seed0_BS32_lr3e-5_epoch5/predict_results_rte.txt"

accumulated_results = {}
model = "large"
"""
for s in [1,2,3,4,5]:
    for epoch in [5]:
        for lr in ["3e-5"]:
            for epoch in [5]:
                model_new = "./rte/baseline_{model}_seed{seed}_BS32_lr{lr}_epoch{epoch}/predict_results_rte.txt".format(model=model, seed=s, lr=lr, epoch=epoch)
                metrics = evaluate_model_pair(model_old, model_new, tasks)
                name = model_new.replace("_seed{}".format(s), "")
                nfr = metrics[tasks[0]]["Negative_Flip"] / metrics[tasks[0]]['Total']
                acc = (metrics[tasks[0]]["Positive_Flip"] + metrics[tasks[0]]["Positive_NoFlip"]) / metrics[tasks[0]]['Total']
                if not name in accumulated_results:
                    accumulated_results[name] = {"nfr": [nfr], "acc": [acc]}
                else:
                    accumulated_results[name]["nfr"].append(nfr)
                    accumulated_results[name]["acc"].append(acc)
"""
"""
for s in [1,2,3,4,5]:
    for epoch in [5]:
        for lr in ["1e-5", "3e-5"]:
            for temp in [1, 2, 3]:
                for alpha in [0.5, 1.0, 2.0]:
                    for epoch in [5]:
                        model_new = "./distill/rte/bert-large_seed{seed}_BS16_lr{lr}_epoch{epoch}_temp{temp}_alpha{alpha}/predict_results_rte.txt".format(
                            seed=s, lr=lr, epoch=epoch, temp=temp, alpha=alpha)
                        metrics = evaluate_model_pair(model_old, model_new, tasks)
                        name = model_new.replace("_seed{}".format(s), "")
                        print (metrics)
                        nfr = metrics[tasks[0]]["Negative_Flip"] / metrics[tasks[0]]['Total']
                        acc = (metrics[tasks[0]]["Positive_Flip"] + metrics[tasks[0]]["Positive_NoFlip"]) / \
                              metrics[tasks[0]]['Total']
                        if not name in accumulated_results:
                            accumulated_results[name] = {"nfr": [nfr], "acc": [acc]}
                        else:
                            accumulated_results[name]["nfr"].append(nfr)
                            accumulated_results[name]["acc"].append(acc)
"""
# get all the folders in the specified directory
#ROOT_FOLDER = "gated_v3_bugfix/rte/"
#ROOT_FOLDER = "rte/"
ROOT_FOLDER = "gated_electra2/rte/"
#ROOT_FOLDER = "gated_v3_bugfix_debug/rte/"
#ROOT_FOLDER = "gated_v4/rte/"

for folder in glob.glob(ROOT_FOLDER + "*"):
    #if not "_continueepoch" in folder:
    #    continue
    #if not "notraingate" in folder:
    #    continue
    if "seed0" in folder:
        continue
    #if not "traingated_base" in folder or not "continueepoch" in folder:
    #    continue
    if not "dropgate" in folder:
        continue
    if not "traingated_base" in folder:
        continue
    #if "continueepoch" in folder:
    #    continue
    #if not "baseline_bert_large_uncased" in folder:
    #    continue
    #if not "electra_old_new_base_uncased_seed" in folder:
    #    continue
    #if not "electrabase_uncased_old_bertbase_cased_new" in folder:
    #    continue
    #if not (("gatedfinal" in folder) and ("gatesize-1" in folder or "gatesize100" in folder)):
    #    continue
    #if not "large" in folder:
    #    continue
    #if not "old_new" in folder:
    #    continue

    model_new = folder + "/predict_results_rte.txt"
    metrics = evaluate_model_pair(model_old, model_new, tasks)
    name = re.sub(r"_seed\d", "", model_new)
    try:
        metrics_test = metrics[tasks[0]]["Negative_Flip"]
    except:
        continue
    nfr = metrics[tasks[0]]["Negative_Flip"] / metrics[tasks[0]]['Total']
    nfi = metrics[tasks[0]]["Negative_Flip"] / (metrics[tasks[0]]["Negative_Flip"] + metrics[tasks[0]]["Negative_NoFlip"])
    acc = (metrics[tasks[0]]["Positive_Flip"] + metrics[tasks[0]]["Positive_NoFlip"]) / \
          metrics[tasks[0]]['Total']
    if not name in accumulated_results:
        accumulated_results[name] = {"nfr": [nfr], "nfi": [nfi], "acc": [acc]}
    else:
        accumulated_results[name]["nfr"].append(nfr)
        accumulated_results[name]["nfi"].append(nfi)
        accumulated_results[name]["acc"].append(acc)
print ('=======')

for name in accumulated_results.keys():
    nfrs = accumulated_results[name]["nfr"]
    nfis = accumulated_results[name]["nfi"]
    accs = accumulated_results[name]["acc"]
    nfr_mean, nfr_std = np.mean(nfrs), np.std(nfrs)
    nfi_mean, nfi_std = np.mean(nfis), np.std(nfis)
    acc_mean, acc_std = np.mean(accs), np.std(accs)
    nfr_mean = np.around(nfr_mean, 4) * 100
    nfr_std = np.around(nfr_std, 4) * 100
    nfi_mean = np.around(nfi_mean, 4) * 100
    nfi_std = np.around(nfi_std, 4) * 100
    acc_mean = np.around(acc_mean, 4) * 100
    acc_std = np.around(acc_std, 4) * 100
    print ("Name {}".format(name))
    print ("NFIs")
    print (nfis)
    print ("NFRs")
    print (nfrs)
    print ("Accs")
    print (accs)
    print ("NFR {}+-{}, NFI {}+-{}, Acc {}+-{}".format(nfr_mean, nfr_std, nfi_mean, nfi_std, acc_mean, acc_std))
    print ("---------")
