import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import datetime
import csv
import os

from tqdm import tqdm
from argparse import ArgumentParser as ArgPar
from collections import namedtuple
from objective_functions.dataset import get_data
from objective_functions.model.CNN import CNN


def accuracy(y, target):
    pred = y.data.max(1, keepdim = True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()

    return acc

def print_result(values, model, num, n_jobs):
    f1 = "  {"
    f2 = "}  "
    f_int = "{}:<20"
    f_float = "{}:<20.5f"

    f_vars = ""
    
    for i, v in enumerate(values):
        if type(v) == float:
            f_vars += f1 + f_float.format(i) + f2
        else:
            f_vars += f1 + f_int.format(i) + f2
    
    s = f_vars.format(*values)
    print(s)

    with open("log/{}/{}/log{}.csv".format(model, num, n_jobs), "a", newline = "") as f:
        writer = csv.writer(f, delimiter = ",", quotechar = " ")
        #s += "\n"
        writer.writerow([s])

def train(device, optimizer, learner, train_data, loss_func):
    train_acc, train_loss, n_train = 0, 0, 0
    learner.train()
    bar = tqdm(desc = "Training", total = len(train_data), leave = False)
    
    for data, target in train_data:
        data, target =  data.to(device), target.to(device) 
        y = learner(data)            
        loss = loss_func(y, target)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

        train_acc += accuracy(y, target)

        train_loss += loss.item() * target.size(0)
        n_train += target.size(0)

        bar.set_description("Loss: {0:.6f}, Accuracy: {1:.6f}".format(train_loss / n_train, float(train_acc) / n_train))
        bar.update()
    bar.close()
    
    return float(train_acc) / n_train, train_loss / n_train

def test(device, optimizer, learner, test_data, loss_func):
    test_acc, test_loss, n_test = 0, 0, 0
    bar = tqdm(desc = "Testing", total = len(test_data), leave = False)
    
    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        y = learner(data)
        loss = loss_func(y, target)

        test_acc += accuracy(y, target)
        test_loss += loss.item() * target.size(0)
        n_test += target.size(0)

        bar.update()
    bar.close()
    
    return float(test_acc) / n_test, test_loss / n_test

def main(learner, n_cuda, model, num, n_jobs):

    if torch.cuda.is_available():
        device = torch.device("cuda", n_cuda) 
    else:
        device = torch.device("cpu")
    
    train_data, test_data = get_data(learner.batch_size)
    
    learner = learner.to(device)
    cudnn.benchmark = True

    optimizer = optim.SGD( \
                        learner.parameters(), \
                        lr = learner.lr, \
                        momentum = learner.momentum, \
                        weight_decay = learner.weight_decay, \
                        nesterov = True \
                        )
    
    loss_func = nn.CrossEntropyLoss().cuda()

    milestones = learner.lr_step
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = learner.lr_decay)


    rsl_keys = ["lr", "epoch", "TrainAcc", "TrainLoss", "TestAcc", "TestLoss", "Time"]
    rsl = []
    loss_min = 1.0e+8
    acc_max = 0.0

    print_result(rsl_keys, model, num, n_jobs)
    
    for epoch in range(learner.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_acc, train_loss = train(device, optimizer, learner, train_data, loss_func)     
        
        learner.eval() 

        with torch.no_grad():
            test_acc, test_loss = test(device, optimizer, learner, test_data, loss_func)

        time_now = str(datetime.datetime.today())
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_acc, train_loss, test_acc, test_loss, time_now])})
       
        loss_min = min(loss_min, test_loss)
        acc_max = max(acc_max, test_acc)

        print_result(rsl[-1].values(), model, num, n_jobs)
        scheduler.step()

    return loss_min, acc_max

def func(hp_dict, model, num, n_cuda, n_jobs):
    hp_tuple = namedtuple("_hyperparameters", (var_name for var_name in hp_dict.keys() ) )
    hyperparameters = hp_tuple(**hp_dict)
    learner = eval(model)(hyperparameters)
    
    with open("log/{}/{}/log{}.csv".format(model, num, n_jobs), "w", newline = "") as f:
        writer = csv.writer(f, delimiter = ",", quotechar = " ")
        s = "### Hyperparameters ### \n"

        for name, value in hp_dict.items():
            s += "{}: {} \n".format(name, value)
        s += "\n"
        writer.writerow([s])        

    print("Start Training")
    print("")
    print("")
    loss, acc = main(learner, n_cuda, model, num, n_jobs)
    
    print(loss, acc)
    hp_dict["loss"] = loss
    save_evaluation(hp_dict, model, num)

def save_evaluation(hp_dict, model, num):
    
    if os.path.isfile("evaluation/{}/{}/evaluation.csv".format(model, num)):
    
        with open("evaluation/{}/{}/evaluation.csv".format(model, num), "r", newline = "") as f:
            head = list(csv.reader(f, delimiter = ",", quotechar = "'"))[0]
        
        with open("evaluation/{}/{}/evaluation.csv".format(model, num), "a", newline = "") as f:
            writer = csv.writer(f, delimiter = ",", quotechar = "'")
            row = [hp_dict[k] for k in head]
            writer.writerow(row)
    else:
        with open("evaluation/{}/{}/evaluation.csv".format(model, num), "w", newline = "") as f:
            writer = csv.DictWriter(f, delimiter = ",", quotechar = "'", fieldnames = hp_dict.keys())
            writer.writeheader()
            writer.writerow(hp_dict)

    