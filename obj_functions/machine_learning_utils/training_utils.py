import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import datetime
import csv
from tqdm import tqdm


def accuracy(y, target):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()
    return acc


def start_train(model, train_data, test_data, cuda_id, save_path):
    device = torch.device("cuda", cuda_id) if torch.cuda.is_available() else torch.device("cpu")
    print_resource(torch.cuda.is_available(), cuda_id, save_path)
    model = model.to(device)
    cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), lr=model.lr, momentum=model.momentum, weight_decay=model.weight_decay, nesterov=True)
    loss_func = nn.CrossEntropyLoss().cuda()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=model.lr_step, gamma=model.lr_decay)

    rsl, rsl_keys = [], ["lr", "epoch", "TrA", "TrL", "TeA", "TeL", "Time"]
    loss_min, acc_max = 1.0e+8, 0.0
    print_result(rsl_keys, save_path)

    for epoch in range(model.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_acc, train_loss = train(device, optimizer, model, train_data, loss_func)
        model.eval()
        with torch.no_grad():
            test_acc, test_loss = test(device, optimizer, model, test_data, loss_func)
        scheduler.step()

        time_now = str(datetime.datetime.today())[:-10]
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_acc, train_loss, test_acc, test_loss, time_now])})
        loss_min, acc_max = min(loss_min, test_loss), max(acc_max, test_acc)
        print_result(list(rsl[-1].values()), save_path)

    print_last_result([loss_min, acc_max], save_path)

    return loss_min, acc_max


def train(device, optimizer, model, train_data, loss_func):
    train_acc, train_loss, n_train = 0, 0, 0
    model.train()
    bar = tqdm(desc="Training", total=len(train_data), leave=False)
    for data, target in train_data:
        data, target = data.to(device), target.to(device)
        y = model(data)
        loss = loss_func(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += accuracy(y, target)
        train_loss += loss.item() * target.size(0)
        n_train += target.size(0)

        bar.set_description("Loss: {0:.3f}, Accuracy: {1:.3f}".format(train_loss / n_train, float(train_acc) / n_train))
        bar.update()
    bar.close()

    return float(train_acc) / n_train, train_loss / n_train


def test(device, optimizer, model, test_data, loss_func):
    test_acc, test_loss, n_test = 0, 0, 0
    bar = tqdm(desc="Testing", total=len(test_data), leave=False)

    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        y = model(data)
        loss = loss_func(y, target)
        test_acc += accuracy(y, target)
        test_loss += loss.item() * target.size(0)
        n_test += target.size(0)

        bar.update()
    bar.close()

    return float(test_acc) / n_test, test_loss / n_test


def print_last_result(values, save_path):
    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar=" ")
        s = "\n MinTestLoss: {}\n MaxTestAcc: {}".format(*values)
        writer.writerow([s])


def print_result(values, save_path):
    if type(values[0]) == str:
        s = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(*values)
        print(s)
        s = "\t{}\t\t{}\t{}\t\t{}\t\t{}\t\t{}\t\t{}".format(*values)
    else:
        s = "{:.4f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(*values)
        print(s)
        s = "\t{:.4f}\t{}\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(*values)

    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar=" ")
        writer.writerow([s])


def print_resource(is_available, gpu_id, save_path):
    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar=" ")
        s = "Computing on GPU{}\n".format(gpu_id) if is_available else "Not Computing on GPU\n"
        print(s)
        writer.writerow([s])


def print_config(hp_dict, save_path, is_out_of_domain=False):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar=" ")
        s = "### Hyperparameters ###\n"

        for name, value in hp_dict.items():
            s += "{}: {}\n".format(name, value)
        writer.writerow([s])

        if is_out_of_domain:
            s = "Out of Domain\n"
            s += "\nMinTestLoss: {}\nMaxTestAcc: {}".format(1.0e+8, 1.0e+8)
            writer.writerow([s])
