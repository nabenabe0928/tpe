import subprocess as sp
import sys
import os
from optimize import optimize

class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'    


def main(config_space, model = None, num = None, n_parallels = None, n_jobs = None, rerun = None):
    
    if model is None or num is None or n_parallels is None or rerun is None:
        print("")
        print("###### ERROR ######")
        models = [file.split(".")[0] for file in os.listdir( "objective_functions/model" )]
        
        print("YOUR COMMAND MUST BE LIKE AS FOLLOWED:")
        print(pycolor.YELLOW + "python ####.py -num 0 -parallel 2 -jobs 10 -re 0" + pycolor.END)
        #print(pycolor.YELLOW + "python main.py -model CNN -num 0 -parallel 1 -itr 10 -cuda 0 1 -re 0" + pycolor.END)
        print("")
        print(pycolor.RED + "SET the variables shown below:" + pycolor.END)
        print("model: which model you run:")
        print("\t[", end = "")
        
        for model in models:
            if not "cache" in model:
                print("{}, ".format(model), end = "")
        print("]")
        print("")

        print("     num: how many times this experiment is: ")
        print("\t[0, 1, 2, ...]")
        print("")
        print("paralell: how many resources parallelized: ")
        print("\t[0, 1, 2, ...]")
        print("")
        print("    jobs: how many times evaluating the model: ")
        print("\t[any natural number]")
        print("")
        print("     re: When you want to restart the searching.")
        print("\t0 or 1 : Default is False(= 0)")
        print("\t1: Restart the searching")
        print("\t0: Set up from initilization.")
        print("")
        print("")
        sys.exit()
    
    if not os.path.isdir("evaluation"):
        os.mkdir("evaluation")
    if not os.path.isdir("evaluation/{}".format(model)):
        os.mkdir("evaluation/{}".format(model))
    
    if not os.path.isdir("log"):
        os.mkdir("log")
    if not os.path.isdir("log/{}".format(model)):
        os.mkdir("log/{}".format(model))
    if not os.path.isdir("log/{}/{:0>3}".format(model, num)):
        os.mkdir("log/{}/{:0>3}".format(model, num))

    init_sh = \
        ["#!/bin/bash", \
        "USER=$(whoami)", \
        "CWD=$(dirname $0)", \
        "\n", \
        "rm evaluation/{}/evaluation{:0>3}.csv".format(model, num), \
        "echo $USER:~$CWD$ rm evaluation/{}/evaluation{:0>3}.csv".format(model, num), \
        "rm log/{}/{:0>3}/*".format(model, num), \
        "echo $USER:~$CWD$ rm log/{}/{:0>3}/*".format(model,num), \
        ]

    if not rerun:
        files = ["log/{}/{:0>3}/".format(model, num) + f for f in os.listdir("log/{}/{:0>3}".format(model, num))]
                    
        
        rm_files = []
        script = ""

        for line in init_sh:
            script += line + "\n"
        
        for file in files:
                rm_files.append(file)
        rm_files.append("evaluation/{}/evaluation{:0>3}.csv".format(model, num))

        if len(rm_files) > 0:
            print("")
            print("#### NOTICE ###") 
            print(pycolor.YELLOW + "Model is {} and the trial number is {}".format(model, num) + pycolor.END)
            print(pycolor.RED + "You are going to remove some files." + pycolor.END)
            print("")
            
            for i, file in enumerate(rm_files):
                if i % 3 == 0 and i != 0:
                    print("")    
                print(pycolor.GREEN + "{0:<45}".format(file) + pycolor.END, end = "")

            
            answer = ""
            while not answer in {"y", "n"}:
                print("")
                print("")
                answer = input("Is it okay? [y or n] : ")
        
            if answer == "y":
                with open("init.sh", "w") as f:
                    f.writelines(script)
            else:
                print("Permission Denied.")
                sys.exit()

            print("")
            print("#########################")
            print("# WILL REMOVE THE FILES #")
            print("#########################")
            print("")

            sp.call("chmod +x init.sh", shell = True)
            sp.call("./init.sh", shell = True)
            
            print("")
            print("#########################")
            print("### REMOVED THE FILES ###")
            print("#########################")
            print("")
    else:
        n_ex = os.listdir("log/{}/{:0>3}".format(model, num))
        for del_idx in range(max(0, n_ex - 1), n_ex):
            sp.call("rm {}".format("log/{}/{:0>3}/log{:0>5}.csv".format(model, num, del_idx)), shell = True)

        with open("init.sh", "w") as f:
            f.writelines(script)

    print("")
    print("#########################")
    print("##### RENEW THE ENV #####")
    print("#########################")
    print("")
        
    optimize(model, num, config_space, max_jobs = n_jobs, n_parallels = n_parallels)