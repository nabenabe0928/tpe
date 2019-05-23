def get_variable_features(model):
    with open("feature/{}/feature.csv".format(model), "r", newline = "") as f:
        reader = dict(csv.DictReader(f, delimieter = ";", quotechar = '"'))
        config_space = CS.ConfigurationSpace()
        
        for row in reader:
            default = eval(row["default"])
            param_name = row["var_name"]
            var_type = row["type"]
            dist = row["dist"]

            if var_type == "cat":
                choices = eval(row["bound"])
                hp = CSH.CategoricalHyperparameter(name = param_name, choices = choices, default_value = default)
            else:
                b = eval(row["bound"])

                if var_type == "int":
                    if dist == "log":
                        hp = CSH.UniformIntegerHyperparameter(name = row["var_name"], lower = b[0], upper = b[1], default_value = default, log = True)
                    else:
                        hp = CSH.UniformIntegerHyperparameter(name = row["var_name"], lower = b[0], upper = b[1], default_value = default, log = False)
                elif var_type == "float":
                    if dist == "log":
                        hp = CSH.UniformFloatHyperparameter(name = row["var_name"], lower = b[0], upper = b[1], default_value = default, log = True)
                    else:
                        hp = CSH.UniformFloatHyperparameter(name = row["var_name"], lower = b[0], upper = b[1], default_value = default, log = False)
            
            config_space.add_hyperparameter(hp)
    
    return config_space
