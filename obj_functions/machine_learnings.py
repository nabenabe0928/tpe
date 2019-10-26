import obj_functions.machine_learning_utils as ml_utils
from obj_functions import models, datasets


def train(model, hp_dict, train_data, test_data, cuda_id, save_path):
    print(hp_dict)
    loss_min, acc_max = ml_utils.start_train(model, train_data, test_data, cuda_id, save_path)
    return {"error": 1. - acc_max, "cross_entropy": loss_min}


def cnn(experimental_settings):
    train_dataset, test_dataset = datasets.get_dataset(dataset_name=experimental_settings["dataset_name"],
                                                       n_cls=experimental_settings["n_cls"],
                                                       image_size=experimental_settings["image_size"],
                                                       data_frac=experimental_settings["data_frac"],
                                                       biased_cls=experimental_settings["biased_cls"]
                                                       )

    def _imp(hp_dict, cuda_id, save_path):
        model = models.CNN(**hp_dict, n_cls=experimental_settings["n_cls"])
        train_data, test_data = datasets.get_data(train_dataset, test_dataset, batch_size=model.batch_size)
        return train(model, hp_dict, train_data, test_data, cuda_id, save_path)

    return _imp
