import yaml
import pickle


def save_params(model, file_name):
    # save model
    pickle.dump(model, open(file_name, "wb"))
    return file_name


def save_hyper_params_to_yaml(data: dict, filename: str):
    """
    Save a dictionary to a YAML file.

    Parameters
    ----------
    data : dict
        The data to save.
    filename : str
        The output YAML file path.
    """
    with open(filename, "w") as file:
        yaml.dump(data, file)
    return filename
