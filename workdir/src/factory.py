import src.models as models
import src.criterion as criterion

def get_model(config):
    c = config["model"]
    return getattr(models, c["class"])(**c["param"])


def get_criterion(config):
    c = config["criterion"]
    return getattr(criterion, c["class"])(**c["param"])
