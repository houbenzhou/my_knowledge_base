import yaml
from dotmap import DotMap


def get_config_from_yaml(yaml_file, encoding='utf8'):
    """
    Get the config from a yml or yaml file
    :param yaml_file: 文件路径
    :param encoding: encoding default: utf8
    :return: config(namespace) or config(dictionary)
    """
    with open(yaml_file, encoding=encoding) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config_dict)
    return config
