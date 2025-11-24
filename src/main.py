import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

# import run program
from run import run as run
from meta_train_run import run as meta_train_run
from meta_test_run import run as meta_test_run
from odis_train_run import run as odis_train_run
from updet_m_train_run import run as updet_m_train_run
from updet_l_train_run import run as updet_l_train_run
from meta_distral_train_run import run as xdistral_train_run
from original_distral_run import run as original_distral_run


# 将sacred的日志输出到文件中，而不是输出到控制台
SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds  # 过滤掉日志中的回车和退格符号

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")  # 日志保存路径


@ex.main  # 标记这个函数为主要运行函数,Sacred会调用这个函数来运行实验
# 设置随机数种子,并根据配置中的meta_train和meta_test字段来调用不同的运行函数
def my_main(_run, _config, _log):
    config = config_copy(_config)  # 深拷贝配置字典_config,避免修改外部传入的配置
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])  # th是pytorch的别名。th.manual_seed会在pytorch的随机数生成器中设置一个种子。
    config['env_args']['seed'] = config["seed"]
    if config.get("meta_train", False):  # 检查配置中是否有meta_train字段,如果没有则默认为False
        if config.get("name") == "xtrans_train":
            meta_train_run(_run, config, _log)  # 如果meta_train=True,则调用meta_train_run函数运行meta训练逻辑
        elif config.get("name") == "updet_m_train":
            updet_m_train_run(_run, config, _log)
        elif config.get("name") == "updet_l_train":
            updet_l_train_run(_run, config, _log)
        elif config.get("name") == "odis_train":
            odis_train_run(_run, config, _log)
        elif config.get("name") == "xdistral_train":
            xdistral_train_run(_run, config, _log)
        elif config.get("name") == "original_distral":
            original_distral_run(_run, config, _log)
        elif config.get("name") == "xdistral_weight_train":
            xdistral_train_run(_run, config, _log)
    elif config.get("meta_test", False):  # 检查配置中是否有meta_test字段,如果没有则默认为False
        meta_test_run(_run, config, _log)  # 如果meta_test=True,则调用meta_test_run函数运行meta测试逻辑

    else:
        run(_run, config, _log)  # 如果meta_train和meta_test都不是True,则调用默认的run函数运行常规逻辑


"""
params: 函数接收的参数列表,包含命令行传入的参数
arg_name: 需要检查的参数名,如"--config"
subfolder: 配置文件所在的子文件夹,如"algs"
"""


def _get_config(params, arg_name, subfolder):  # 从命令行参数params中获取指定名称arg_name的配置文件,加载并返回配置字典。
    config_name = None  # 初始化配置文件名为空
    for _i, _v in enumerate(params):  # 遍历命令行参数列表params
        if _v.split("=")[0] == arg_name:  # 如果当前参数项的名称是arg_name
            config_name = _v.split("=")[1]  # 则提取参数值作为配置文件名config_name
            # 并从params中删除这个已经读取的参数
            del params[_i]
            break
    # 如果config_name不为空,加载配置文件
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.safe_load(f)  # 读取并解析YAML文件为配置字典
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def _get_env_config(env_name):  # 加载指定名称env_name的环境配置文件,并返回配置字典
    with open(os.path.join(os.path.dirname(__file__), "config", "envs", "{}.yaml".format(env_name)), "r") as f:
        try:
            config_dict = yaml.safe_load(f)  # 读取并解析YAML文件为配置字典
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(env_name, exc)
    return config_dict


def recursive_dict_update(d, u):  # 递归地更新字典d,使用字典u的数据
    if u is not None:  # 只有当 u 不是 None 时才继续
        for k, v in u.items():
            if isinstance(v, collections.Mapping):  # 如果v是一个映射对象，比如字典
                d[k] = recursive_dict_update(d.get(k, {}), v)  # 递归调用recursive_dict_update,更新键k对应的嵌套字典
            else:
                d[k] = v
    return d


"""
config_copy函数的作用是深拷贝配置字典config,避免修改外部传入的配置。
它的代码逻辑是:
    判断config是字典还是列表
    如果是字典:
        对字典的每一项key-value递归调用config_copy,得到拷贝后的value
        使用拷贝后的key-value构造一个新的字典并返回
    如果是列表:
        对列表的每一项递归调用config_copy,得到拷贝后的item
        使用拷贝后的item构造一个新的列表并返回
        否则,直接深拷贝config并返回
"""


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)  # 深拷贝命令行参数列表,避免修改外部传入的参数
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm base configs
    alg_config = _get_config(params, "--config", "algs")  # 从命令行参数params中获取指定名称"--config"的配置文件,加载并返回配置字典。
    # config_dict = {**config_dict, **alg_config}
    config_dict = recursive_dict_update(config_dict, alg_config)  # 递归地更新字典config_dict,使用字典alg_config的数据

    # check whether is meta_train
    meta_train = config_dict.get("meta_train", False)  # 从配置字典config_dict中获取meta_train字段,如果没有则默认为False
    meta_test = config_dict.get("meta_test", False)  # 从配置字典config_dict中获取meta_test字段,如果没有则默认为False

    if meta_train:
        task_config = _get_config(params, "--task-config",
                                  "tasks")  # 从命令行参数params中获取指定名称"--task-config"的配置文件,加载并返回配置字典。
        config_dict = recursive_dict_update(config_dict, task_config)  # 递归地更新字典config_dict,使用字典task_config的数据
        env_name = config_dict["env"]
        env_config = _get_env_config(env_name)  # 加载指定名称env_name的环境配置文件,并返回配置字典
        config_dict = recursive_dict_update(config_dict, env_config)
    elif meta_test:
        task_config = _get_config(params, "--task-config", "tasks")
        config_dict = recursive_dict_update(config_dict, task_config)
        env_name = config_dict["env"]
        env_config = _get_env_config(env_name)
        config_dict = recursive_dict_update(config_dict, env_config)
        if env_name == "sc2":
            config_dict["env_args"]["map_name"] = config_dict[
                "test_task"]  # 如果是sc2环境,则将配置字典config_dict中的map_name字段设置为test_task字段
        else:
            config_dict["env_args"]["task_id"] = config_dict[
                "test_task"]  # 否则,将配置字典config_dict中的task_id字段设置为test_task字段
    else:
        # 如果既非meta_train也非meta_test，直接加载env_config,递归合并
        env_config = _get_config(params, "--env-config", "envs")  #
        config_dict = recursive_dict_update(config_dict, env_config)


    # get config from argv, such as "remark"
    def _get_argv_config(params):  # 从命令行参数params中提取参数配置
        config = {}  # 初始化config字典用于保存提取的配置
        to_del = []  # 初始化to_del列表,用于记录需要删除的参数
        for _i, _v in enumerate(params):
            item = _v.split("=")[0]  # 获取参数名称
            if item[:2] == "--" and item not in ["envs", "algs"]:  # 如果参数名以"--"开头,且不是"envs"和"algs"
                config_v = _v.split("=")[1]
                try:
                    config_v = eval(config_v)  # 将字符串config_v转换为字典或列表等对象
                except:
                    pass
                config[item[2:]] = config_v  # 将参数名去掉"--"后作为键,参数值作为值,添加到config字典中
                to_del.append(_v)  # 将参数_v添加到to_del列表中
        for _v in to_del:
            params.remove(_v)  # 从params中删除to_del列表中的参数
        return config


    config_dict = recursive_dict_update(config_dict, _get_argv_config(params))  # 将从命令行参数params中提取的参数配置,递归合并到全局配置

    # if set map_name, we overwrite it
    if "map_name" in config_dict:
        assert not meta_train and meta_test, "Unexpected scenario!!!"  # 断言meta_train和meta_test都不为True
        config_dict["env_args"]["map_name"] = config_dict["map_name"]

    if "task_id" in config_dict:
        assert not meta_train and meta_test, "Unexpected scenario!!!"  # 断言meta_train和meta_test都不为True
        config_dict["env_args"]["task_id"] = config_dict["task_id"]

    # 单任务测试特典
    single_trans = config_dict.get("trans_phase", False)  # 从配置字典config_dict中获取trans_phase字段,如果没有则默认为False

    if single_trans:
        config_dict['epsilon_start'] = 0.05

        # now add all the config to sacred
    ex.add_config(config_dict)  # 将配置字典config_dict添加到sacred实验中

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")  # 日志保存路径
    ex.observers.append(FileStorageObserver.create(file_obs_path)) # 将日志保存到指定路径
    ex.run_commandline(params) # 运行实验
