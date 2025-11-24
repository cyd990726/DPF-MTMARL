import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import json

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)  # 对参数进行合理性检查
    args = SN(**_config)  # 将配置字典_config转换成一个简化的命名空间(SimpleNamespace)对象args。这里的args是一个命名空间对象，可以通过args.xxx的方式访问其中的属性
    args.device = "cuda" if args.use_cuda else "cpu"
    logger = Logger(_log)
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")
    remark_str = getattr(args, "remark", "nop")  # 从args中获取remark参数,如果不存在则设置为默认值"nop"
    unique_token = "{}__{}_{}".format(args.name, remark_str, datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"))  # 构造一个独一无二的token,作为该次实验运行的唯一标识
    args.unique_token = unique_token  # 将该token添加到args中

    # 在训练模式下设置TensorBoard的日志目录,并保存实验配置文件
    if args.use_tensorboard and not args.evaluate:  # 判断是否同时满足使用TensorBoard(args.use_tensorboard)和训练模式(not args.evaluate)
        # 根据不同环境设置子目录
        if args.env == "sc2":
            tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", args.env,
                                         args.env_args["map_name"], args.name)
        elif args.env == "mpe":
            tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", args.env,
                                         args.env_args["scenario_name"], args.name)
        else:
            tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", args.env, args.name)
        # 如果日志目录不存在则创建目录
        if not os.path.exists(tb_logs_direc):
            os.makedirs(tb_logs_direc)
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)  # 构造本次运行的日志目录tb_exp_direc,包含唯一token
        logger.setup_tb(tb_exp_direc)  # 设置TensorBoard的日志目录
        config_str = json.dumps(vars(args), indent=4)  # 将args参数转换为JSON格式
        # 保存到配置文件config.json中
        with open(os.path.join(tb_exp_direc, "config.json"), "w") as f:
            f.write(config_str)

    # 获取唯一的输出目录名称并设置Sacred的输出文件
    if args.env == "sc2":
        output_dirname = os.path.join(dirname(dirname(abspath(__file__))), "outputs", args.env,
                                      args.env_args["map_name"], args.name)
    elif args.env == "mpe":
        output_dirname = os.path.join(dirname(dirname(abspath(__file__))), "outputs", args.env,
                                      args.env_args["scenario_name"], args.name)
    else:
        output_dirname = os.path.join(dirname(dirname(abspath(__file__))), "outputs", args.env, args.name)
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)
    args.output_dir = os.path.join(output_dirname, unique_token)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file = os.path.join(output_dirname, f"{unique_token}.out")
    logger.setup_sacred(_run, output_file)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    runner = r_REGISTRY[args.runner](args=args, logger=logger)  # 初始化runner,获取环境信息env_info
    env_info = runner.get_env_info()  # 获取环境信息
    args.n_agents = env_info["n_agents"]  # 智能体数量
    args.n_actions = env_info["n_actions"]  # 动作数量
    args.state_shape = env_info["state_shape"]  # 状态表示形状
    # scheme(状态表示)
    scheme = {
        "state": {"vshape": env_info["state_shape"]},  # 环境状态
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},  # 智能体私有观测
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},  # 智能体动作
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},  # 智能体可用动作
        "reward": {"vshape": (1,)},  # 奖励
        "terminated": {"vshape": (1,), "dtype": th.uint8},  # 终止标志
    }
    # groups(智能体分组)
    groups = {
        "agents": args.n_agents
    }
    # preprocess定义对动作的onehot编码处理
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 初始化经验回放池buffer
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # 加载预训练模型
    if args.checkpoint_path != "":  # 检查是否有传入检查点目录的参数,如果检查点目录参数不为空,则进行加载检查点的操作
        timesteps = [] # 初始化一个列表,用于存储找到的模型检查点的timestep
        timestep_to_load = 0 # 初始化将要加载的模型的timestep为0，默认情况下会加载最新的模型检查点

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return


        for name in os.listdir(args.checkpoint_path): # 遍历检查点目录下的所有文件
            full_name = os.path.join(args.checkpoint_path, name)  # 构造每个文件的完整路径full_name
            if os.path.isdir(full_name) and name.isdigit(): # 判断full_name是否为目录且目录名是否为数字
                timesteps.append(int(name)) # 如果同时满足是目录和数字命名,说明这是存放模型的目录,将目录名(即模型的timestep)添加到timesteps列表中

        # 如果load_step为0,选择timesteps中最大的step,即加载最新模型
        if args.load_step == 0:
            timestep_to_load = max(timesteps)
        else:
            # 否则找到最接近load_step的timestep
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))  # 构造目标模型的完整路径
        logger.console_logger.info("Loading model from {}".format(model_path)) # 打印加载模型的路径
        learner.load_models(model_path) # 调用learner的加载方法,加载模型文件
        # 如果设置了断点训练,将runner的timestep设置为加载的模型的step
        if getattr(args, "breakpoint_train", False):
            runner.t_env = timestep_to_load
        # 如果是评估模式,进行评估然后返回
        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # 训练迭代开始
    episode = 0  # 当前训练的回合数
    last_test_T = -args.test_interval - 1  # 上次测试迭代的计数器,开始值先设很小,可以立即进行第一次测试
    last_log_T = 0  # 上次日志输出的迭代计数器
    model_save_time = 0  # 上次保存模型的迭代计数器

    start_time = time.time()  # 训练开始时间
    last_time = start_time  # 上次打印时间统计的时间点

    # 打印将要进行训练的迭代总数
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # 正常的MARL算法不需要预训练阶段
    pretrain_phase = getattr(args, "pretrain", False)

    while runner.t_env <= args.t_max:  # 主循环,当迭代数小于最大值时

        # 生成一个完整episode的数据
        episode_batch = runner.run(test_mode=False, pretrain_phase=pretrain_phase)

        # 将生成的数据插入经验池
        buffer.insert_episode_batch(episode_batch)

        # 如果经验池可以样本则进行训练
        if buffer.can_sample(args.batch_size):

            terminated = False # 初始化终止标志为False

            # 遍历runner的批大小次数进行多次训练
            for _run in range(runner.batch_size):

                # 从经验池中采样一个批次的数据
                episode_sample = buffer.sample(args.batch_size)

                # 截取样本中实际填充的训练步数
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                # 检查样本设备是否为模型设备
                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                # 进行一次模型训练
                terminated = learner.train(episode_sample, runner.t_env, episode)

                # 如果模型训练结束,提前结束循环
                if terminated:
                    break

            if terminated:
                # 如果只进行表示学习,打印提示并结束训练
                if getattr(args, "only_repre_learning", False):
                    logger.console_logger.info("Only task repre learning!")
                    break

                # 否则表示预训练结束,打印提示开始正式训练
                else:
                    logger.console_logger.info("Finish pretrain and begin training for {} timesteps".format(args.t_max))
                    # 重置一些训练相关变量,不需要修改episode和last_log_T
                    pretrain_phase = False # 预训练阶段结束
                    start_time = time.time() # 重置训练开始时间
                    last_time = start_time # 重置上次打印时间统计的时间点
                    # 重置buffer和runner中的一些属性
                    buffer.clear()
                    runner.t_env = 0
                    continue  # 继续训练循环

        # 仅在正式强化学习阶段执行测试
        if not pretrain_phase:
            # 计算需要运行的测试次数
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            # 判断是否到达测试间隔
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                # 打印当前训练进度
                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                # 打印预计剩余时间和已用时间
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time)))
                # 更新最后测试时间
                last_time = time.time()
                last_test_T = runner.t_env # 更新最后测试迭代计数器
                # 进行多次测试评估
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                local_results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
                if args.env == "sc2":
                    save_path = os.path.join(local_results_path, args.env, args.env_args["map_name"], args.name,
                                             "models", args.unique_token, str(runner.t_env))
                elif args.env == "mpe":
                    save_path = os.path.join(local_results_path, args.env, args.env_args["scenario_name"], args.name,
                                             "models", args.unique_token, str(runner.t_env))
                else:
                    save_path = os.path.join(local_results_path, args.env, args.name, "models", args.unique_token,
                                             str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)

            episode += args.batch_size_run

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

    # close environment
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):  # 对参数进行合理性检查
    # 检查use_cuda参数:若设置为True,但没有CUDA可用,则自动切换为False
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")
    # 检查test_nepisode参数:如果小于batch_size_run,则设置为batch_size_run，否则设置为batch_size_run的整数倍
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]
    return config
