REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .multi_task import EpisodeRunner as MultiTaskEpisodeRunner
REGISTRY["mt_episode"] = MultiTaskEpisodeRunner

from .multi_task import ParallelRunner as MultiTaskParallelRunner
REGISTRY["mt_parallel"] = MultiTaskParallelRunner