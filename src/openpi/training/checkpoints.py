import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol
import glob # 新增导入
import os # 新增导入
import shutil # 新增导入

from etils import epath
import jax
import orbax.checkpoint as ocp

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            # async_options=ocp.AsyncOptions(timeout_secs=7200),
            # 关键修改：尝试禁用 Orbax 的主要异步保存机制
            async_options=None, # 设置为 None 通常会禁用异步特性或回退到同步
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    # 在调用 manager.save() 之前，尝试清理潜在的冲突临时文件
    try:
        # checkpoint_manager.directory 应该指向实验的根目录
        # 例如: /path/to/checkpoints/config_name/exp_name
        checkpoint_base_dir = checkpoint_manager.directory
        
        # Orbax 创建的临时文件/目录通常以 {step}.orbax-checkpoint-tmp-{uuid} 或类似模式命名
        # 它们与最终的 {step}/ 目录是同级的
        temp_entity_pattern = str(checkpoint_base_dir / f"{step}.orbax-checkpoint-tmp-*")
        
        logging.info(f"Proactively cleaning up temporary checkpoint entities matching: {temp_entity_pattern}")
        
        for entity_path_str in glob.glob(temp_entity_pattern):
            entity_path = epath.Path(entity_path_str)
            logging.warning(f"Found existing temporary checkpoint entity: {entity_path}. Attempting to remove.")
            if entity_path.is_dir():
                shutil.rmtree(entity_path)
                logging.info(f"Removed temporary directory: {entity_path}")
            elif entity_path.is_file():
                os.remove(entity_path)
                logging.info(f"Removed temporary file: {entity_path}")
            else:
                # 处理损坏的符号链接等情况，或者仅记录日志
                logging.warning(f"Found {entity_path} but it is neither a file nor a directory. Skipping removal.")
                
    except Exception as e:
        logging.error(f"Error during proactive cleanup of temporary checkpoint entities for step {step}: {e}", exc_info=True)

    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


# 修改 CallbackHandler 使其同步
# class CallbackHandler(ocp.AsyncCheckpointHandler): # 不再继承 AsyncCheckpointHandler
class CallbackHandler(ocp.CheckpointHandler): # 继承基础的 CheckpointHandler
    """A CheckpointHandler for calling an arbitrary function. Only for saving, not for restoring."""

    # def __init__(self): # 如果不再使用 executor，可以移除
    #     self._executor = futures.ThreadPoolExecutor(max_workers=1)

    # def close(self): # 如果不再使用 executor，可以移除
    #     self._executor.shutdown()

    def save(self, directory: epath.Path, args: "CallbackSave"):
        # 直接在主线程中执行回调
        if jax.process_index() == 0: # 确保只在主进程执行
            logging.info(f"CallbackHandler: Synchronously calling callback for directory: {directory}")
            args.callback(directory)
            logging.info(f"CallbackHandler: Synchronous callback finished for directory: {directory}")

    # async def async_save(self, directory: epath.Path, args: "CallbackSave") -> list[futures.Future]: # 移除 async_save
    #     return [self._executor.submit(self.save, directory, args)]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")
    
    # 如果 CheckpointHandler 基类需要 finalize 或 commit 方法，可能需要实现它们（通常是空操作或简单返回）
    # 查阅 Orbax 文档或 CheckpointHandler 的定义
    def commit(self, directory: epath.Path, *args, **kwargs):
        """Commits the save. No-op for this synchronous handler unless specific commit logic is needed."""
        logging.info(f"CallbackHandler: Commit called for directory {directory}. No-op.")
        pass

    def finalize(self, directory: epath.Path):
        """Finalizes the save. No-op for this handler."""
        logging.info(f"CallbackHandler: Finalize called for directory {directory}. No-op.")
        pass


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
