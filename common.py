import os
import re
import sys
import json
import random
import torch
import tempfile
import networkx as nx
from loguru import logger
from lean_dojo import Pos
import pytorch_lightning as pl
from dataclasses import dataclass, field
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from transformers import get_cosine_schedule_with_warmup
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from typing import Optional, List, Dict, Any, Tuple, Generator
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

Example = Dict[str, Any]
Batch = Dict[str, Any]

MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"


def remove_marks(s: str) -> str:
    """Remove all :code:`<a>` and :code:`</a>` from ``s``."""
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")


@dataclass(unsafe_hash=True)
class Context:
    """Contexts are "queries" in our retrieval setup."""

    path: str
    theorem_full_name: str
    theorem_pos: Pos = field(compare=False)
    state: str

    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.theorem_full_name, str)
        assert isinstance(self.theorem_pos, Pos)
        assert (
                isinstance(self.state, str)
                and "⊢" in self.state
                and MARK_START_SYMBOL not in self.state
                and MARK_END_SYMBOL not in self.state
        )

    def serialize(self) -> str:
        """Serialize the context into a string for Transformers."""
        return self.state


@dataclass(unsafe_hash=True)
class Premise:
    """Premises are "documents" in our retrieval setup."""

    path: str
    """The ``*.lean`` file this premise comes from.
    """

    full_name: str
    """Fully qualified name.
    """

    start: Pos = field(repr=False)
    """Start position of the premise's definition in the ``*.lean`` file.
    """

    end: Pos = field(repr=False, compare=False)
    """End position of the premise's definition in the ``*.lean`` file.
    """

    code: str = field(compare=False)
    """Raw, human-written code for defining the premise.
    """

    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.full_name, str)
        assert (
                isinstance(self.start, Pos)
                and isinstance(self.end, Pos)
                and self.start <= self.end
        )
        assert isinstance(self.code, str) and self.code != ""

    def serialize(self) -> str:
        """Serialize the premise into a string for Transformers."""
        annot_full_name = f"{MARK_START_SYMBOL}{self.full_name}{MARK_END_SYMBOL}"  # 添加<a>标志
        code = self.code.replace(f"_root_.{self.full_name}", annot_full_name)
        fields = self.full_name.split(".")

        for i in range(len(fields)):
            prefix = ".".join(fields[i:])
            new_code = re.sub(f"(?<=\s)«?{prefix}»?", annot_full_name, code)
            if new_code != code:
                code = new_code
                break

        return code


class PremiseSet:
    """A set of premises indexed by their paths and full names."""

    path2premises: Dict[str, Dict[str, Premise]]

    def __init__(self) -> None:
        self.path2premises = {}

    def __iter__(self) -> Generator[Premise, None, None]:
        for _, premises in self.path2premises.items():
            for p in premises.values():
                yield p

    def add(self, p: Premise) -> None:
        if p.path in self.path2premises:
            self.path2premises[p.path][p.full_name] = p
        else:
            self.path2premises[p.path] = {p.full_name: p}

    def update(self, premises: List[Premise]) -> None:
        for p in premises:
            self.add(p)

    def __contains__(self, p: Premise) -> bool:
        return (
                p.path in self.path2premises and p.full_name in self.path2premises[p.path]
        )

    def __len__(self) -> int:
        return sum(len(premises) for premises in self.path2premises.values())


@dataclass(frozen=True)
class File:
    """A file defines 0 or multiple premises."""

    path: str
    """Path of the ``*.lean`` file.
    """

    premises: List[Premise]
    """A list of premises defined in this file.
    """

    @classmethod
    def from_data(cls, file_data: Dict[str, Any]) -> "File":
        """Construct a :class:`File` object from ``file_data``."""
        path = file_data["path"]
        premises = []
        for p in file_data["premises"]:
            full_name = p["full_name"]
            if full_name is None:
                continue
            if "user__.n" in full_name or p["code"] == "":
                # Ignore ill-formed premises (often due to errors in ASTs).
                continue
            if full_name.startswith("[") and full_name.endswith("]"):
                # Ignore mutual definitions.
                continue
            premises.append(
                Premise(
                    path, p["full_name"], Pos(*p["start"]), Pos(*p["end"]), p["code"]
                )
            )
        return cls(path, premises)

    @property
    def is_empty(self) -> bool:
        """Check whether the file contains no premise."""
        return self.premises == []


class Corpus:
    """Our retrieval corpus is a DAG of files. Each file consists of
    premises (theorems, definitoins, etc.) that can be retrieved.
    """

    transitive_dep_graph: nx.DiGraph
    """Transitive closure of the dependency graph among files. 
    There is an edge from file X to Y iff X import Y (directly or indirectly).
    """

    all_premises: List[Premise]
    """All premises in the entire corpus.
    """

    def __init__(self, jsonl_path: str) -> None:
        """Construct a :class:`Corpus` object from a ``corpus.jsonl`` data file."""
        dep_graph = nx.DiGraph()
        self.all_premises = []

        logger.info(f"Building the corpus from {jsonl_path}")

        for line in open(jsonl_path):
            file_data = json.loads(line)
            path = file_data["path"]
            assert not dep_graph.has_node(path)
            file = File.from_data(file_data)

            dep_graph.add_node(path, file=file)
            self.all_premises.extend(file.premises)

            for p in file_data["imports"]:
                assert dep_graph.has_node(p)
                dep_graph.add_edge(path, p)

        assert nx.is_directed_acyclic_graph(dep_graph)
        self.transitive_dep_graph = nx.transitive_closure_dag(dep_graph)

        self.imported_premises_cache = {}
        self.fill_cache()

    def _get_file(self, path: str) -> File:
        return self.transitive_dep_graph.nodes[path]["file"]

    def __len__(self) -> int:
        return len(self.all_premises)

    def __contains__(self, path: str) -> bool:
        return path in self.transitive_dep_graph

    def __getitem__(self, idx: int) -> Premise:
        return self.all_premises[idx]

    @property
    def files(self) -> List[File]:
        return [self._get_file(p) for p in self.transitive_dep_graph.nodes]

    @property
    def num_files(self) -> int:
        return len(self.files)

    def get_dependencies(self, path: str) -> List[str]:
        """Return a list of (direct and indirect) dependencies of the file ``path``."""
        return list(self.transitive_dep_graph.successors(path))

    def get_premises(self, path: str) -> List[Premise]:
        """Return a list of premises defined in the file ``path``."""
        return self._get_file(path).premises

    def num_premises(self, path: str) -> int:
        """Return the number of premises defined in the file ``path``."""
        return len(self.get_premises(path))

    def locate_premise(self, path: str, pos: Pos) -> Optional[Premise]:
        """Return a premise at position ``pos`` in file ``path``.

        Return None if no such premise can be found.
        """
        for p in self.get_premises(path):
            assert p.path == path
            if p.start <= pos <= p.end:
                return p
        return None

    def fill_cache(self) -> None:
        for path in self.transitive_dep_graph.nodes:
            self._get_imported_premises(path)

    def _get_imported_premises(self, path: str) -> List[Premise]:
        """Return a list of premises imported in file ``path``. The result is cached."""
        premises = self.imported_premises_cache.get(path, None)
        if premises is not None:
            return premises

        premises = []
        for p in self.transitive_dep_graph.successors(path):
            premises.extend(self._get_file(p).premises)
        self.imported_premises_cache[path] = premises
        return premises

    def get_accessible_premises(self, path: str, pos: Pos) -> PremiseSet:
        """Return the set of premises accessible at position ``pos`` in file ``path``,
        i.e., all premises defined in the (transitively) imported files or earlier in the same file.
        """
        premises = PremiseSet()
        for p in self.get_premises(path):
            if p.end <= pos:
                premises.add(p)
        premises.update(self._get_imported_premises(path))
        return premises

    def get_accessible_premise_indexes(self, path: str, pos: Pos) -> List[int]:
        return [
            i
            for i, p in enumerate(self.all_premises)
            if (p.path == path and p.end <= pos)
               or self.transitive_dep_graph.has_edge(path, p.path)
        ]

    # 获取最相近的前提
    def get_nearest_premises(
            self,
            premise_embeddings: torch.FloatTensor,  # 查询前提
            batch_context: List[Context],  # 查询前提的上下文，用于限制前提访问
            batch_context_emb: torch.Tensor,  # 从中寻找相似度高的前提
            k: int,  # 前k个相近的
    ) -> Tuple[List[List[Premise]], List[List[float]]]:
        """Perform a batch of nearest neighbour search."""
        similarities = batch_context_emb @ premise_embeddings.t()  # @表示矩阵乘法，premise_embeddings.t()是前提嵌入向量的转置
        idxs_batch = similarities.argsort(dim=1, descending=True).tolist()
        results = [[] for _ in batch_context]
        scores = [[] for _ in batch_context]

        for j, (ctx, idxs) in enumerate(zip(batch_context, idxs_batch)):
            accessible_premises = self.get_accessible_premises(  # 提取当前上下文可访问的前提列表
                ctx.path, ctx.theorem_pos
            )
            for i in idxs:
                p = self.all_premises[i]
                if p in accessible_premises:
                    results[j].append(p)
                    scores[j].append(similarities[j, i].item())
                    if len(results[j]) >= k:
                        break
            else:
                raise ValueError

        return results, scores


@dataclass(frozen=True)
class IndexedCorpus:
    """A corpus with premise embeddings."""

    corpus: Corpus
    embeddings: torch.FloatTensor

    def __post_init__(self):
        assert self.embeddings.device == torch.device("cpu")
        assert len(self.embeddings) == len(self.corpus)


def get_all_pos_premises(annot_tac, corpus: Corpus) -> List[Premise]:
    """Return a list of all premises that are used in the tactic ``annot_tac``."""
    _, provenances = annot_tac
    all_pos_premises = set()

    for prov in provenances:
        def_path = prov["def_path"]
        p = corpus.locate_premise(def_path, Pos(*prov["def_pos"]))
        if p is not None:
            all_pos_premises.add(p)
        else:
            logger.warning(f"Cannot locate premise: {prov}")

    return list(all_pos_premises)


_SPACES_REGEX = re.compile(r"\s+", re.DOTALL)


def normalize_spaces(s: str) -> str:
    """Repalce any consecutive block of whitespace characters in ``s`` with a single whitespace."""
    return _SPACES_REGEX.sub(" ", s).strip()


def format_tactic(annot_tac: str, provenances, normalize: bool) -> str:
    """Use full names for the all <a>...</a>."""
    if normalize:
        annot_tac = normalize_spaces(annot_tac)
    if len(provenances) == 0:
        return annot_tac

    tac = ""
    marks = list(re.finditer(r"<a>(?P<ident>.+?)</a>", annot_tac))

    for i, (m, prov) in enumerate(zip_strict(marks, provenances)):
        last_end = marks[i - 1].end() if i > 0 else 0
        tac += annot_tac[last_end: m.start()] + "<a>" + prov["full_name"] + "</a>"

    tac += annot_tac[marks[-1].end():]
    return tac


def format_state(s: str) -> str:
    m = re.match(r"\d+ goals", s)
    if m is not None:
        return s[m.end():].strip()
    else:
        return s


def format_augmented_state(
        s: str, premises: List[Premise], max_len: int, p_drop: float
) -> str:
    """Format a state with retrieved premises and drop some of them with probability ``p_drop``."""
    s = format_state(s)

    aug_s = ""
    length = 0
    max_premises_len = max_len - len(bytes(s.encode("utf-8")))

    for p in premises:
        if random.random() < p_drop:
            continue
        p_str = f"{p.serialize()}\n\n"
        l = len(bytes(p_str.encode("utf-8")))
        if length + l > max_premises_len:
            continue
        length += l
        aug_s = p_str + aug_s

    aug_s += s
    return aug_s

# 用于创建一个优化器
def get_optimizers(
        parameters, trainer: pl.Trainer, lr: float, warmup_steps: int
) -> Dict[str, Any]:
    """
    Return an AdamW optimizer with cosine warmup learning rate schedule.

    DeepSpeedCPUAdam: 这是一种专为Microsoft DeepSpeed库设计的优化器。DeepSpeed是一个用于深度学习优化的库，它提供了一系列提升大规模训练性能的工具。D
    eepSpeedCPUAdam优化器是为了在CPU上进行模型参数的优化，并且在使用Zero Redundancy Optimizer（零冗余优化器）时，它支持内存优化，可以将部分参数或优化状态卸载到CPU来节省GPU内存。

    FusedAdam: 同样是DeepSpeed库的一部分，FusedAdam是一个高性能的优化器，它将某些计算步骤融合起来，以减少计算时间和提高效率。
    这种融合操作通常是在GPU上执行的，利用更高效的内存访问模式和计算操作。

    AdamW: 这是一个标准的优化器，它在Adam算法的基础上加入了权重衰减。
    AdamW优化器更加注重正则化，有助于防止过拟合。这种优化器不针对特定的硬件进行优化，而是可以在大多数标准的训练场景中使用。

    """
    strategy = trainer.strategy

    if isinstance(strategy, DeepSpeedStrategy):  # 检查策略类型是否为DeepSpeedStrategy
        # 如果配置中包含'offload_optimizer'选项，则使用DeepSpeedCPUAdam优化器
        if "offload_optimizer" in strategy.config["zero_optimization"]:
            logger.info("Optimizing with DeepSpeedCPUAdam")
            optimizer = DeepSpeedCPUAdam(parameters, lr=lr, adamw_mode=True)
        else:
            logger.info("Optimizing with FusedAdam")
            optimizer = FusedAdam(parameters, lr=lr, adam_w_mode=True)
    else:
        logger.info("Optimizing with AdamW")
        optimizer = torch.optim.AdamW(parameters, lr=lr)

    # 计算最大步数，可以是直接设置的max_steps，如果没有设置，则根据最大训练轮数、训练数据加载器的长度和梯度累积批次数动态计算
    if trainer.max_steps != -1:
        max_steps = trainer.max_steps
    else:
        assert trainer.max_epochs is not None
        max_steps = (
                trainer.max_epochs
                * len(trainer.datamodule.train_dataloader())
                // trainer.accumulate_grad_batches
        )

    # 创建学习率调度器，使用余弦预热调度
    """
    学习率调度器，使用余弦预热调度意味着：

    预热（Warmup）: 在训练开始的若干步内，学习率会从0或一个较小的值逐渐增加到初始设定的学习率。
    这个过程叫做预热，目的是在训练初期阶段慢慢提升学习率，避免模型在开始训练时由于过高学习率导致的不稳定。
    
    余弦调度（Cosine Scheduling）: 学习率在预热后会按照余弦函数的形状逐渐下降。
    具体来说，当预热结束后，学习率会开始以余弦曲线的形式缓慢降低到接近零的值，这个过程会持续到训练结束。这种方法可以在训练的后期阶段使学习率平滑地减小，有助于模型收敛到更好的性能。
    """
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }


def _is_deepspeed_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileExistsError(f"Checkpoint {path} does not exist.")
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "zero_to_fp32.py"))


def load_checkpoint(model_cls, ckpt_path: str, device, freeze: bool):
    """Handle DeepSpeed checkpoints in model loading."""
    if not _is_deepspeed_checkpoint(ckpt_path):
        model = model_cls.load_from_checkpoint(ckpt_path, strict=False).to(device)
    else:
        with tempfile.TemporaryDirectory() as dirname:
            path = os.path.join(dirname, "lightning.cpkt")
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, path)
            model = model_cls.load_from_checkpoint(path, strict=False)
            model = model.to(device)
    if freeze:
        model.freeze()
    return model


#
def zip_strict(*args):
    assert len(args) > 1 and all(len(args[0]) == len(a) for a in args[1:])  # 保证掩码的长度和ID一致
    return zip(*args)


def set_logger(verbose: bool) -> None:
    """
    Set the logging level of loguru.
    The effect of this function is global, and it should
    be called only once in the main function
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


# 检查是否配置了cpu检查点
def cpu_checkpointing_enabled(pl_module) -> bool:
    """
    DeepSpeed 的 CPU 检查点
    CPU检查点功能是一种内存优化技术，
    它可以在训练过程中将中间层的激活值临时存储到CPU内存中，而不是GPU内存。
    当这些激活值在反向传播中需要时，再将其重新加载到GPU中。
    这样可以显著减少GPU内存的使用，因为GPU内存通常比CPU内存更小、更昂贵，并且对于大型模型来说是一种稀缺资源。
    """

    try:
        trainer = pl_module.trainer
        return (
                trainer.strategy is not None
                and isinstance(trainer.strategy, DeepSpeedStrategy)
                and trainer.strategy.config["activation_checkpointing"]["cpu_checkpointing"]  # 配置以确定是否启用了 cpu_checkpointing
        )
    except RuntimeError:
        return False
