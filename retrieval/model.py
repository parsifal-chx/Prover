"""Ligihtning module for the premise retriever."""
import os
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm
from lean_dojo import Pos
from loguru import logger
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Union
from transformers import T5EncoderModel, AutoTokenizer

from common import (
    Premise,
    Context,
    Corpus,
    get_optimizers,
    load_checkpoint,
    zip_strict,
    cpu_checkpointing_enabled,
)

torch.set_float32_matmul_precision("medium")


# 该模型用于前提检索
# 主要的作用是将前提和状态转化为向量
class PremiseRetriever(pl.LightningModule):
    # 以下四个函数是必须要有的，分别对应
    # __init__：模型初始化
    # forward：模型前向传递过程，主要指val和test，但我也推荐在train中使用，保持代码统一
    # training_step:单次的训练过程
    # configure_optimizers: 优化器定义
    def __init__(
            self,
            model_name: str,
            lr: float,
            warmup_steps: int,
            max_seq_len: int,
            num_retrieved: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_retrieved = num_retrieved
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.embeddings_staled = True  # 嵌入层是否是当前的语料库的向量

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "PremiseRetriever":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    # 加载语料库
    def load_corpus(self, path_or_corpus: Union[str, Corpus]) -> None:
        """Associate the retriever with a corpus."""
        if isinstance(path_or_corpus, Corpus):  # 如果path_or_corpus已经是Corpus的实例
            self.corpus = path_or_corpus
            self.corpus_embeddings = None
            self.embeddings_staled = True
            return

        path = path_or_corpus  # 反之，使用path_or_corpus实例化Corpus
        if path.endswith(".jsonl"):  # A raw corpus without embeddings.
            self.corpus = Corpus(path)
            self.corpus_embeddings = None
            self.embeddings_staled = True
        else:  # A corpus with pre-computed embeddings. 已经被预先转化过的
            indexed_corpus = pickle.load(open(path, "rb"))  # 将文件反序列化为 Python 对象
            self.corpus = indexed_corpus.corpus
            self.corpus_embeddings = indexed_corpus.embeddings
            self.embeddings_staled = False

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.encoder.config.hidden_size

    # 私有方法：编码一个前提或者一段文字
    def _encode(
            self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a premise or a context into a feature vector."""
        if cpu_checkpointing_enabled(self):  # 是否使用 CPU 检查点
            hidden_states = torch.utils.checkpoint.checkpoint(
                # 执行当前类实例的 encoder，并将 input_ids 和 attention_mask 作为参数。
                # 使用检查点是为了节省内存
                # use_reentrant=False 表示该检查点不支持可重入的后向传播，这是 PyTorch 中可以处理复杂计算图的一种后向传播类型。
                self.encoder, input_ids, attention_mask, use_reentrant=False
            )[0]
        else:
            hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,  # return_dict=True，这意味着编码器将返回一个包含输出的字典
            ).last_hidden_state

        # Masked average.
        lens = attention_mask.sum(dim=1)  # 通过沿第一个维度（dim=1）对 attention_mask 求和，来计算序列的长度
        # attention_mask增加维度后的形状为(batch_size, sequence_length, 1)
        # 每个元素相乘不是点乘
        # 相乘后的维度为(batch_size, sequence_length, features)
        # 求和后维度为(batch_size, features)
        features = (hidden_states * attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        # Normalize the feature vector to have unit norm.
        # 规范化，使每个向量都具有单位范数（即向量的长度为 1），这样在算余弦相似度时就不用除向量的模长
        return F.normalize(features, dim=1)

    def forward(
            self,
            context_ids: torch.LongTensor,  #
            context_mask: torch.LongTensor,
            pos_premise_ids: torch.LongTensor,  # 正例
            pos_premise_mask: torch.LongTensor,
            neg_premises_ids: torch.LongTensor,  # 负例
            neg_premises_mask: torch.LongTensor,
            label: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the contrastive loss for premise retrieval."""
        # Encode the query and positive/negative documents.
        context_emb = self._encode(context_ids, context_mask)
        pos_premise_emb = self._encode(pos_premise_ids, pos_premise_mask)
        neg_premise_embs = [
            self._encode(ids, mask)
            for ids, mask in zip_strict(neg_premises_ids, neg_premises_mask)
        ]
        all_premise_embs = torch.cat([pos_premise_emb, *neg_premise_embs], dim=0)

        # Cosine similarities for unit-norm vectors are just inner products.
        # tensor.t() ，用于将一个张量的维度 transpose（转置）
        similarity = torch.mm(context_emb, all_premise_embs.t())
        assert -1 <= similarity.min() <= similarity.max() <= 1
        loss = F.mse_loss(similarity, label)  # 均方误差 (Mean Squared Error, MSE)
        return loss

    ############
    # Training #
    ############

    # 在拟合一开始就调用
    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # 记录了模型的超参数
            logger.info(f"Logging to {self.trainer.log_dir}")  # 记录了一个信息级别的日志，显示日志文件被保存的位置

        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True

    # 每个batch的处理函数
    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        loss = self(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )
        self.log(
            "loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch)
        )
        # 返回值为损失
        # 返回值：
        #    Tensor - 一个损失张量
        #    dict - 一个字典. 但所必须key包含loss
        #    None - 训练将会跳过这个batch
        return loss

    # 在每个批次结束时调用
    def on_train_batch_end(self, outputs, batch, _) -> None:
        """Mark the embeddings as staled after a training batch."""
        self.embeddings_staled = True  # 表示当前的语料库与嵌入层的向量不符合，需要更新

    # 优化器定义，返回一个优化器，或数个优化器，或两个List（优化器，Scheduler）
    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    ##############
    # Validation #
    ##############

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        if not self.embeddings_staled:  # 嵌入层与语料库匹配
            return
        logger.info("Re-indexing the retrieval corpus")

        # corpus_embeddings [all_premises,embedding_size]
        self.corpus_embeddings = torch.zeros(
            len(self.corpus.all_premises),
            self.embedding_size,
            dtype=self.encoder.dtype,
            device=self.device,
        )

        for i in tqdm(range(0, len(self.corpus), batch_size)):
            batch_premises = self.corpus.all_premises[i: i + batch_size]  # 获取当前批次的所有前提
            tokenized_premises = self.tokenizer(  # 使用tokenized转化为向量
                [p.serialize() for p in batch_premises],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.corpus_embeddings[i: i + batch_size] = self._encode(  # 编码
                tokenized_premises.input_ids, tokenized_premises.attention_mask
            )

        self.embeddings_staled = False

    # 在验证开始前，先判断向量与预料库是否匹配
    def on_validation_start(self) -> None:
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)

    # validation_step：单次的验证过程
    # 使用Recall@K and MRR 模型
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate metrics such as Recall@K and MRR."""
        # Retrieval.
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, _ = self.corpus.get_nearest_premises(  # 获取前k个相似的前提
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []  # MRR将记录平均倒数排名的数值
        num_with_premises = 0
        tb = self.logger.experiment  # tb是TensorBoard日志记录器的简写

        for i, (all_pos_premises, premises) in enumerate(
                zip_strict(batch["all_pos_premises"], retrieved_premises)
        ):
            # Only log the first example in the batch. 仅记录批次中的第一个例子
            if i == 0:
                msg_gt = "\n\n".join([p.serialize() for p in all_pos_premises])  # 正面的例子
                msg_retrieved = "\n\n".join(
                    [f"{j}. {p.serialize()}" for j, p in enumerate(premises)]  # 检索到的例子
                )
                TP = len(set(premises).intersection(all_pos_premises))  # TP（检索的正例）是检索到的前提和所有正面前提集合的交集的数量
                if len(all_pos_premises) == 0:  # r是该样本的召回率
                    r = math.nan  # r被设为NaN（非数字）
                else:
                    r = float(TP) / len(all_pos_premises)  # 检索的正例数量除以所有正面前提的数量
                msg = f"Recall@{self.num_retrieved}: {r}\n\nGround truth:\n\n```\n{msg_gt}\n```\n\nRetrieved:\n\n```\n{msg_retrieved}\n```"
                tb.add_text(f"premises_val", msg, self.global_step)

            all_pos_premises = set(all_pos_premises)
            if len(all_pos_premises) == 0:
                continue
            else:
                num_with_premises += 1
            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premises.intersection(premises[: (j + 1)]))
                recall[j].append(float(TP) / len(all_pos_premises))
                # 排名为j的前提是真正的正例，并且是首次找到的匹配，则在MRR中添加该排名的倒数，并将first_match_found设置为True
                if premises[j] in all_pos_premises and not first_match_found:
                    MRR.append(1.0 / (j + 1))
                    first_match_found = True
            if not first_match_found:  # 没有发现任何真正的正例，则在MRR列表中添加0.0
                MRR.append(0.0)

        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            self.log(
                f"Recall@{j + 1}_val",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=num_with_premises,
            )

        self.log(
            "MRR",
            np.mean(MRR),
            on_epoch=True,
            sync_dist=True,
            batch_size=num_with_premises,
        )

    ##############
    # Prediction #
    ##############
    # 在预测开始
    def on_predict_start(self) -> None:
        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)
        self.predict_step_outputs = []

    # 预测步骤
    def predict_step(self, batch: Dict[str, Any], _):
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )

        for (
                url,
                commit,
                file_path,
                full_name,
                start,
                tactic_idx,
                ctx,
                pos_premises,
                premises,
                s,
        ) in zip_strict(
            batch["url"],
            batch["commit"],
            batch["file_path"],
            batch["full_name"],
            batch["start"],
            batch["tactic_idx"],
            batch["context"],
            batch["all_pos_premises"],
            retrieved_premises,
            scores,
        ):
            self.predict_step_outputs.append(
                {
                    "url": url,
                    "commit": commit,
                    "file_path": file_path,
                    "full_name": full_name,
                    "start": start,
                    "tactic_idx": tactic_idx,
                    "context": ctx,
                    "all_pos_premises": pos_premises,
                    "retrieved_premises": premises,
                    "scores": s,
                }
            )

    # 预测结束后
    def on_predict_epoch_end(self) -> None:
        if self.trainer.log_dir is not None:
            path = os.path.join(self.trainer.log_dir, "predictions.pickle")
            with open(path, "wb") as oup:
                pickle.dump(self.predict_step_outputs, oup)
            logger.info(f"Retrieval predictions saved to {path}")

        self.predict_step_outputs.clear()

    # 从语料库中搜寻k个前提，状态和策略前缀
    def retrieve(
            self,
            state: List[str],
            file_name: List[str],
            theorem_full_name: List[str],
            theorem_pos: List[Pos],
            k: int,
    ) -> Tuple[List[Premise], List[float]]:
        """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        self.reindex_corpus(batch_size=32)

        ctx = [
            Context(*_)
            for _ in zip_strict(file_name, theorem_full_name, theorem_pos, state)
        ]
        ctx_tokens = self.tokenizer(
            [_.serialize() for _ in ctx],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        context_emb = self._encode(
            ctx_tokens.input_ids.to(self.device),
            ctx_tokens.attention_mask.to(self.device),
        )

        if self.corpus_embeddings.device != context_emb.device:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.device)
        if self.corpus_embeddings.dtype != context_emb.dtype:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.dtype)

        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            ctx,
            context_emb,
            k,
        )
        return retrieved_premises, scores
