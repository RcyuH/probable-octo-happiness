"""Tree-augmented sampler from the original PROS implementation.

This module ports the algorithmic behavior from
``PROS/original_pros/verl/experimental/dataset/tree_engine.py`` without ray or
verl-specific ``DataProto`` dependencies.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ProsTreeConfig:
    selector: str = "entropy"
    sampler: str = "pg"
    mu0: float = 0.0
    tau0: float = 1.5
    sigma0: float = 0.0
    delta: float = 0.1
    gamma: float = 0.995
    gibbs_sweeps: int = 5
    min_window_tokens: int = 1000
    score_threshold: float = 1.0
    allow_fallback_fill: bool = True
    random_seed: int = 42


@dataclass
class ProsTreeNode:
    item: int
    father_item: Optional[int] = None
    partial_rollout: Optional[List[int]] = None
    step_num: int = 0

    def __post_init__(self) -> None:
        self.children_items: List[int] = []
        if self.step_num > 0 and (self.partial_rollout is None or self.father_item is None):
            raise ValueError("Non-root tree nodes require a father and partial rollout.")

    @property
    def partial_rollout_len(self) -> int:
        return len(self.partial_rollout or [])

    def depth(self, item2node: Dict[int, "ProsTreeNode"]) -> int:
        if self.step_num <= 0:
            return 0
        return item2node[self.father_item].depth(item2node) + 1  # type: ignore[index]

    def original_ancestor(self, item2node: Dict[int, "ProsTreeNode"]) -> int:
        if self.step_num <= 0:
            return self.item
        return item2node[self.father_item].original_ancestor(item2node)  # type: ignore[index]


@dataclass
class ProsRolloutRecord:
    item: int
    response_ids: List[int]
    reward: float
    response_mask: List[int]
    partial_rollout_len: int
    entropies: List[float]
    values: Optional[List[float]] = None

    @property
    def binary_score(self) -> int:
        return int(self.reward)

    @property
    def response_len(self) -> int:
        return int(np.sum(self.response_mask))


class ProsTreeEngine:
    """PROS tree engine with the PG posterior selector from the reference code."""

    def __init__(self, original_data_len: int, config: ProsTreeConfig):
        if original_data_len <= 0:
            raise ValueError("original_data_len must be positive")
        self.config = config
        self.original_data_len = original_data_len
        self.rng = np.random.default_rng(config.random_seed)

        self.root = ProsTreeNode(item=-1, father_item=None, step_num=-1)
        self.item2node: Dict[int, ProsTreeNode] = {-1: self.root}
        self.next_item = 0
        self.parent_selection_counts = [0] * original_data_len
        for item in range(original_data_len):
            node = ProsTreeNode(item=item, father_item=-1, step_num=0)
            self.root.children_items.append(item)
            self.item2node[item] = node
            self.next_item += 1

        self.mu0 = float(config.mu0)
        self.tau0 = float(config.tau0)
        self.tau0_2 = self.tau0**2
        self.sigma0 = float(config.sigma0) if config.sigma0 and config.sigma0 > 0 else None
        self.delta = float(config.delta)
        self.gamma = float(config.gamma)
        self.gibbs_sweeps = int(max(1, config.gibbs_sweeps))

        self.psi = self.rng.normal(loc=self.mu0, scale=self.tau0, size=original_data_len)
        self.variance = np.ones(original_data_len) * self.tau0_2
        self.s = np.zeros(original_data_len)
        self.n = np.zeros(original_data_len)
        self.last_touch = np.zeros(original_data_len)
        self.father_last_touch = np.zeros(original_data_len)
        self.select_num = np.zeros(original_data_len)
        self.father_select_num = np.zeros(original_data_len)

    def __len__(self) -> int:
        return self.next_item

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "item2node": self.item2node,
            "next_item": self.next_item,
            "parent_selection_counts": self.parent_selection_counts,
            "psi": self.psi,
            "variance": self.variance,
            "s": self.s,
            "n": self.n,
            "last_touch": self.last_touch,
            "father_last_touch": self.father_last_touch,
            "select_num": self.select_num,
            "father_select_num": self.father_select_num,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.item2node = state["item2node"]
        self.next_item = state["next_item"]
        self.root = self.item2node[-1]
        self.parent_selection_counts = state.get("parent_selection_counts", [0] * self.original_data_len)
        for key in ["psi", "variance", "s", "n", "last_touch", "father_last_touch", "select_num", "father_select_num"]:
            setattr(self, key, state[key])

    def get_node(self, item: int) -> ProsTreeNode:
        return self.item2node[item]

    def get_original_ancestor_item(self, item: int) -> int:
        return self.item2node[item].original_ancestor(self.item2node)

    def get_children_items(self, item: int) -> List[int]:
        return self.item2node[item].children_items

    def create_new_node(self, father_node: ProsTreeNode, partial_rollout: List[int], step_num: int, score: float) -> int:
        new_item = self.next_item
        node = ProsTreeNode(
            item=new_item,
            father_item=father_node.item,
            partial_rollout=list(partial_rollout),
            step_num=step_num,
        )
        self.item2node[new_item] = node
        father_node.children_items.append(new_item)
        self.next_item += 1

        father_item = father_node.item
        father_psi = self.psi[father_item]
        if self.sigma0 is None:
            father_p = 1.0 / (1.0 + np.exp(-father_psi))
            p_low = max(0.01, father_p - self.delta)
            p_high = min(0.99, father_p + self.delta)
            psi_low = np.log(p_low / (1.0 - p_low))
            psi_high = np.log(p_high / (1.0 - p_high))
            sigma_low = (father_psi - psi_low) / 1.96
            sigma_high = (psi_high - father_psi) / 1.96
            final_sigma = max(max(sigma_low, sigma_high), 0.02)
        else:
            final_sigma = self.sigma0

        cur_psi = self.rng.normal(loc=father_psi, scale=final_sigma)
        self.psi = np.append(self.psi, cur_psi)
        self.s = np.append(self.s, 0.0)
        self.n = np.append(self.n, 0.0)
        self.variance = np.append(self.variance, final_sigma**2)
        self.last_touch = np.append(self.last_touch, step_num)
        self.select_num = np.append(self.select_num, 0.0)
        self.father_last_touch[int(father_item)] = step_num
        return new_item

    def update_data_source(self, records: Sequence[ProsRolloutRecord], step_num: int) -> Dict[str, float]:
        groups: Dict[int, List[ProsRolloutRecord]] = defaultdict(list)
        for record in records:
            groups[int(record.item)].append(record)

        partial_lens: List[int] = []
        partial_ratios: List[float] = []
        for item, group_records in groups.items():
            metrics = self._batch_create_nodes(item=item, records=group_records, step_num=step_num)
            partial_lens.extend(metrics["partial_lens"])
            partial_ratios.extend(metrics["partial_ratios"])

        partial_arr = np.asarray(partial_lens, dtype=np.float64)
        ratio_arr = np.asarray(partial_ratios, dtype=np.float64)
        return {
            "dataset/num_nodes": float(self.next_item),
            "dataset/partial_rollout_len_mean": float(partial_arr.mean()) if partial_arr.size else 0.0,
            "dataset/partial_rollout_len_std": float(partial_arr.std()) if partial_arr.size else 0.0,
            "dataset/partial_rollout_len_max": float(partial_arr.max()) if partial_arr.size else 0.0,
            "dataset/partial_rollout_len_min": float(partial_arr.min()) if partial_arr.size else 0.0,
            "dataset/partial_rollout_len_ratio_mean": float(ratio_arr.mean()) if ratio_arr.size else 0.0,
            "dataset/partial_rollout_len_ratio_std": float(ratio_arr.std()) if ratio_arr.size else 0.0,
            "dataset/partial_rollout_zero_ratio": float(np.mean(partial_arr == 0)) if partial_arr.size else 0.0,
        }

    def _batch_create_nodes(self, item: int, records: Sequence[ProsRolloutRecord], step_num: int) -> Dict[str, List[float]]:
        father_node = self.item2node[item]
        if father_node.depth(self.item2node) > 0:
            return {"partial_lens": [], "partial_ratios": []}

        candidates = []
        for row_id, record in enumerate(records):
            score = 1 if record.reward >= self.config.score_threshold else 0
            response_len = record.response_len
            start = max(record.partial_rollout_len, int(math.floor(response_len * 0.25)))
            end = int(math.floor(response_len * 0.75))
            valid_len = end - start
            if score == 0 or valid_len <= self.config.min_window_tokens:
                continue
            entropies = np.asarray(record.entropies[: len(record.response_ids)], dtype=np.float64)
            values = np.asarray(record.values[: len(record.response_ids)], dtype=np.float64) if record.values is not None else None
            if entropies.size < end:
                continue
            candidates.append((row_id, record, start, end, entropies, values))

        if not candidates:
            return {"partial_lens": [], "partial_ratios": []}

        best: Optional[Tuple[float, ProsRolloutRecord, int]] = None
        if self.config.selector == "entropy":
            for _, record, start, end, entropies, _ in candidates:
                local = entropies[start:end]
                if local.size == 0:
                    continue
                col_id = start + int(np.argmax(local))
                score = float(entropies[col_id])
                if best is None or score > best[0]:
                    best = (score, record, col_id)
        elif self.config.selector == "mix":
            for _, record, start, end, entropies, values in candidates:
                if values is None or values.size < end:
                    continue
                window_values = values[start:end]
                if window_values.size == 0:
                    continue
                threshold = np.partition(window_values, max(int(window_values.size * 0.8) - 1, 0))[max(int(window_values.size * 0.8) - 1, 0)]
                valid_positions = np.where(window_values > threshold)[0]
                if valid_positions.size == 0:
                    continue
                local_entropy = entropies[start:end][valid_positions]
                selected = start + int(valid_positions[int(np.argmax(local_entropy))])
                score = float(entropies[selected])
                if best is None or score > best[0]:
                    best = (score, record, selected)
        else:
            raise ValueError(f"Invalid tree selector: {self.config.selector}")

        if best is None:
            return {"partial_lens": [], "partial_ratios": []}

        _, record, partial_rollout_len = best
        partial_rollout = record.response_ids[:partial_rollout_len]
        self.create_new_node(father_node, partial_rollout, step_num, record.reward)
        response_len = max(record.response_len, 1)
        return {"partial_lens": [float(partial_rollout_len)], "partial_ratios": [float(partial_rollout_len / response_len)]}

    def update_posterior(self, item_lst: Iterable[int], reward_lst: Iterable[float], step_num: int) -> Dict[str, float]:
        items = np.asarray(list(item_lst), dtype=np.int64)
        rewards = np.asarray(list(reward_lst), dtype=np.float64)
        if items.size == 0:
            return {}

        binary_rewards = (rewards >= self.config.score_threshold).astype(np.float64)
        self.s *= self.gamma
        self.n *= self.gamma
        for item, reward in zip(items, binary_rewards):
            self.s[item] += reward
            self.n[item] += 1.0
        self.last_touch[items] = step_num
        for item in items:
            parent = self.get_original_ancestor_item(int(item))
            self.father_last_touch[parent] = step_num

        parent_items = list(range(self.original_data_len))
        for _ in range(self.gibbs_sweeps):
            self._gibbs_one_sweep_selected(parent_items)

        metrics: Dict[str, float] = {}
        grouped: Dict[int, List[float]] = defaultdict(list)
        for item, reward in zip(items, binary_rewards):
            grouped[int(item)].append(float(reward))
        if len(grouped) > 1:
            accs = np.asarray([np.mean(v) for v in grouped.values()])
            thetas = np.asarray([self._sigmoid(self.psi[item]) for item in grouped])
            if accs.std() > 0 and thetas.std() > 0:
                corr = float(np.corrcoef(accs, thetas)[0, 1])
            else:
                corr = 0.0
            metrics["sampler/pg_correlation"] = corr
            metrics["sampler/pg_error"] = float(np.mean(np.abs(accs - thetas)))
        return metrics

    def _sample_pg(self, b: float, c: float, trunc: int = 200) -> float:
        if b <= 0:
            return 0.0
        c_term = (c * c) / (4.0 * math.pi * math.pi)
        ks = np.arange(1, trunc + 1, dtype=np.float64) - 0.5
        gammas = self.rng.gamma(shape=b, scale=1.0, size=trunc)
        return float(np.sum(gammas / (ks * ks + c_term)) / (2.0 * math.pi * math.pi))

    def _gibbs_one_sweep_selected(self, parent_items: Sequence[int]) -> None:
        inv_tau02 = 1.0 / self.tau0_2
        sum_inv_sigma2: Dict[int, float] = {}
        sum_inv_sigma2_w_psi: Dict[int, float] = {}

        for parent in parent_items:
            sum_inv = 0.0
            sum_weighted = 0.0
            for child in self.get_children_items(parent):
                n_child = float(self.n[child])
                s_child = float(self.s[child])
                kappa = s_child - n_child / 2.0
                omega = self._sample_pg(n_child, float(self.psi[child]), self._pg_trunc())
                inv_sigma2 = 1.0 / float(self.variance[child])
                variance = 1.0 / (inv_sigma2 + omega)
                mean = variance * (inv_sigma2 * self.psi[parent] + kappa)
                self.psi[child] = self.rng.normal(loc=mean, scale=math.sqrt(max(variance, 1e-12)))
                sum_inv += inv_sigma2
                sum_weighted += inv_sigma2 * self.psi[child]
            sum_inv_sigma2[parent] = sum_inv
            sum_inv_sigma2_w_psi[parent] = sum_weighted

        for parent in parent_items:
            n_parent = float(self.n[parent])
            s_parent = float(self.s[parent])
            kappa = s_parent - n_parent / 2.0
            omega = self._sample_pg(n_parent, float(self.psi[parent]), self._pg_trunc()) if n_parent > 0 else 0.0
            variance = 1.0 / (inv_tau02 + sum_inv_sigma2[parent] + omega)
            mean = variance * (inv_tau02 * self.mu0 + sum_inv_sigma2_w_psi[parent] + kappa)
            self.psi[parent] = self.rng.normal(loc=mean, scale=math.sqrt(max(variance, 1e-12)))
            self.variance[parent] = variance

    def _pg_trunc(self) -> int:
        return 200

    def select_batch(self, batch_size: int, step_num: int) -> Tuple[List[int], Dict[str, Any]]:
        thetas = self._sigmoid(self.psi)
        diverse_threshold = 3
        while diverse_threshold > 0:
            if int((step_num - self.father_last_touch > diverse_threshold).sum()) < batch_size:
                diverse_threshold -= 1
            else:
                break

        ids = np.argsort(np.abs(thetas - 0.5))
        batch: List[int] = []
        parent_set = set()
        for idx in ids:
            parent = self.get_original_ancestor_item(int(idx))
            if parent in parent_set:
                continue
            if step_num - self.father_last_touch[parent] < diverse_threshold:
                continue
            parent_set.add(parent)
            batch.append(int(idx))
            self.select_num[idx] += 1
            self.father_select_num[parent] += 1
            if len(batch) == batch_size:
                break

        fallback_fill = 0
        if len(batch) < batch_size and self.config.allow_fallback_fill:
            for idx in range(self.original_data_len):
                if idx in batch:
                    continue
                batch.append(idx)
                fallback_fill += 1
                if len(batch) == batch_size:
                    break
        if len(batch) != batch_size:
            raise ValueError(f"Only {len(batch)} items collected for batch size {batch_size}")

        fixed_ids = np.arange(min(20, self.original_data_len))
        metrics: Dict[str, Any] = {
            "sampler/selected_thetas_mean": float(np.mean(thetas[batch])),
            "sampler/selected_thetas_min": float(np.min(thetas[batch])),
            "sampler/selected_thetas_max": float(np.max(thetas[batch])),
            "sampler/diverse_threshold": float(diverse_threshold),
            "sampler/fallback_fill": float(fallback_fill),
            "sampler/fixed_thetas": [[float(thetas[c]) for c in self.get_children_items(int(p))] for p in fixed_ids],
        }
        return batch, metrics

    def update_and_select(
        self,
        records: Sequence[ProsRolloutRecord],
        step_num: int,
        batch_size: int,
    ) -> Tuple[List[int], Dict[str, Any]]:
        posterior_metrics = self.update_posterior(
            [record.item for record in records],
            [record.reward for record in records],
            step_num,
        )
        data_metrics = self.update_data_source(records, step_num)
        batch, selection_metrics = self.select_batch(batch_size, step_num)
        return batch, {**posterior_metrics, **data_metrics, **selection_metrics}

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
