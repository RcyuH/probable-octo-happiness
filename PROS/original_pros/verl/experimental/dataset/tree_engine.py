from collections import defaultdict
import time
import traceback
import scipy
from dataclasses import dataclass, is_dataclass
import math
import random
from typing import Any, Dict, List, Optional, Sized, Tuple
import numpy as np
import torch
import ray
from omegaconf import DictConfig
from verl.protocol import DataProto


@dataclass
class TreeSpec:
    num_parents: int
    children_per_parent: List[int]

class TreeNode:
    """
    A class representing a node in a tree structure.
    """

    def __init__(self, 
                 item,
                 father_item: Optional[int] = None,
                 partial_rollout: Optional[list[int]] = None,
                 step_num=0):
        """
        Initialize the TreeNode with the given data.
        """
        self.item = item
        self.father_item = father_item
        self.children_items = []
        self.step_num = step_num
        self.partial_rollout = partial_rollout

        assert step_num <= 0 or partial_rollout is not None and father_item is not None
    
    @property
    def partial_rollout_len(self):
        """
        The length of the partial rollout.
        If partial_rollout is None, return 0.
        """
        return len(self.partial_rollout)
    
    def depth(self, item2node):
        """
        The depth of the node in the tree.
        """
        if self.step_num <= 0:
            return 0
        father_node = item2node[self.father_item]
        return father_node.depth(item2node) + 1
    
    def get_original_ancestor_item(self, item2node):
        """
        Get the original ancestor of this node.
        The original ancestor is the root node of the tree.
        """
        if self.step_num <= 0:
            return self.item
        father_node = item2node[self.father_item]
        return father_node.get_original_ancestor_item(item2node)

    def add_child(self, child_item: int):
        """
        Add a child node to this node.
        """
        self.children_items.append(child_item)

    def __repr__(self):
        return f"TreeNode(item={self.item})"



class TreeEngine:
    def __init__(self, original_data_len, data_config):
        self.spec = TreeSpec(num_parents=original_data_len, children_per_parent=[0] * original_data_len)
        self.rng = np.random.default_rng()
        self.tree_config = data_config.tree_data
        self.original_datalength = original_data_len

        # Initialize an empty dataset for new data
        self.root = TreeNode(item=-1, father_item=None, step_num=-1)
        self.item2node = {-1: self.root}
        self.next_item = 0

        self.parent_selection_counts = [0] * original_data_len

        for i in range(self.original_datalength):
            node = TreeNode(item=i, father_item=-1, step_num=0)
            self.root.add_child(i)
            self.item2node[i] = node
            self.next_item += 1

    
    def __len__(self):
        return self.next_item
    
    def get_node(self, item: int) -> TreeNode:
        """
        Get the node of the given item.
        """
        return self.item2node.get(item, None)
    
    def get_original_ancestor_item(self, item: int) -> int:
        """
        Get the original ancestor item of the given item.
        """
        node = self.item2node[item]
        return node.get_original_ancestor_item(self.item2node)
    
    def get_children_items(self, item: int) -> List[int]:
        """
        Get the children items of the given item.
        """
        node = self.item2node[item]
        return node.children_items
    
    
    def state_dict(self):
        """
        Return the state dict of the dataset.
        """
        return {
            "tree_config": self.tree_config,
            "item2node": self.item2node,
            "next_item": self.next_item,
            "parent_selection_counts": self.parent_selection_counts,
        }

    def load_state_dict(self, state_dict):
        """
        Load the state dict of the dataset.
        """
        assert self.tree_config == state_dict["tree_config"]
        self.item2node = state_dict["item2node"]
        self.next_item = state_dict["next_item"]
        self.root = self.item2node[-1]
        if "parent_selection_counts" in state_dict:
            self.parent_selection_counts = state_dict["parent_selection_counts"]
        else:
            self.parent_selection_counts = [0] * self.original_datalength
    
    def create_new_node(self, father_node: TreeNode, partial_rollout: List[int], step_num: int, score: float) -> None:
        """
        Create a new node with the given father node and partial rollout.
        """
        new_item = self.next_item
        self.item2node[new_item] = TreeNode(
            item=new_item,
            father_item=father_node.item,
            partial_rollout=partial_rollout,
            step_num=step_num
        )
        father_node.add_child(new_item)
        self.next_item += 1

        self.spec.children_per_parent[father_node.item] += 1

    def update_data_source(self, batch, step_num: int) -> Dict[str, float]:
        items = torch.as_tensor(batch.non_tensor_batch["item"].astype(np.int32)).to(torch.long)
        all_scores = torch.as_tensor(batch.non_tensor_batch["score"]).to(torch.long)
        assert torch.all((all_scores == 0) | (all_scores == 1)), \
            "Currently only support score in {0, 1}."

        all_partial_rollout_len = torch.as_tensor(
            batch.non_tensor_batch["partial_rollout_len"].astype(np.int32)
        ).to(torch.long)

        all_response_mask = batch.batch["response_mask"].to(torch.bool)
        all_response_len = all_response_mask.sum(dim=-1).to(torch.long)

        all_responses = batch.batch["responses"]
        all_values    = batch.batch.get("values", None)
        all_entropys  = batch.batch["entropys"]

        bsz = items.numel()
        assert all_responses.size(0) == bsz, "batch dims mismatch"

        groups: Dict[int, List[int]] = {}
        for i in range(bsz):
            key = int(items[i].item())
            groups.setdefault(key, []).append(i)

        all_partial_lens: List[int] = []
        all_partial_ratios: List[float] = []

        for item_val, idx_list in groups.items():
            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=items.device)
            gmetrics = self._batch_create_nodes(
                item=item_val,
                idx=idx_tensor,
                all_scores=all_scores,
                all_partial_rollout_len=all_partial_rollout_len,
                all_response_len=all_response_len,
                all_responses=all_responses,
                all_values=all_values,
                all_entropys=all_entropys,
                step_num=step_num,
            )
            all_partial_lens.extend(gmetrics["partial_lens"])
            all_partial_ratios.extend(gmetrics["partial_ratios"])

        def _safe_mean(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        def _safe_std(xs: List[float]) -> float:
            return float(np.std(xs)) if xs else 0.0

        return {
            "dataset/num_nodes": self.next_item,
            "dataset/partial_rollout_len_mean": np.mean(all_partial_lens),
            "dataset/partial_rollout_len_std": np.std(all_partial_lens),
            "dataset/partial_rollout_len_max": np.max(all_partial_lens) if all_partial_lens else 0,
            "dataset/partial_rollout_len_min": np.min(all_partial_lens) if all_partial_lens else 0,
            "dataset/partial_rollout_len_ratio_mean": np.mean(all_partial_ratios),
            "dataset/partial_rollout_len_ratio_std": np.std(all_partial_ratios),
            "dataset/partial_rollout_zero_ratio": np.mean(np.array(all_partial_lens) == 0),
        }

    def _batch_create_nodes(
        self,
        item: int,
        idx: torch.Tensor,
        all_scores: torch.Tensor,
        all_partial_rollout_len: torch.Tensor,
        all_response_len: torch.Tensor,
        all_responses: torch.Tensor,
        all_values: torch.Tensor,
        all_entropys: torch.Tensor,
        step_num: int,
    ) -> Dict[str, Any]:
        cfg = self.tree_config
        name: str = cfg.name

        father_node = self.item2node[item]

        depth = father_node.depth(self.item2node)
        if depth > 0:
            return {"partial_lens": [], "partial_ratios": []}
        
        device = all_responses.device
        T = all_responses.size(1)
        m = idx.numel()

        responses_g = all_responses.index_select(0, idx)
        scores_g   = all_scores.index_select(0, idx)
        rlen_g     = all_response_len.index_select(0, idx)
        values_g   = all_values.index_select(0, idx) if all_values is not None else None
        entropies_g= all_entropys.index_select(0, idx)

        valid_row = torch.ones(m, dtype=torch.bool, device=device)
        valid_row &= (scores_g != 0)

        start_g = torch.floor(rlen_g.to(torch.float32) * 0.25).to(torch.long)
        start_g = torch.maximum(all_partial_rollout_len.index_select(0, idx), start_g)
        end_g = torch.floor(rlen_g.to(torch.float32) * 0.75).to(torch.long)
        valid_len = end_g - start_g
        valid_row &= (valid_len > 1000)

        arange_T = torch.arange(T, device=device).view(1, T)
        start_exp = start_g.view(m, 1)
        end_exp   = end_g.view(m, 1)
        mask_win = (arange_T >= start_exp) & (arange_T < end_exp)
        mask_win &= valid_row.view(m, 1)
        num_valid_tokens = mask_win.sum().item()

        if num_valid_tokens == 0:
            return {"partial_lens": [], "partial_ratios": []}

        mv = torch.where(mask_win, values_g, -float('inf')) if values_g is not None else None
        me = torch.where(mask_win, entropies_g, -float('inf'))

        if name == "entropy":
            pos = me.argmax()
            row_id, col_id = torch.unravel_index(pos, me.shape)
        elif name == "mix":
            flattened_value = torch.masked_select(mv, mask_win)
            percentile_value = torch.kthvalue(flattened_value, int(num_valid_tokens * 0.8))[0]
            masked_valid_values = torch.where(mv > percentile_value, me, -float('inf'))
            pos = masked_valid_values.argmax()
            row_id, col_id = torch.unravel_index(pos, masked_valid_values.shape)
        else:
            raise ValueError(f"Invalid tree config name: {name}")
        
        partial_rollout_len = (col_id).item()
        partial_rollout_ratio = partial_rollout_len / rlen_g[row_id].item()
        partial_rollout = responses_g[row_id, :partial_rollout_len].tolist()

        self.create_new_node(father_node, partial_rollout, step_num, scores_g[row_id].item())

        return {"partial_lens": [partial_rollout_len], "partial_ratios": [partial_rollout_ratio]}

    def update_posterior(self, item_lst: List[int], reward_lst: List[float], step_num: int):
        return {}

    def select_batch(self, batch_size: int, step_num: int) -> Tuple[List[int], Dict[str, float]]:
        raise NotImplementedError

    
    def async_wrap_all(self, batch: DataProto, step_num: int, bsz: int):
        posterior_merics = self.update_posterior(batch.non_tensor_batch["item"].tolist(), batch.non_tensor_batch["score"].tolist(), step_num)
        data_metrics = self.update_data_source(batch, step_num)
        batch, selection_metrics = self.select_batch(bsz, step_num)
        metrics = {
            **posterior_merics,
            **data_metrics,
            **selection_metrics,
        }
        return batch, metrics


@ray.remote
class PGTreeEngine(TreeEngine):
    def __init__(self, original_data_len, data_config):
        super().__init__(original_data_len, data_config)

        # Fixed parameters
        self.mu0 = float(data_config.sampler.tree_sampler.mu0)
        self.tau0 = float(data_config.sampler.tree_sampler.tau0)
        self.sigma0 = float(data_config.sampler.tree_sampler.sigma0) if data_config.sampler.tree_sampler.sigma0 is not None else None
        self.delta = data_config.sampler.tree_sampler.delta
        self.gamma = data_config.sampler.tree_sampler.gamma
        self.gibbs_sweeps = int(max(1, data_config.sampler.tree_sampler.gibbs_sweeps))
        self.rng = np.random.default_rng()

        self.tau0_2 = self.tau0 ** 2

        # State variables
        self.psi = self.rng.normal(loc=self.mu0, scale=self.tau0, size=self.original_datalength)
        self.variance = np.ones(self.original_datalength) * self.tau0_2
        self.s = np.zeros(self.original_datalength)
        self.n = np.zeros(self.original_datalength)
        self.last_touch = np.zeros(self.original_datalength)
        self.father_last_touch = np.zeros(self.original_datalength)
        self.select_num = np.zeros(self.original_datalength)
        self.father_select_num = np.zeros(self.original_datalength)

        self._pg_engine = None
        self._init_pg_engine(data_config.train_batch_size)

    def _init_pg_engine(self, batch_size: int, force: Optional[str] = None):
        from polyagamma import random_polyagamma
        self._pg_engine = ("polyagamma", random_polyagamma)

    def sample_pg(self, b: float, c: float, rng: Optional[np.random.Generator] = None, trunc: int = 200) -> float:
        if b <= 0:
            return 0.0
        kind, eng = self._pg_engine
        try:
            if kind == "pypolyagamma":
                pg = eng
                return float(pg.pgdraw(b, c))
            elif kind == "polyagamma":
                fn = eng
                return float(fn(b, c, random_state=rng)) if rng is not None else float(fn(b, c))
        except Exception:
            print(f"!!! Raise error when{b=}, {c=}")
            traceback.print_exc()
            exit(-1)



    def create_new_node(self, father_node: TreeNode, partial_rollout: List[int], step_num: int, score: float) -> None:
        super().create_new_node(father_node, partial_rollout, step_num, score)

        father_item = father_node.item
        father_psi = self.psi[father_item]
        father_variance = self.variance[father_item]
        if self.sigma0 is None:
            father_p = 1 / (1 + np.exp(-father_psi))
            p_low = max(0.01, father_p - self.delta)
            p_high = min(0.99, father_p + self.delta)
            psi_low = np.log(p_low / (1 - p_low))
            psi_high = np.log(p_high / (1 - p_high))
            sigma_low = (father_psi - psi_low) / 1.96
            sigma_high = (psi_high - father_psi) / 1.96
            sigma = max(sigma_low, sigma_high)

            final_sigma = max(sigma, 0.02)
            print("[PG Engine] Create new node: father_item={}, father_psi={}, father_variance={}, final_sigma={}, final_cliped_sigma={}".format(father_item, father_psi, father_variance, sigma, final_sigma))
        else:
            final_sigma = self.sigma0
            print("[PG Engine] Create new node: father_item={}, father_psi={}, father_variance={}, final_sigma={}".format(father_item, father_psi, father_variance, final_sigma))
        

        cur_psi = self.rng.normal(loc=father_psi, scale=final_sigma)
        self.psi = np.append(self.psi, cur_psi)
        self.s = np.append(self.s, 0.0)
        self.n = np.append(self.n, 0.0)
        self.variance = np.append(self.variance, final_sigma ** 2)
        self.last_touch = np.append(self.last_touch, step_num)
        self.select_num = np.append(self.select_num, 0)
        self.father_last_touch[int(father_item)] = step_num
        self.spec.children_per_parent[father_item] += 1
    
    def update_posterior(self, item_lst: List[int], reward_lst: List[float], step_num: int):
        metrics = {}
        items = np.array(item_lst)
        rewards = np.array(reward_lst)

        discount = self.gamma
        self.s *= discount
        self.n *= discount
        item2rewardlst = defaultdict(list)
        for item, reward in zip(items, rewards):
            self.s[item] += reward
            self.n[item] += 1
            item2rewardlst[item].append(reward)
        item2acc = {item: np.mean(reward_lst).item() for item, reward_lst in item2rewardlst.items()}
        item2theta = {item: 1 / (1 + np.exp(-self.psi[item])).item() for item in item2rewardlst.keys()}
        if len(item2acc) > 1:
            accs = np.array(list(item2acc.values()))
            thetas = np.array(list(item2theta.values()))
            r, pvalue = scipy.stats.pearsonr(accs, thetas)
            error = np.mean(np.abs(accs - thetas))
            metrics.update({
                "sampler/pg_correlation": r,
                "sampler/pg_pvalue": pvalue,
                "sampler/pg_error": error,
            })
            print("[PG Engine] Step {}: correlation={}, error={}".format(step_num, r, error))


        self.last_touch[items] = step_num
        for item in items:
            father_item = self.get_original_ancestor_item(item)
            self.father_last_touch[int(father_item)] = step_num

        start_time = time.time()
        parent_items = list(range(self.original_datalength))
        for _ in range(self.gibbs_sweeps):
            self._gibbs_one_sweep_selected(parent_items)
        end_time = time.time()
        print("[PG Engine] Gibbs one sweep selected time: {}".format(end_time - start_time))
        return metrics

    def _gibbs_one_sweep_selected(self, p_lst):
        inv_tau02 = 1.0 / self.tau0_2

        # Leaves
        sum_inv_sigma2_lst = dict()
        sum_inv_sigma2_w_psi_lst = dict()
        for p in p_lst:
            sum_inv_sigma2 = 0
            sum_inv_sigma2_w_psi = 0
            for j in self.get_children_items(p):
                n_ = float(self.n[j])
                s_ = float(self.s[j])
                kappa = s_ - n_ / 2.0
                psi_cur = float(self.psi[j])
                omega = self.sample_pg(b=n_, c=psi_cur, rng=self.rng)
                inv_sigma2 = 1.0 / self.variance[j]
                V = 1.0 / (inv_sigma2 + omega)
                m = V * (inv_sigma2 * self.psi[p] + kappa)
                self.psi[j] = self.rng.normal(loc=m, scale=math.sqrt(V))
                sum_inv_sigma2 += inv_sigma2
                sum_inv_sigma2_w_psi += inv_sigma2 * self.psi[j]
            sum_inv_sigma2_lst[p] = sum_inv_sigma2
            sum_inv_sigma2_w_psi_lst[p] = sum_inv_sigma2_w_psi

        # Roots
        for p in p_lst:
            n_ = float(self.n[p])
            s_ = float(self.s[p])
            kappa = s_ - n_ / 2.0
            psi_cur = float(self.psi[p])
            omega = self.sample_pg(b=n_, c=psi_cur, rng=self.rng) if n_ > 0 else 0.0

            V0 = 1.0 / (inv_tau02 + sum_inv_sigma2_lst[p] + omega)
            m0 = V0 * (inv_tau02 * self.mu0 + sum_inv_sigma2_w_psi_lst[p] + kappa)

            self.psi[p] = self.rng.normal(loc=m0, scale=math.sqrt(V0)) 
            self.variance[p] = V0

    def select_batch(self, batch_size: int, step_num: int) -> Tuple[List[int], Dict[str, float]]:
        thetas = 1 / (1 + np.exp(-self.psi))
        diverse_threshold = 3
        while diverse_threshold > 0:
            if (step_num - self.father_last_touch > diverse_threshold).sum() < batch_size:
                diverse_threshold -= 1
            else:
                break

        error = np.abs(thetas - 0.5)
        ids = np.argsort(error)
        batch = []
        parent_set = set()
        for idx in ids:
            parent = self.get_original_ancestor_item(idx)
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
            
        else:
            raise ValueError(f"Only {len(batch)} is collected")

        metrics = {}
        seed = 42
        rng = np.random.default_rng(seed)
        random_parent_ids = rng.choice(list(range(self.original_datalength)), size=20, replace=False)
        fixed_thetas = []
        for p in random_parent_ids:
            children_thetas = []
            for j in self.get_children_items(p):
                children_thetas.append(thetas[j].item())
            fixed_thetas.append(children_thetas)
        metrics.update({
            "sampler/fixed_thetas": fixed_thetas,
        })

        metrics.update({
            "sampler/thetas": thetas.tolist(),
            "sampler/father_thetas": thetas[:self.original_datalength].tolist(),
            "sampler/selected_thetas": thetas[batch].tolist(),
        })

        return batch, metrics

    
