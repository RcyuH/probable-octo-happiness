# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractmethod
from collections import deque
from collections.abc import Sized
from dataclasses import dataclass
from multiprocessing import Process
import pprint
import time
import ray
from regex import F
import torch
from omegaconf import DictConfig
from torch.utils.data import Sampler
import numpy as np
from typing import Deque, List, Dict, Optional
import traceback
from torch.utils.data import RandomSampler, SequentialSampler
from concurrent.futures import ProcessPoolExecutor

from verl import DataProto


class AbstractSampler(Sampler[int]):
    """Abstract interface for custom samplers."""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto, step_num: int) -> None:
        pass

class AbstractBatchSampler(Sampler[List[int]]):
    @abstractmethod
    def __init__(*args, **kargs):
        pass

class AbstractCurriculumBatchSampler(AbstractBatchSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto, step_num: int) -> None:
        pass

class AysncUpdater:
    """
    AysncUpdater is a base class for async updating samplers.
    """
    def __init__(self, *args, **kwargs):
        self.worker = ProcessPoolExecutor(max_workers=1)
        self.update_future = None

    # update the sampler in async way
    def async_update(self, *args, **kwargs) -> None:
        assert self.update_future is None, "update_future is not None"
        self.update_future = self.worker.submit(self._update, *args, **kwargs)
    
    # collect the update result
    def update(self, *args, **kwargs) -> None:
        rtn = self.update_future.result()
        self.update_future = None
        return rtn
    
    @staticmethod
    def _update(*args, **kwargs):
        return None



class TreeSampler(AysncUpdater, AbstractCurriculumBatchSampler):
    def __init__(self, data_source: Sized, data_config: DictConfig):
        AbstractCurriculumBatchSampler.__init__(self, data_source, data_config)
        AysncUpdater.__init__(self)

        self.bsz = data_config.train_batch_size
        self.rng = np.random.default_rng()

        self.engine = data_source.engine
        self.queue: Deque[int] = deque(maxlen=self.bsz)

        self.epsilon = data_config.sampler.tree_sampler.epsilon

        # initalize queue
        first_batch, _ = ray.get(self.engine.select_batch.remote(self.bsz, 0))
        # breakpoint()
        self.queue.extend(first_batch)
    
    def state_dict(self):
        return {
            "queue": list(self.queue),
            "bsz": self.bsz,
            "epsilon": self.epsilon,
            "engine": ray.get(self.engine.state_dict.remote()),
        }

    def load_state_dict(self, state_dict):
        self.queue = deque(state_dict["queue"], maxlen=self.bsz)
        self.bsz = state_dict["bsz"]
        self.epsilon = state_dict["epsilon"]
        ray.get(self.engine.load_state_dict.remote(state_dict["engine"]))

    def __iter__(self):
        while True:
            if len(self.queue) < self.bsz:
                print("[Sampler] Not enough items in queue, drop last, raise an StopIterationError")
                return
            batch = [int(self.queue.popleft()) for _ in range(self.bsz)]
            print(batch)
            print(type(batch[0]))
            yield batch

    def async_update(self, batch: DataProto, step_num: int) -> None:
        """ Some heavy operations are done in async way """


        assert self.update_future is None, "future is not None"
        self.update_future = self.engine.async_wrap_all.remote(batch, step_num, self.bsz)

    def update(self, batch: DataProto, step_num: int) -> None:
        # wait for the result
        start_time = time.time()
        new_batch, metrics = ray.get(self.update_future)
        self.update_future = None
        end_time = time.time()
        print(f"[Sampler] Time taken to select batch: {end_time - start_time} seconds")

        # breakpoint()

        # fill in queue
        assert len(new_batch) == self.bsz
        assert len(self.queue) == 0
        self.queue.extend(new_batch)

        return metrics

class PrioritySampler(AbstractBatchSampler):
    """Priority-based BatchSampler that returns batches of indices."""

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        self.bsz = data_config.train_batch_size
        print("Initializing Priority BatchSampler with batch size:", self.bsz)

        # we always assert drop last
        self.data_item_num = len(data_source)
        self.temporal_decay = data_config.sampler.temporal_decay
        
        # Initialize accuracy tensor with -1 (indicating uninitialized)
        self.index2acc = torch.full((self.data_item_num,), -1.0, dtype=torch.float32)

        # initialize priority queue
        self.queue = deque()
        for i in range(self.data_item_num):
            self.queue.append(i)
    

    def update(self, batch: DataProto, step_num: int) -> None:
        """Update the sampler with the current batch."""
        indices = torch.tensor(batch.non_tensor_batch['item'].astype(np.int32))  # item is the index passed to the dataset.__getitem__
        scores = torch.tensor(batch.non_tensor_batch['score'])

        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)

        counts = torch.bincount(inverse_indices, minlength=len(unique_indices))

        score_sums = torch.bincount(inverse_indices, weights=scores, minlength=len(unique_indices))
        score_complements = counts.to(torch.float32) - score_sums

        # update the accuracy tracking with the new indices and scores using tensor operations
        new_acc = score_sums.float() / counts.float()  # compute new accuracy for each unique index
        
        # Create masks for first-time and existing indices
        first_time_mask = self.index2acc[unique_indices] == -1.0
        existing_mask = ~first_time_mask
        
        # Update first-time indices
        self.index2acc[unique_indices[first_time_mask]] = new_acc[first_time_mask]
        
        # Update existing indices with EMA
        self.index2acc[unique_indices[existing_mask]] = (
            self.temporal_decay * new_acc[existing_mask] + 
            (1 - self.temporal_decay) * self.index2acc[unique_indices[existing_mask]]
        )
        
        self.fill_queue()
        
    def fill_queue(self) -> None:
        """Fill the queue with the highest priority items."""
        k = self.bsz - len(self.queue)
        if k <= 0:
            return

        # Check that all indices have been initialized (no -1 values)
        # assert torch.all(self.index2acc != -1.0), "Should not fill queue before calculate acc for all indices"

        # Use tensor operations directly
        reverse_acc_tensor = 1.0 - self.index2acc
        prob_dist = reverse_acc_tensor / (reverse_acc_tensor.sum() + 1e-8)

        # Sample k indices proportional to their reverse accuracy
        sampled_indices = torch.multinomial(prob_dist, num_samples=k, replacement=False)

        # Add sampled indices to queue
        for idx in sampled_indices:
            self.queue.append(idx.item())

    def __iter__(self):
        """Iterate over the sampler, yielding batches of indices."""
        while True:
            if len(self.queue) < self.bsz:
                print("[Sampler] Current queue size: ", len(self.queue))
                print("[Sampler] Not enough items in queue, filling queue...")
                # print current runtime stack
                traceback.print_stack()
                self.fill_queue()
            # assert len(self.queue) >= self.bsz

            print("[Sampler] Pop queue")
            batch = [self.queue.popleft() for _ in range(self.bsz)]
            print("[Sampler] Current Train Batch: ", batch)
            yield batch

    
    def state_dict(self):
        return {
            "index2acc": self.index2acc,
            "queue": list(self.queue),
            "data_item_num": self.data_item_num,
            "bsz": self.bsz,
            "temporal_decay": self.temporal_decay
        }

    def load_state_dict(self, state):
        self.index2acc = state['index2acc']
        self.queue = deque(state['queue'])
        self.data_item_num = state['data_item_num']
        self.bsz = state['bsz']
        self.temporal_decay = state['temporal_decay']