#!/usr/bin/env python3

import numpy as np

class StatsAggregator:
    def __init__(self):
        self._data = dict()

    def add_batch(self, tag, batch):
        if tag not in self._data:
            self._data[tag] = []
        if not isinstance(batch, np.ndarray):
            if hasattr(batch, 'numpy'):
                batch = batch.detach().cpu().numpy()
            elif isinstance(batch, list):
                batch = np.array(batch)
        self._data[tag].append(batch)

    def __getitem__(self, tag):
        return np.concatenate(self._data[tag], 0)

    def __iter__(self):
        return iter(self.tags)

    @property
    def tags(self):
        return self._data.keys()
