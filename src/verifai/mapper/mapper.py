import pandas as pd
import numpy as np


class Mapper:
    def __init__(self, table):
        self.table = table
        self.functions = []

    def defineTransformFunctions(self, functions):
        self.functions.append(functions)

    def defineProjectFunctions(self, project):
        self.project = project

    def applyMapping(self):
        mappedTable = self.table.copy()
        for function in self.functions:
            mappedTable[function.modify] = np.zeros((len(mappedTable),))
            for idx, _ in df.iterrows():
                mappedTable.at[idx, function.modify] = function.apply(idx, self.table)

        toPrune = list(set(mappedTable.columns).difference(self.project))
        for prune in toPrune:
            assert prune in mappedTable, 'Attempting to project undefined column in table'
            del mappedTable[prune]

        return mappedTable


class TableFunctions:
    def __init__(self, modify):
        self.modify = modify

    def apply(self, idx, table):
        return table[idx, modify]
