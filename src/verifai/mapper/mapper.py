import pandas as pd


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
            assert function.mod in self.table.columns, 'Attemping to modify undefined column in table'
            # create new column instead
            for idx, _ in df.iterrows():
                mappedTable.at[idx, function.mod] = function.apply(idx, self.table)

        for project in self.project:
            assert project in self.table, 'Attempting to project undefined column in table'
            del mappedTable[project]

        return mappedTable


class TableFunctions:

    def apply(self, idx, table):
        return table[idx, 'colMod']


class SumColumns(TableFunctions):
    def __init__(self, modify):
        super(SumColumns, self).__init__()
        self.mod = modify

    def apply(self, idx, table):
        return table.at[idx, 'x'] + table.at[idx, 'y']


class SumPreviousRow(TableFunctions):
    def __init__(self, modify):
        super(SumPreviousRow, self).__init__()
        self.mod = modify

    def apply(self, idx, table):
        if idx > 0:
            return table.at[idx, 'x'] + table.at[idx - 1, 'x']
        return table.at[idx, 'x']


df = pd.DataFrame(index=list(range(3)), columns=['x','y'])
df.at[0, 'x'] = 2
df.at[0, 'y'] = 1
df.at[1, 'x'] = 7
df.at[1, 'y'] = 4
df.at[2, 'x'] = 6
df.at[2, 'y'] = 3
print(df)

mapper = Mapper(df)
col_sum = SumColumns('x')
row_sum = SumPreviousRow('x')
mapper.defineTransformFunctions(col_sum)
mapper.defineTransformFunctions(row_sum)
mapper.defineProjectFunctions(['y']) # to keep not to remove
new_df = mapper.applyMapping()

print(new_df)
