from mapper import Mapper, TableFunctions


class SumColumns(TableFunctions):
    def __init__(self, modify):
        super(SumColumns, self).__init__(modify)

    def apply(self, idx, table):
        return table.at[idx, 'x'] + table.at[idx, 'y']


class SumPreviousRow(TableFunctions):
    def __init__(self, modify):
        super(SumPreviousRow, self).__init__(modify)

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
col_sum = SumColumns('z')
row_sum = SumPreviousRow('a')
mapper.defineTransformFunctions(col_sum)
mapper.defineTransformFunctions(row_sum)
mapper.defineProjectFunctions(['x', 'z', 'a']) # to keep not to remove
new_df = mapper.applyMapping()

print(new_df)
