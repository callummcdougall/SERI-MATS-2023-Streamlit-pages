# %% [markdown] [2]:

import torch

L = """112 86 139 96 72 79 114 1 65 65 138 74 30 117 141 119 83 34 138 45 73 39 1 112 140 128 138 112 92 67 134 1 82 77 136 106 34 123 141 127 63 63 138 103 60 106 1 110 27 123 141 139 37 60 131 103 74 116 1 12""".split(" ")

lists = [[] for _ in range(24)]

for i in range(len(L)):
    if len(lists[i%24]) == 2 and (i//24) >= 2:
        print("Hello", i)
        lists[i%24].append(141/2)
    lists[(i%24) + (12 if (i//24)>=2 else 0)].append(int(L[i]))
m = torch.FloatTensor(lists)
m /= 141.0
m -= 0.5
m *= 2.0

m = m**8

bools = m>=0.0
bools = bools.flatten()
# bools = torch.cat((bools[:48], bools[-12:]), dim=0)
m = m.abs()

print(
    str(m.tolist()).replace("\n","") + ";"
)
print()
print(
    str(bools.flatten().tolist()).lower() + ";"
)
