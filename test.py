import numpy as np

a = [0, 0, 1, 1, 5]
class DataLoader:
    def __init__(self, size: int):
        self.data = a
        self.size = size
        self.iterator = 0

    def __iter__(self):
        for n in range(3):
            batch = self.data[self.iterator:self.iterator+self.size]
            self.iterator += self.size
            yield batch

    def __len__(self):
        return np.ceil(len(self.data) // self.size)

dl = DataLoader(2)
for b in dl:
    print(b)
