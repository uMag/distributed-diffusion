from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
ds = load_dataset("chavinlo/stream-test-2", streaming=True, split="train").with_format("torch")
this = ds.n_shards
print(type(this))
print(str(this))
dataloader = DataLoader(ds, batch_size=32, num_workers=4)
for data in dataloader:
    print(data)