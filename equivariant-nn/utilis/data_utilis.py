def load_qm9_dataset(batch_size=32, split="train"):
    # Placeholder loader: you can use torch_geometric.datasets.QM9 for actual implementation
    # Simulate dummy data for now
    class DummyQM9(torch.utils.data.Dataset):
        def __len__(self): return 100
        def __getitem__(self, idx):
            x = torch.rand(5, 1)  # 5 atoms, 1 feature
            pos = torch.rand(5, 3)
            edge_index = torch.randint(0, 5, (2, 10))
            y = torch.tensor([1.0])
            return x, pos, edge_index, y

    return DataLoader(DummyQM9(), batch_size=batch_size)
