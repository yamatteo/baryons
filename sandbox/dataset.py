import torch

def mock_element():
    return (
            torch.randn((1, 16, 16)),
            torch.rand((1, 16, 16)),
        )

def mock_batch(size=8):
    return (
        torch.stack([mock_element()[0] for i in range(size)], dim=0),
        torch.stack([mock_element()[1] for i in range(size)], dim=0)
    )


class DatasetMock(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return mock_element()

    def __len__(self):
        return 1024
