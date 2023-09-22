def load_model(n_features, hidden_dim):
    pass


def get_dataset(dataset_path: str, with_split: bool, test_size: float, shuffle: bool):
    pass


def train(model: torch.nn.Module, train_set: torch.utils.data.Dataset, epochs, batch_size, lr, valid_set: torch.utils.data.Dataset = None):
    pass


def test(model: torch.nn.Module, test_set: torch.utils.data.Dataset, return_output: bool):
    pass