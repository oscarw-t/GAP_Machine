import datasets


class DatasetLoader:

    def __init__(self, dataset: str, seed: int):

        if not dataset:
            raise ValueError("Dataset name must be provided.")
        self.dataset = datasets.load_dataset(dataset)

        self._train_test_split = self.dataset["train"].train_test_split(
            test_size=0.3, shuffle=True, seed=seed
        )

        self._val_test_split = self._train_test_split["test"].train_test_split(
            test_size=0.5, shuffle=True, seed=seed
        )

    @property
    def training_split(self):
        return self._train_test_split["train"]

    @property
    def validation_split(self):
        return self._val_test_split["train"]

    @property
    def test_split(self):
        return self._val_test_split["test"]
