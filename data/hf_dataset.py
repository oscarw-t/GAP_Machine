import torch.utils.data


class HFDataset(torch.utils.data.Dataset):

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]
        image = item["image"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label
