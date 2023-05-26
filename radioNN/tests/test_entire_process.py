import unittest
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from radioNN.dataloader import AntennaDataset, custom_collate_fn
from radioNN.main_network import AntennaNetwork
from radioNN.process_network import train


class MyTestCase(unittest.TestCase):
    def base_setup(self) -> None:
        RADIO_DATA_PATH = "/home/sampathkumar/radio_data"
        memmap_mode = "r"
        if not os.path.exists(RADIO_DATA_PATH):
            RADIO_DATA_PATH = "/home/pranav/work-stuff-unsynced/radio_data"
            memmap_mode = "r"
        assert os.path.exists(RADIO_DATA_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_data_file = os.path.join(RADIO_DATA_PATH, "input_data.npy")
        self.input_meta_file = os.path.join(RADIO_DATA_PATH, "meta_data.npy")
        self.antenna_pos_file = os.path.join(RADIO_DATA_PATH, "antenna_pos_data.npy")
        self.output_meta_file = os.path.join(RADIO_DATA_PATH, "output_meta_data.npy")
        self.output_file = os.path.join(RADIO_DATA_PATH, "output_gece_data.npy")

        self.criterion = nn.MSELoss()


class TestCaseOneShower(MyTestCase, unittest.TestCase):
    def setUp(self) -> None:
        super().base_setup()

    def test_dataloading(self):
        dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode="r",
            one_shower=1,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )
        output_channels = dataset.output.shape[-1]
        print(output_channels)
        assert 2 <= output_channels <= 3
        self.model = AntennaNetwork(output_channels).to("cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        pass

    def test_training(self):
        self.test_dataloading()
        train_loss = train(
            self.model, self.dataloader, self.criterion, self.optimizer, "cpu"
        )
        self.assertTrue(np.isfinite(train_loss))


class TestCaseSmallDataset(MyTestCase, unittest.TestCase):
    def setUp(self) -> None:
        super().base_setup()

    def test_dataloading(self):
        dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode="r",
            percentage=0.02,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )
        output_channels = dataset.output.shape[-1]
        print(output_channels)
        assert 2 <= output_channels <= 3
        self.model = AntennaNetwork(output_channels).to("cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        pass

    def test_training(self):
        self.test_dataloading()
        train_loss = train(
            self.model, self.dataloader, self.criterion, self.optimizer, "cpu"
        )
        self.assertTrue(np.isfinite(train_loss))


if __name__ == "__main__":
    unittest.main()
