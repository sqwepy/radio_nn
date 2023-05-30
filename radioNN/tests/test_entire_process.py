"""
Unit tests for the network process

Tests dataloading and training
"""
import unittest
import os
import tqdm

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from radioNN.dataloader import AntennaDataset, custom_collate_fn
from radioNN.main_network import AntennaNetwork
from radioNN.process_network import train


class MyTestCase(unittest.TestCase):
    """
    Base class for all the tests.
    """

    def base_setup(self) -> None:
        """
        Setup for the tests
        Returns
        -------

        """
        radio_data_path = "/home/sampathkumar/radio_data"
        if not os.path.exists(radio_data_path):
            radio_data_path = "/home/pranav/work-stuff-unsynced/radio_data"
        assert os.path.exists(radio_data_path)
        self.input_data_file = os.path.join(radio_data_path, "input_data.npy")
        self.input_meta_file = os.path.join(radio_data_path, "meta_data.npy")
        self.antenna_pos_file = os.path.join(
            radio_data_path, "antenna_pos_data.npy"
        )
        self.output_meta_file = os.path.join(
            radio_data_path, "output_meta_data.npy"
        )
        self.output_file = os.path.join(radio_data_path, "output_gece_data.npy")

        self.criterion = nn.MSELoss()

    def base_dataloading(self, dataset):
        """
        The boilerplate after loading dataset

        Parameters
        ----------
        dataset: AntennaDataset class with loaded arrays.

        Returns
        -------

        """
        self.dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )
        for batch in tqdm.tqdm(self.dataloader):
            self.assertTrue(torch.all(torch.isfinite(batch[0])))
            self.assertTrue(torch.all(torch.isfinite(batch[1])))
            self.assertTrue(torch.all(torch.isfinite(batch[2])))
            self.assertTrue(torch.all(torch.isfinite(batch[3])))
            self.assertTrue(torch.all(torch.isfinite(batch[4])))
        output_channels = dataset.output.shape[-1]
        print(output_channels)
        assert 2 <= output_channels <= 3
        self.model = AntennaNetwork(output_channels).to("cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)


class TestCaseOneShower(MyTestCase, unittest.TestCase):
    """
    Test for the case of a single shower
    """

    def setUp(self) -> None:
        """Fixure."""
        super().base_setup()

    def test_dataloading(self):
        """Test dataloading."""
        dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode="r",
            one_shower=1,
        )
        super().base_dataloading(dataset)

    def test_training(self):
        """Test Training."""
        self.test_dataloading()
        train_loss = train(
            self.model, self.dataloader, self.criterion, self.optimizer, "cpu"
        )
        self.assertTrue(np.isfinite(train_loss))


class TestCaseSmallDataset(MyTestCase, unittest.TestCase):
    def setUp(self) -> None:
        """Fixure."""
        super().base_setup()

    def test_dataloading(self):
        """Test dataloading."""
        dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode="r",
            percentage=0.02,
        )
        super().base_dataloading(dataset)

    def test_training(self):
        self.test_dataloading()
        train_loss = train(
            self.model, self.dataloader, self.criterion, self.optimizer, "cpu"
        )
        self.assertTrue(np.isfinite(train_loss))


@unittest.skip("Too slow to test by default, Try manually.")
class TestCaseEntireDataset(MyTestCase, unittest.TestCase):
    """
    Tests involving the entire dataset.
    """

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
            percentage=100,
        )
        super().base_dataloading(dataset)


if __name__ == "__main__":
    unittest.main()
