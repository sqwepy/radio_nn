"""
Unit tests for the network process.

Tests dataloading and training
"""

import unittest

import numpy as np
import torch
import tqdm

from radioNN.networks.antenna_cnn_network import AntennaNetworkCNN
from radioNN.networks.antenna_fc_network import AntennaNetworkFC
from radioNN.networks.antenna_resnet_network import AntennaNetworkResNet
from radioNN.networks.antenna_skipfc_network import AntennaNetworkSkipFC
from radioNN.process_network import NetworkProcess

# TODO: Write seperate tests for dataloader classes


class ProcessTest(unittest.TestCase):
    """Test the process class and setup for other network tests."""

    def test_process_init(self, percentage=0.01, one_shower=None) -> None:
        """
        Setup for the tests.

        Returns
        -------

        """
        self.process = NetworkProcess(
            percentage=percentage,
            one_shower=one_shower,
            wb=False,
        )
        for batch in tqdm.tqdm(self.process.dataloader):
            if batch is None:
                continue
            self.assertTrue(torch.all(torch.isfinite(batch[0])))
            self.assertTrue(torch.all(torch.isfinite(batch[1])))
            self.assertTrue(torch.all(torch.isfinite(batch[2])))
            self.assertTrue(torch.all(torch.isfinite(batch[3])))
            self.assertTrue(torch.all(torch.isfinite(batch[4])))
        print(f"Output Channels: {self.process.output_channels}")
        assert 2 <= self.process.output_channels <= 3


class TestOneShower(ProcessTest, unittest.TestCase):
    """Test for the case of a single shower."""

    def setUp(self) -> None:
        """Fixure."""
        super().test_process_init(one_shower=np.random.randint(1, high=2158))

    def test_training_cnn(self) -> None:
        """Test Training."""
        self.process.model = AntennaNetworkCNN(self.process.output_channels).to("cpu")
        train_loss = self.process.train()
        self.assertTrue(np.isfinite(train_loss))

    def test_training_fc(self) -> None:
        """Test Training."""
        self.process.model = AntennaNetworkFC(self.process.output_channels).to("cpu")
        train_loss = self.process.train()
        self.assertTrue(np.isfinite(train_loss))

    def test_training_skipfc(self) -> None:
        """Test Training."""
        self.process.model = AntennaNetworkSkipFC(self.process.output_channels).to(
            "cpu"
        )
        train_loss = self.process.train()
        self.assertTrue(np.isfinite(train_loss))

    def test_training_resnet(self) -> None:
        """Test Training."""
        self.process.model = AntennaNetworkResNet(self.process.output_channels).to(
            "cpu"
        )
        train_loss = self.process.train()
        self.assertTrue(np.isfinite(train_loss))


class TestSmallDataset(ProcessTest, unittest.TestCase):
    def setUp(self) -> None:
        """Fixure."""
        super().test_process_init(percentage=0.01)

    def test_training_cnn(self) -> None:
        self.process.model = AntennaNetworkCNN(self.process.output_channels).to("cpu")
        train_loss = self.process.train()
        self.assertTrue(np.isfinite(train_loss))

    def test_training_fc(self) -> None:
        """Test Training."""
        self.process.model = AntennaNetworkFC(self.process.output_channels).to("cpu")
        train_loss = self.process.train()
        self.assertTrue(np.isfinite(train_loss))

    def test_training_resnet(self) -> None:
        """Test Training."""
        self.process.model = AntennaNetworkResNet(self.process.output_channels).to(
            "cpu"
        )
        train_loss = self.process.train()
        self.assertTrue(np.isfinite(train_loss))

    def test_training_skipfc(self) -> None:
        """Test Training."""
        self.process.model = AntennaNetworkSkipFC(self.process.output_channels).to(
            "cpu"
        )
        train_loss = self.process.train()
        self.assertTrue(np.isfinite(train_loss))


@unittest.skip("Too slow to test by default, Try manually.")
class TestEntireDataset(ProcessTest, unittest.TestCase):
    """Tests involving the entire dataset."""

    def test_dataloading(self) -> None:
        super().test_process_init(percentage=100)


if __name__ == "__main__":
    unittest.main()
