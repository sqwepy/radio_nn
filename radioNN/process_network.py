"""Process network class which takes of setup training and inference."""
import os
from datetime import datetime

import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader

import wandb
from radioNN.data.loader import AntennaDataset, custom_collate_fn
from radioNN.networks.antenna_fc_network import AntennaNetworkFC
from RadioPlotter.radio_plotter import plot_pulses_interactive


class CustomWeightedLoss(torch.nn.Module):
    """
    MSE Loss after applying a log.

    The loss goes though backpropagation.
    """

    def __init__(self, tol=1e-14):
        super().__init__()
        self.mse_loss = torch.nn.L1Loss()
        self.fluence = lambda x: torch.sum(x**2)

    def forward(self, inp, outp):
        """Forward call."""
        inp_pol1, outp_pol1 = inp.T[0].T, outp.T[0].T
        inp_pol2, outp_pol2 = 10 * inp.T[1].T, 10 * outp.T[1].T
        inp_pol3, outp_pol3 = 100 * inp.T[2].T, 100 * outp.T[2].T
        pol1_mse = self.mse_loss(inp_pol1, outp_pol1)
        pol2_mse = self.mse_loss(inp_pol2, outp_pol2)
        pol3_mse = self.mse_loss(inp_pol3, outp_pol3)
        # pol1_fluence = self.mse_loss(self.fluence(inp_pol1), self.fluence(outp_pol1))
        # pol2_fluence = self.mse_loss(self.fluence(inp_pol2), self.fluence(outp_pol2))
        # pol3_fluence = self.mse_loss(self.fluence(inp_pol3), self.fluence(outp_pol3))
        return pol1_mse + pol2_mse + pol3_mse  # \
        # +1e-3*( pol1_fluence + pol2_fluence + pol3_fluence)


def fit_plane_and_return_3d_grid(pos):
    import numpy as np
    from scipy.sparse.linalg import lsqr

    design_matrix = np.ones_like(pos)
    design_matrix[:, :2] = pos[:, :2]
    fit = lsqr(design_matrix, pos[:, 2])[0]
    xs = np.linspace(np.min(pos[:, 0]), np.max(pos[:, 0]), 100)
    ys = np.linspace(np.min(pos[:, 1]), np.max(pos[:, 1]), 100)
    X, Y = np.meshgrid(xs, ys)
    Z = fit[0] * X + fit[1] * Y + fit[2]
    return np.dstack([X, Y, Z]).reshape(-1, 3)


class NetworkProcess:
    def __init__(
        self,
        percentage=100,
        one_shower=None,
        model_class=AntennaNetworkFC,
        batch_size=4,
        n_epochs=10,
        lr=1e-3,
        weight_decay=1e-6,
        lr_scale=100,
        lr_decay=0.5,
        wb=True,
        base_path="./runs/",
    ):
        """
        Create the classes to be processed while training the network.

        Parameters
        ----------
        percentage: Percentage of data to be used.
        one_shower: if not None, use only the shower of given number.

        Returns
        -------
        criterion: Loss function
        dataloader: Dataloader Class to load data.
        device: cpu or gpu
        model: Model Class
        optimizer: Optimization Algorithm

        """
        self.wandb = wb
        radio_data_path = "/home/sampathkumar/radio_data"
        memmap_mode = "r"
        if not os.path.exists(radio_data_path):
            radio_data_path = "/home/pranav/work-stuff-unsynced/radio_data"
            memmap_mode = "r"
        if not os.path.exists(radio_data_path):
            radio_data_path = "/cr/work/sampathkumar/radio_data"
            memmap_mode = "r"
        assert os.path.exists(radio_data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if one_shower is not None:
            print(
                f"Using the data from {radio_data_path} in {self.device} with "
                f"memmap "
                f"mode: {memmap_mode} using only shower {one_shower}"
            )
        else:
            print(
                f"Using the data from {radio_data_path} in {self.device} with "
                f"memmap "
                f"mode: {memmap_mode} using {percentage}% of data"
            )
        self.input_data_file = os.path.join(radio_data_path, "input_data.npy")
        self.input_meta_file = os.path.join(radio_data_path, "meta_data.npy")
        self.antenna_pos_file = os.path.join(radio_data_path, "antenna_pos_data.npy")
        self.output_meta_file = os.path.join(radio_data_path, "output_meta_data.npy")
        # self.output_file = os.path.join(radio_data_path, "output_gece_data.npy")
        self.output_file = os.path.join(radio_data_path, "output_vBvvB_data.npy")
        self.dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode=memmap_mode,
            percentage=percentage,
            one_shower=one_shower,
            device=self.device,
        )
        self.output_channels = self.dataset.output.shape[-1]
        print(self.output_channels)
        assert 2 <= self.output_channels <= 3
        print(f"Using {model_class}")
        self.run_name = datetime.now().strftime("%y%m%b%d%a_%H%M%S")
        self.base_path = base_path
        self.log_dir = f"{self.base_path}/{self.run_name}"
        self.model = model_class(self.output_channels).to(self.device)
        # self.criterion = nn.L1Loss()
        self.criterion = CustomWeightedLoss()
        if self.wandb:
            wandb.init(
                project="RadioNN",
                name=self.run_name,
                entity="pranavsampathkumar",
                config={
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "lr_scale": lr_scale,
                    "lr_decay": lr_decay,
                },
                save_code=True,
                group=type(self.model).__name__,
            )
            try:
                os.mkdir(f"{self.log_dir}")
            except FileExistsError:
                pass
            wandb.watch(self.model)
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=wandb.config.lr,
                weight_decay=wandb.config.weight_decay,
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, verbose=True, eps=1e-12, patience=wandb.config.lr_scale
            )
        try:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=wandb.config.batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn,
            )
        except:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=8,
                shuffle=True,
                collate_fn=custom_collate_fn,
            )

    def send_wandb_data(self, epoch, train_loss, test_loss=None, real=None, sim=None):
        torch.save(self.model, f"{self.log_dir}/SavedModel")
        wandb.save(  # pylint: disable=unexpected-keyword-arg
            f"{self.log_dir}/SavedModel",
            # base_path=f"runs",
        )
        wandb.log(
            {
                "LearningRate": self.optimizer.param_groups[-1]["lr"],
                "Train Loss": train_loss,
                "Test Loss": test_loss,
            },
            step=epoch,
        )
        if epoch % 2 == 0 and real is not None:
            antennas = [7, 47, 79]
            for ant in antennas:
                figures = plot_pulses_interactive(real, sim, antenna=ant)
                wandb.log(
                    {
                        f"Pol 1 {ant}": figures[0],
                        f"Pol 2 {ant}": figures[1],
                    },
                    step=epoch,
                )

    def full_training(self):
        num_epochs = wandb.config.n_epochs
        for epoch in tqdm.trange(num_epochs):
            train_loss = self.train()
            test_loss, pred_output, output = self.one_shower_loss()
            tqdm.tqdm.write(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {train_loss}")
            tqdm.tqdm.write(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {test_loss}")
            if self.optimizer.param_groups[-1]["lr"] <= 1e-11:
                break
            self.scheduler.step(train_loss)
            if self.wandb:
                self.send_wandb_data(
                    epoch, train_loss, test_loss, real=output, sim=pred_output
                )

    def train(self, loss_obj=False):
        """
        Train the given model using given data, criteria and optimizer.

        Parameters
        ----------
        model: Model Class
        dataloader: Dataloader Class to load data.
        criterion: Loss function
        optimizer: Optimization Algorithm
        device: cpu or gpu
        loss_obj: If True, just return a single batch loss.

        Returns
        -------

        """
        running_loss = 0.0
        valid_batch_count = 0

        for batch in tqdm.autonotebook.tqdm(self.dataloader, leave=False):
            if batch is None:
                tqdm.tqdm.write(f"Skipped batch {batch}")
                continue

            event_data, meta_data, antenna_pos, output_meta, output = batch

            self.optimizer.zero_grad()
            pred_output_meta, pred_output = self.model(
                event_data, meta_data, antenna_pos
            )

            # TODO: instead of self.criterion(pred_output_meta, output_meta)
            # add meta to criterion and handle via loss function
            loss_output = self.criterion(250 * pred_output, 250 * output)
            loss = loss_output  # + loss_meta

            if loss_obj:
                return loss
            loss.backward()
            self.optimizer.step()

            if valid_batch_count % 100 == 0 and valid_batch_count != 0:
                tqdm.tqdm.write(f"Batch Loss is {loss.item()}")

            running_loss += loss.item()

            valid_batch_count += 1

        return running_loss / valid_batch_count

    def one_shower_loss(self):
        for _ in range(10):
            try:
                choice = torch.randint(high=26387, size=[1])
                pred_output_meta, pred_output, output = self.pred_one_shower(choice)
            except RuntimeError as e:
                print(e)
                print("Trying again")
                continue
            break
        loss_output = self.criterion(250 * pred_output, 250 * output)
        return loss_output.item(), pred_output.cpu().numpy(), output.cpu().numpy()

    def pred_one_shower(self, one_shower):
        # TODO : Use dataloader.return_single_shower
        one_sh_dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode="r",
            one_shower=one_shower,
        )
        dataloader = DataLoader(
            one_sh_dataset,
            batch_size=len(one_sh_dataset),
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        assert len(dataloader) == 1
        for batch in dataloader:
            if batch is None:
                raise RuntimeError(f"Not a valid Shower {one_shower}")

            event_data, meta_data, antenna_pos, output_meta, output = batch
            with torch.no_grad():
                pred_output_meta, pred_output = self.model(
                    event_data, meta_data, antenna_pos
                )

            return (pred_output_meta, pred_output, output)

    def pred_one_shower_entire_array(self, one_shower):
        # TODO : Use dataloader.return_single_shower
        one_sh_dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode="r",
            one_shower=one_shower,
        )
        dataloader = DataLoader(
            one_sh_dataset,
            batch_size=len(one_sh_dataset),
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        assert len(dataloader) == 1
        for batch in dataloader:
            if batch is None:
                raise RuntimeError("Not a valid Shower {one_shower}")

            event_data, meta_data, antenna_pos, output_meta, output = batch
            # TODO: Fix it in the input file and stop swapaxes.
            antenna_pos = torch.Tensor(
                fit_plane_and_return_3d_grid(antenna_pos.cpu().numpy())
            )
            assert torch.all(event_data == event_data[0])
            assert torch.all(meta_data == meta_data[0])
            with torch.no_grad():
                pred_output_meta, pred_output = self.model(
                    event_data[0].expand(antenna_pos.shape[0], *event_data[0].shape),
                    meta_data[0].expand(antenna_pos.shape[0], *meta_data[0].shape),
                    antenna_pos,
                )

            return pred_output_meta.cpu().numpy(), pred_output.cpu().numpy()
