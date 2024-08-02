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
from RadioPlotter.radio_plotter import plot_hist, plot_pulses_interactive


class CustomWeightedLoss(torch.nn.Module):
    """
    L1 loss with different weights for different polarizations.

    The loss goes through backpropagation.

    Returns
    -------
    loss: torch.nn.Tensor (loss which can be used for gradients)
    """

    def __init__(self: "CustomWeightedLoss", fluence_weight: float = 0) -> None:
        super().__init__()
        self.mse_loss = torch.nn.L1Loss()
        self.fluence_weight = fluence_weight
        CONVERSION_FACTOR = 2.65441729e-3 * 6.24150934e18 * 1e-9  # c*e0*delta_t
        self.fluence = lambda x: torch.sum(x**2) * CONVERSION_FACTOR

    def forward(self, inp, outp):
        """Forward call."""
        inp_pol1, outp_pol1 = inp.T[0].T, outp.T[0].T
        inp_pol2, outp_pol2 = 10 * inp.T[1].T, 10 * outp.T[1].T
        #inp_pol3, outp_pol3 = inp.T[2].T, outp.T[2].T
        pol1_mse = self.mse_loss(inp_pol1, outp_pol1)
        pol2_mse = self.mse_loss(inp_pol2, outp_pol2)
        #pol3_mse = self.mse_loss(inp_pol3, outp_pol3)
        pol1_fluence = self.mse_loss(self.fluence(inp_pol1), self.fluence(outp_pol1))
        pol2_fluence = self.mse_loss(self.fluence(inp_pol2), self.fluence(outp_pol2))
        #pol3_fluence = self.mse_loss(self.fluence(inp_pol3), self.fluence(outp_pol3))
        return (
            pol1_mse
            + pol2_mse
        #    + pol3_mse
            + self.fluence_weight * torch.sqrt(pol1_fluence + pol2_fluence)
        )


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
    """
    A class for training and testing the given network.

    Also used for making predictions with a network in certain ways.
    """

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
        flu_weight=0,
        wb=True,
        base_path="./runs/",
    ) -> None:
        """
        Initialize the classes to be processed while training the network.

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
        self.output_channels = 2 #self.dataset.output.shape[-1]
        print(self.output_channels)
        assert 2 <= self.output_channels <= 3
        print(f"Using {model_class}")
        self.run_name = datetime.now().strftime("%y%m%b%d%a_%H%M%S")
        self.base_path = base_path
        self.log_dir = f"{self.base_path}/{self.run_name}"
        self.model = model_class(self.output_channels).to(self.device)
        # self.criterion = nn.L1Loss()
        self.criterion = CustomWeightedLoss(fluence_weight=flu_weight)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-6,
        )
        if self.wandb:
            # TODO: Do this better
            # Weird ordering of things to do, the non-wandb mode is very adhoc as is
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
                    "flu_weight": flu_weight,
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
            self.criterion = CustomWeightedLoss(fluence_weight=wandb.config.flu_weight)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=wandb.config.lr_scale,
                gamma=wandb.config.lr_decay,
            )
            # Convert the indices into wandb table
            print(self.dataset.indices)
            index_table = wandb.Table(
                columns=["indices"],
                data=self.dataset.indices.reshape(-1, 1),
            )
            wandb.log({"Indices": index_table})
            print("Uploaded Index")
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

    def send_wandb_data(
        self, epoch, train_loss, test_loss=None, real=None, sim=None
    ) -> None:
        torch.save(self.model, f"{self.log_dir}/SavedModel")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": train_loss,
                "indices": self.dataset.indices,
            },
            f"{self.log_dir}/SavedState",
        )
        model_scripted = torch.jit.script(self.model)  # Export to TorchScript
        model_scripted.save(f"{self.log_dir}/model_scripted.pt")  # Save
        wandb.save(f"{self.log_dir}/model_scripted.pt")  # Save
        wandb.save(  # pylint: disable=unexpected-keyword-arg
            f"{self.log_dir}/SavedModel",
            # base_path=f"runs",
        )
        wandb.save(  # pylint: disable=unexpected-keyword-arg
            f"{self.log_dir}/SavedState",
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

    def full_training(self: "NetworkProcess") -> None:
        """
        Do full training by loop over total number of epochs.

        Also upload data to wandb.

        Returns
        -------
        None
        """
        num_epochs = wandb.config.n_epochs
        # TODO: Fix this for non wandb cases.
        for epoch in tqdm.trange(num_epochs):
            train_loss = self.train()
            test_loss, pred_output, output = self.one_shower_loss()
            bias, resolution, xmax_dist = self.xmax_reco_loss()
            tqdm.tqdm.write(
                f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss}"
            )
            tqdm.tqdm.write(f"Epoch: {epoch + 1}/{num_epochs}, Test Loss: {test_loss}")
            tqdm.tqdm.write(f"Epoch: {epoch + 1}/{num_epochs}, Xmax Bias: {bias}")
            tqdm.tqdm.write(f"Epoch: {epoch + 1}/{num_epochs}, Xmax Res.: {resolution}")
            if self.optimizer.param_groups[-1]["lr"] <= 1e-11:
                break
            self.scheduler.step()
            if self.wandb:
                self.send_wandb_xmax(epoch, bias, resolution, xmax_dist)
                self.send_wandb_data(
                    epoch, train_loss, test_loss, real=output, sim=pred_output
                )

    def send_wandb_xmax(self, epoch, bias, resolution, xmax_dist):
        if self.wandb:
            wandb.log(
                {
                    "Xmax Loss": abs(bias) + abs(resolution),
                    "Xmax bias": bias,
                    "Xmax resolution": resolution,
                },
                step=epoch,
            )
            fig = plot_hist(xmax_dist)
            wandb.log(
                {
                    f"Xmax Hist": fig,
                },
                step=epoch,
            )

    def train(self: "NetworkProcess", loss_obj: bool = False) -> float:
        """
        Train the given model for one epoch.

        Parameters
        ----------
        loss_obj: Bool - If True, just return a single batch loss.

        Returns
        -------
        loss: Float - average loss over all batches

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

        if valid_batch_count != 0:
            return running_loss / valid_batch_count
        else:
            raise RuntimeWarning("No valid batches use a different showers/rerun tests")

    def xmax_reco(self, index):
        import numpy as np
        from scipy.optimize import minimize_scalar
        (event_data, meta_data, antenna_pos, output_meta, output) =\
        self.dataset.data_of_single_shower(index)
        xmax = meta_data[2]
        xmax_min = 550/700
        xmax_max = 1150/700
        # print(make_table(meta_data[index]))
        # print(xmax, xmax_min, xmax_max)
        # Filters
        # Chi Square
        xmax_list = []
        chi2 = []
        hfit = self.dataset.return_hfit_of_shower(index)
        for xmax_bin in np.linspace(xmax_min, xmax_max, 200):
            event_data_t = torch.Tensor(np.repeat([event_data.T], 240,
                                                            axis=0).T)
            meta_data_inp = torch.Tensor(
                np.copy(np.repeat([meta_data], 240, axis=0)))
            meta_data_inp.T[2] = torch.Tensor(np.repeat([xmax_bin], 240,
                                                                  axis=0).T)
            meta_data_inp.T[4] = torch.Tensor(hfit(meta_data_inp.T[2]))
            antenna_pos = torch.Tensor(np.copy(antenna_pos))
            pred_output_meta, pred_output = self.model(event_data_t, meta_data_inp,
                                                       antenna_pos)
            pred_output = pred_output.detach().numpy()
            res = minimize_scalar(lambda x: np.sum(((x ** 2) * np.sum(
                pred_output[:, :, 0] ** 2, axis=1) - np.sum(
                output[:, :, 0] ** 2, axis=1)) ** 2) * 1e10)
            # print(res.x)
            res2 = minimize_scalar(lambda x: np.sum(((x ** 2) * np.sum(
                pred_output[:, :, 1] ** 2, axis=1) - np.sum(
                output[:, :, 1] ** 2, axis=1)) ** 2) * 1e10)

            xmax_list.append(xmax_bin)
            chi2.append(res.fun + res2.fun)

        return (meta_data[2] - np.array(xmax_list)[np.argmin(np.array(chi2))])*700

    def xmax_reco_loss(self):
        import numpy as np
        reco_accuray_nn_full = np.full((2000), np.nan)
        for i, index in tqdm.tqdm(enumerate(np.random.choice(
                np.unique(self.dataset.indices//240),
                                                         size=50)),
                             total=50):
            reco_accuray_nn_full[i] = self.xmax_reco(index)
        bias = np.mean(reco_accuray_nn_full[np.isfinite(reco_accuray_nn_full)])
        resolution = np.std(reco_accuray_nn_full[np.isfinite(reco_accuray_nn_full)])
        return bias, resolution, reco_accuray_nn_full[np.isfinite(reco_accuray_nn_full)]

    def one_shower_loss(self):
        test_loss = CustomWeightedLoss()
        for _ in range(10):
            try:
                train_indices = torch.unique(torch.Tensor(self.dataset.indices // 240))
                indices = [x for x in range(26387) if x not in train_indices]
                choice = indices[torch.randint(high=len(indices), size=[1])]
                pred_output_meta, pred_output, output = self.pred_one_shower(choice)
            except RuntimeError as e:
                print(e)
                print("Trying again")
                continue
            break
        loss_output = test_loss(250 * pred_output, 250 * output)
        print(f"Choosing shower {choice}")
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
