"""
Main training file which needs to be run.
"""
import numpy as np
import torch
from torch.autograd import profiler
import tqdm

from radioNN.process_network import network_process_setup, train
from radioNN.tests.draw_graph import draw_graph


def main():
    """
    Run Code.
    Returns
    -------

    """
    criterion, dataloader, device, model, optimizer = network_process_setup()
    num_epochs = 100

    for epoch in tqdm.trange(num_epochs):
        train_loss = train(model, dataloader, criterion, optimizer, device)
        tqdm.tqdm.write(
            f"Epoch: {epoch + 1}/{num_epochs}, Loss:" f" {train_loss:.6f}"
        )

    torch.save(model.state_dict(), "antenna_network.pth")


def profile():
    """
    Profile code.
    Returns
    -------

    """

    criterion, dataloader, device, model, optimizer = network_process_setup()
    _ = train(model, dataloader, criterion, optimizer, device)  # warmup
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        _ = train(model, dataloader, criterion, optimizer, device)
        print("ONE EPOCH DONE")

    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cpu_time_total", row_limit=5
        )
    )


def one_shower_training(one_shower=1):
    """
    Run Code for just one shower to see if memorizes the shower.
    Returns
    -------

    """
    print(f"Use shower {one_shower}")
    criterion, dataloader, device, model, optimizer = network_process_setup(
        one_shower=one_shower
    )
    num_epochs = 100

    for epoch in tqdm.trange(num_epochs):
        train_loss = train(model, dataloader, criterion, optimizer, device)
        tqdm.tqdm.write(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {train_loss}")

    torch.save(model.state_dict(), "antenna_network.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--one_shower", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--graph", action="store_true")
    opt = parser.parse_args()
    if opt.one_shower:
        print("Training with a single shower")
        one_shower_training(np.random.randint(low=1, high=2300))
        exit()
    if opt.profile:
        print("Profiling Code")
        profile()
        exit()
    if opt.graph:
        print("Drawing the computation graph")
        draw_graph()
        exit()
    print("No options provided, executing main training")
    main()
