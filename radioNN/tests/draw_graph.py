"""
Draw graph of the radio network
"""
from radioNN.process_network import network_process_setup, train


def draw_graph():
    """
    Draw a graph of the model to check for bad gradients.

    Returns
    -------
    None
    """
    import radioNN.tests.bad_grad_viz as bgv

    criterion, dataloader, device, model, optimizer = network_process_setup(
        percentage=0.01
    )

    _ = train(model, dataloader, criterion, optimizer, device)
    loss = train(model, dataloader, criterion, optimizer, device, loss_obj=True)
    get_dot = bgv.register_hooks(loss)
    loss.backward(retain_graph=True)
    dot = get_dot(
        params=dict(model.named_parameters()),
        show_attrs=True,
    )
    dot.save(f"radio_nn.dot")
    dot.render(format="pdf")
