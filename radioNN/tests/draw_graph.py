"""
Draw graph of the radio network
"""
from radioNN.process_network import NetworkProcess


def draw_graph():
    """
    Draw a graph of the model to check for bad gradients.

    Returns
    -------
    None
    """
    import radioNN.tests.bad_grad_viz as bgv

    process = NetworkProcess(one_shower=33)

    _ = process.train()
    loss = process.train(loss_obj=True)
    get_dot = bgv.register_hooks(loss)
    loss.backward(retain_graph=True)
    dot = get_dot(
        params=dict(process.model.named_parameters()),
        show_attrs=True,
    )
    dot.save(f"radio_nn.dot")
    dot.render(format="pdf")
    print("Saved file to radio_nn.dot.pdf")
