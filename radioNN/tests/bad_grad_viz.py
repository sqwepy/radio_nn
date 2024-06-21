"""Draw a graph of the network highlighting gradient information."""

import itertools
import textwrap

import torch
from graphviz import Digraph

none_count = itertools.count()

SAVED_PREFIX = "_saved_"


def hex_colour(red, green, blue) -> str:
    """Return Colour from int to Hex."""
    return f"#{red:02x}{green:02x}{blue:02x}"


def iter_graph(root, callback) -> None:
    """Iterate over graph nodes from the root function."""
    queue = [root]
    seen = set()
    while queue:
        function = queue.pop()
        if function in seen:
            continue
        seen.add(function)
        for next_fn, _ in function.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(function)


def register_hooks(var):
    """Register hooks for all the functions part of the computation graph."""
    fn_dict_input = {}
    fn_dict_output = {}

    def get_var_html_label(function, param_map):
        """Make HTML label for the variable."""
        _var = function.variable
        node_name = [
            f"{type(_var).__name__}: " + str(id(function))[-4:],
            get_var_name(_var, param_map),
        ]
        label = (
            r"""<<TABLE CELLPADDING="0" CELLBORDER="1" CELLSPACING="0"
                    BORDER="0">"""
            + "".join(
                [
                    f"""<TR> <TD BGCOLOR="{hex_colour(173, 216, 230)}"
                        > {node_name[ii]} </TD> </TR> """
                    for ii in range(len(node_name))
                ]
            )
            + """</TABLE >>"""
        )
        return label

    def get_fn_html_label(function, show_attr, max_attr_chars):
        """Make HTML label for functions."""
        num_grads = len(fn_dict_input[function])
        assert num_grads > 0
        string = str(type(function).__name__) + ":" + str(id(function))[-4:]
        parts = textwrap.wrap(string, int(len(string) / num_grads))
        attr_params = []
        if show_attr:
            attrs = {}
            for attr in dir(function):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(function, attr)
                attr = attr[len(SAVED_PREFIX) :]
                if torch.is_tensor(val):
                    attrs[attr] = (
                        f"[saved tensor] " f"{torch.mean(val, dtype=torch.float):.3e}"
                    )
                elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
                    attrs[attr] = (
                        f"[saved tensors] " f"{torch.mean(val, dtype=torch.float):.3e}"
                    )
                else:
                    attrs[attr] = val
            for i, grad_inp in enumerate(fn_dict_input[function]):
                if grad_inp is None:
                    attrs[f"mean_grad_{i}"] = "None"
                else:
                    attrs[f"mean_grad_{i}"] = f"{torch.mean(grad_inp).item():.3e}"
            max_attr_chars = max(max_attr_chars, 3)
            attr_params = [f"{k}: {v!s}<BR/>" for (k, v) in attrs.items()]
        label = (
            r"""<
            <TABLE CELLPADDING="1" CELLBORDER="1" CELLSPACING="0"
            BORDER="0">
            <TR>
            """
            + r"".join(
                [
                    f'<TD BGCOLOR="{get_fill_colour2(fn_dict_input[function][i])}"'
                    f' PORT="f{i}"'
                    f">"
                    f"{parts[i]} </TD>"
                    for i in range(num_grads)
                ]
            )
            + rf"""</TR><TR><TD COLSPAN="{num_grads}"
            ROWSPAN= "{max(1,len(attr_params))}"
            PORT="attr"
                >"""
            + r"".join([att for att in attr_params])
            + """</TD></TR></TABLE>
            >"""
        )
        return label

    def hook_cb(function) -> None:
        """Hook callback to function."""

        def register_grad(grad_input, grad_output) -> None:
            fn_dict_input[function] = grad_input
            fn_dict_output[function] = grad_output

        function.register_hook(register_grad)

    iter_graph(var.grad_fn, hook_cb)

    def size_to_str(size):
        """Convert size of tensor to string."""
        return "(" + ", ".join([f"{v}" for v in size]) + ")"

    def get_var_name(variab, param_map, name=None) -> str:
        """Get variable name from object."""
        if not name:
            name = param_map[id(variab)] if id(variab) in param_map else ""
        return f"{name}: {size_to_str(variab.size())} {torch.mean(variab):.1e}"

    def bad_grad_value(grad_output):
        """Metric for bad gradients in a mode."""
        grad_output = grad_output.data
        sum_bad = torch.sum(
            grad_output.ne(grad_output) | grad_output.gt(1e6) | grad_output.le(-1e6)
        )
        return sum_bad / torch.numel(grad_output)

    def get_fill_colour2(grad):
        """Get fill colour for node based on bad gradients."""
        if grad is None:
            return hex_colour(150, 150, 150)
        mean_bad_grad_col = int(255 - bad_grad_value(grad) * 255)
        fillcolor = hex_colour(255, mean_bad_grad_col, mean_bad_grad_col)
        return fillcolor

    def make_dot(params=None, show_attrs=False, max_attr_chars=50):
        """Make node."""
        if params is not None:
            assert all(isinstance(p, torch.autograd.Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}
        else:
            param_map = {}
        node_attr = dict(
            shape="plaintext",
            align="center",
            fontsize="12",
            ranksep="0.1",
            height="0.2",
        )
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def build_graph(function) -> None:
            """Build graph elements of a particular function."""
            if hasattr(function, "variable"):  # if GradAccumulator
                label = get_var_html_label(function, param_map)
                dot.node(str(id(function)), label)
            else:
                assert function in fn_dict_input, function
                node_label = get_fn_html_label(function, show_attrs, max_attr_chars)
                dot.node(str(id(function)), node_label)
            for next_fn, _ in function.next_functions:
                if next_fn is not None:
                    index = None
                    for i, k in enumerate(fn_dict_input[function]):
                        if k is None or fn_dict_output[next_fn][0] is None:
                            continue
                        if torch.equal(fn_dict_output[next_fn][0], k):
                            index = i

                    next_id = id(next_fn)
                    string = ""
                    if index is not None:
                        string = f":f{index}"
                    dot.edge(
                        str(next_id),
                        str(id(function)) + string,
                    )
                else:
                    idn = id(next_fn) + next(none_count)
                    dot.node(
                        str(idn),
                        rf"""<
                            <TABLE
                            CELLPADDING="0"
                            CELLBORDER="1"
                            CELLSPACING="0"
                            BORDER="0">
                            <TR>
                            <TD BGCOLOR="{hex_colour(150, 150, 150)}"
                            >CONST (None Grad) </TD>
                            </TR></TABLE>
                            >""",
                    )
                    index = None
                    for i, k in enumerate(fn_dict_input[function]):
                        if k is None:
                            index = i
                    string = ""
                    if index is not None:
                        string = f":f{index}"
                    dot.edge(
                        str(idn),
                        str(id(function)) + string,
                        _attributes={"style": "dotted"},
                    )

        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot
