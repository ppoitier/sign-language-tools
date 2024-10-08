import numpy as np
import plotly.graph_objects as go


def _get_graph_data(nodes, edges, i=0, label='graph'):
    node_trace = go.Scatter(
        x=nodes[:, 0],
        y=nodes[:, 1],
        mode="markers",
        marker=dict(colorscale="YlGnBu", size=10, line_width=1, color=[i]*nodes.shape[0]),
        name=label,
        showlegend=False,
    )
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = nodes[edge[0]]
        x1, y1 = nodes[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(
            width=4,
        ),
        hoverinfo="none",
        mode="lines",
        name=label,
    )
    return node_trace, edge_trace


def _get_graphs_data(
    nodes: dict[str, np.ndarray], edges: dict[str, np.ndarray], t: int | None = None
):
    data = []
    for i, key in enumerate(nodes.keys()):
        sub_nodes = nodes[key]
        if t is not None:
            sub_nodes = sub_nodes[t]
        data.extend(_get_graph_data(sub_nodes, edges[key], i=i, label=key))
    return data


def plot_graph(
    nodes: dict[str, np.ndarray],
    edges: dict[str, np.ndarray],
    show=True,
    renderer=None,
    t=None,
):
    data = _get_graphs_data(nodes, edges, t)
    fig = go.Figure(data=data)
    fig.update_layout(
        title="Graph Visualization",
        showlegend=True,
        autosize=False,
        xaxis=dict(range=(0, 1)),
        yaxis=dict(range=(1, 0)),
        margin=dict(b=0, l=0, r=0, t=40),
        hovermode="closest",
        annotations=[
            dict(showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)
        ],
    )
    if show:
        fig.show(renderer=renderer)
    return fig


def plot_interactive_graphs(
    nodes_sequences: dict[str, np.ndarray],
    edges: dict[str, np.ndarray],
    show=True,
    renderer=None,
):
    ...

    # init_data = _get_graph_2d_data(nodes_sequence[0], edges)
    #
    # frames = []
    # slider_steps = []
    # for idx, nodes in enumerate(nodes_sequence):
    #     frames.append(
    #         go.Frame(
    #             data=_get_graph_2d_data(nodes, edges),
    #             layout=go.Layout(title=f"Frame {idx}"),
    #             name=f"frame_{idx}",
    #         )
    #     )
    #     slider_step = dict(
    #         label=f"{idx}",
    #         method="animate",
    #         args=[
    #             [f"frame_{idx}"],
    #             dict(
    #                 frame=dict(duration=100, redraw=True),
    #                 mode="immediate",
    #                 transition=dict(duration=100),
    #             ),
    #         ],
    #     )
    #     slider_steps.append(slider_step)
    #
    # fig = go.Figure(data=init_data, frames=frames)
    #
    # slider = dict(
    #     active=0,
    #     yanchor="top",
    #     xanchor="left",
    #     currentvalue=dict(
    #         font=dict(size=20),
    #         prefix="Current frame:",
    #         visible=True,
    #         xanchor="right",
    #     ),
    #     transition={"duration": 00, "easing": "cubic-in-out"},
    #     pad={"b": 10, "t": 50},
    #     len=0.9,
    #     x=0.1,
    #     y=0,
    #     steps=slider_steps,
    # )
    #
    # menus = dict(
    #     type="buttons",
    #     buttons=[
    #         dict(label="Play", method="animate", args=[None]),
    #         dict(
    #             label="Pause",
    #             method="animate",
    #             args=[
    #                 [None],
    #                 dict(
    #                     frame=dict(duration=0, redraw=False),
    #                     mode="immediate",
    #                     transition=dict(duration=0),
    #                 ),
    #             ],
    #         ),
    #     ],
    # )
    #
    # fig.update_layout(
    #     title="Graph Visualization",
    #     showlegend=False,
    #     autosize=False,
    #     scene=dict(
    #         xaxis=dict(showbackground=False, range=x_range),
    #         yaxis=dict(showbackground=False, range=z_range),
    #         zaxis=dict(showbackground=False, range=y_range),
    #         aspectmode="cube",
    #     ),
    #     margin=dict(b=0, l=0, r=0, t=40),
    #     hovermode="closest",
    #     annotations=[
    #         dict(showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)
    #     ],
    #     sliders=[slider],
    #     updatemenus=[menus],
    # )
    # if show:
    #     fig.show(renderer=renderer)
    # return fig
