import plotly.graph_objects as go


def _get_graph_3d_data(nodes, edges):
    node_trace = go.Scatter3d(
        x=nodes[:, 0],
        y=nodes[:, 2],
        z=nodes[:, 1],
        mode="markers",
        marker=dict(colorscale="YlGnBu", size=1, line_width=1),
    )
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in edges:
        x0, y0, z0 = nodes[edge[0]]
        x1, y1, z1 = nodes[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_z,
        z=edge_y,
        line=dict(
            width=4,
            color="#888",
        ),
        hoverinfo="none",
        mode="lines",
    )
    return node_trace, edge_trace


def plot_graph_3d(
    nodes,
    edges,
    show=True,
    renderer=None,
):
    node_trace, edge_trace = _get_graph_3d_data(nodes, edges)
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="3D Graph Visualization",
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        margin=dict(b=0, l=0, r=0, t=40),
        hovermode="closest",
        annotations=[
            dict(showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)
        ],
    )
    if show:
        fig.show(renderer=renderer)
    return fig


def plot_interactive_graph_3d(
    nodes_sequence,
    edges,
    x_range=None,
    y_range=None,
    z_range=None,
    show=True,
    renderer=None,
):
    init_data = _get_graph_3d_data(nodes_sequence[0], edges)
    if x_range is None:
        x_range = (nodes_sequence[:, :, 0].min(), nodes_sequence[:, :, 0].max())
    if y_range is None:
        y_range = (nodes_sequence[:, :, 1].max(), nodes_sequence[:, :, 1].min())
    if z_range is None:
        z_range = (nodes_sequence[:, :, 2].min(), nodes_sequence[:, :, 2].max())

    frames = []
    slider_steps = []
    for idx, nodes in enumerate(nodes_sequence):
        frames.append(
            go.Frame(
                data=_get_graph_3d_data(nodes, edges),
                layout=go.Layout(title=f"Frame {idx}"),
                name=f"frame_{idx}",
            )
        )
        slider_step = dict(
            label=f"{idx}",
            method="animate",
            args=[
                [f"frame_{idx}"],
                dict(
                    frame=dict(duration=100, redraw=True),
                    mode="immediate",
                    transition=dict(duration=100),
                ),
            ],
        )
        slider_steps.append(slider_step)

    fig = go.Figure(data=init_data, frames=frames)
    fig.update_layout(
        title="3D Graph Visualization",
        showlegend=False,
        autosize=False,
        scene=dict(
            xaxis=dict(showbackground=False, range=x_range),
            yaxis=dict(showbackground=False, range=z_range),
            zaxis=dict(showbackground=False, range=y_range),
            aspectmode="cube",
        ),
        margin=dict(b=0, l=0, r=0, t=40),
        hovermode="closest",
        annotations=[
            dict(showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=20),
                    prefix="Current frame:",
                    visible=True,
                    xanchor="right",
                ),
                transition={"duration": 00, "easing": "cubic-in-out"},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=slider_steps,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None]),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
    )
    if show:
        fig.show(renderer=renderer)
    return fig
