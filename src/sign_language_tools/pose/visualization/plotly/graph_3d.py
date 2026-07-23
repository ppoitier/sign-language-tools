import numpy as np
import plotly.graph_objects as go


__all__ = [
    "plot_pose_graph_3d",
]


def plot_pose_graph_3d(
    nodes: dict[str, np.ndarray],
    edges: dict[str, tuple[tuple[int, int], ...]],
    *,
    t: int | None = None,
    show: bool = True,
    renderer: str | None = None,
) -> go.Figure:
    """Plots one or more landmark groups as an interactive 3D graph.

    Each group in `nodes` is drawn as a set of vertices connected by the
    edges given in the matching entry of `edges`. The y- and z-axis are
    swapped relative to the input coordinates so that the (typically
    downward-pointing) y-axis of pose landmarks is rendered as height.

    Args:
        nodes: Mapping from a group name (e.g. `"left_hand"`) to its
            landmark coordinates, of shape `(L, 3)` or, if `t` is given,
            `(T, L, 3)`.
        edges: Mapping from a group name to the edges drawn for that
            group, as `(start, end)` landmark index pairs. Must have the
            same keys as `nodes`.
        t: If given, the frame index to select from each `(T, L, 3)`
            array in `nodes`.
        show: Whether to display the figure immediately.
        renderer: Plotly renderer used when `show` is `True`.

    Returns:
        The plotly figure containing the graph.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.visualization import plot_pose_graph_3d
        >>> from sign_language_tools.pose.mediapipe.edges import HAND_EDGES
        >>> nodes = {"right_hand": np.random.rand(21, 3)}
        >>> edges = {"right_hand": HAND_EDGES}
        >>> fig = plot_pose_graph_3d(nodes, edges, show=False)
    """
    fig = go.Figure(data=_build_traces(nodes, edges, t))
    fig.update_layout(
        title="3D Pose Graph",
        showlegend=True,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        margin=dict(b=0, l=0, r=0, t=40),
        hovermode="closest",
    )
    if show:
        fig.show(renderer=renderer)
    return fig


def _build_traces(
    nodes: dict[str, np.ndarray],
    edges: dict[str, tuple[tuple[int, int], ...]],
    t: int | None,
) -> list[go.Scatter3d]:
    traces = []
    for group_name, group_nodes in nodes.items():
        if t is not None:
            group_nodes = group_nodes[t]
        traces.append(_edge_trace(group_nodes, edges[group_name], group_name))
        traces.append(_node_trace(group_nodes, group_name))
    return traces


def _node_trace(nodes: np.ndarray, name: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=nodes[:, 0],
        y=nodes[:, 2],
        z=nodes[:, 1],
        mode="markers",
        marker=dict(size=1, line_width=1),
        name=name,
        showlegend=False,
    )


def _edge_trace(nodes: np.ndarray, edges: tuple[tuple[int, int], ...], name: str) -> go.Scatter3d:
    edge_x = []
    edge_y = []
    edge_z = []
    for start, end in edges:
        x0, y0, z0 = nodes[start]
        x1, y1, z1 = nodes[end]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    return go.Scatter3d(
        x=edge_x,
        y=edge_z,
        z=edge_y,
        line=dict(width=4, color="#888"),
        hoverinfo="none",
        mode="lines",
        name=name,
    )
