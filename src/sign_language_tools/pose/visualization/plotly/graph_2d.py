import numpy as np
import plotly.graph_objects as go


__all__ = [
    "plot_pose_graph_2d",
]


def plot_pose_graph_2d(
    nodes: dict[str, np.ndarray],
    edges: dict[str, tuple[tuple[int, int], ...]],
    *,
    t: int | None = None,
    show: bool = True,
    renderer: str | None = None,
) -> go.Figure:
    """Plots one or more landmark groups as an interactive 2D graph.

    Each group in `nodes` is drawn as a set of vertices connected by the
    edges given in the matching entry of `edges`.

    Args:
        nodes: Mapping from a group name (e.g. `"left_hand"`) to its
            landmark coordinates, of shape `(L, 2)` or, if `t` is given,
            `(T, L, 2)`.
        edges: Mapping from a group name to the edges drawn for that
            group, as `(start, end)` landmark index pairs. Must have the
            same keys as `nodes`.
        t: If given, the frame index to select from each `(T, L, 2)`
            array in `nodes`.
        show: Whether to display the figure immediately.
        renderer: Plotly renderer used when `show` is `True`.

    Returns:
        The plotly figure containing the graph.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.visualization import plot_pose_graph_2d
        >>> from sign_language_tools.pose.mediapipe.edges import HAND_EDGES
        >>> nodes = {"right_hand": np.random.rand(21, 2)}
        >>> edges = {"right_hand": HAND_EDGES}
        >>> fig = plot_pose_graph_2d(nodes, edges, show=False)
    """
    fig = go.Figure(data=_build_traces(nodes, edges, t))
    fig.update_layout(
        title="2D Pose Graph",
        showlegend=True,
        xaxis=dict(range=(0, 1)),
        yaxis=dict(range=(1, 0)),
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
) -> list[go.Scatter]:
    traces = []
    for group_name, group_nodes in nodes.items():
        if t is not None:
            group_nodes = group_nodes[t]
        traces.append(_node_trace(group_nodes, group_name))
        traces.append(_edge_trace(group_nodes, edges[group_name], group_name))
    return traces


def _node_trace(nodes: np.ndarray, name: str) -> go.Scatter:
    return go.Scatter(
        x=nodes[:, 0],
        y=nodes[:, 1],
        mode="markers",
        marker=dict(size=10, line_width=1),
        name=name,
        showlegend=False,
    )


def _edge_trace(nodes: np.ndarray, edges: tuple[tuple[int, int], ...], name: str) -> go.Scatter:
    edge_x = []
    edge_y = []
    for start, end in edges:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    return go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=4),
        hoverinfo="none",
        mode="lines",
        name=name,
    )
