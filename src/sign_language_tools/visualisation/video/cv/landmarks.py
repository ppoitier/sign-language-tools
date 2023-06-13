from sign_language_tools.visualisation.video.cv.utils import draw_connections, draw_points


def draw_pose(
        img,
        landmarks,
        connections=None,
        edge_color=(0, 255, 0),
        vertex_color=(0, 0, 255),
        show_vertices: bool = True,
):
    if connections is not None:
        draw_connections(img, landmarks, connections, edge_color)
    if show_vertices:
        draw_points(img, landmarks, vertex_color)
