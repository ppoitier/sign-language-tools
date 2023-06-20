import cv2


def absolute_position(img, pos):
    w = int(img.shape[1])
    h = int(img.shape[0])
    return int(max(0, min(w, w * pos[0]))), int(max(0, min(h, h * pos[1])))


def draw_connections(img, positions, connections, color=(255, 0, 0), thickness=2):
    pos_nb = len(positions)
    for i, j in connections:
        if i < pos_nb and j < pos_nb:
            x1, y1 = absolute_position(img, positions[i])
            x2, y2 = absolute_position(img, positions[j])
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_points(img, positions, color=(0, 0, 255), thickness=cv2.FILLED, radius=3):
    for pos in positions:
        x, y = absolute_position(img, pos)
        cv2.circle(img, (x, y), radius, color, thickness)


def draw_rect(img, pos1, pos2, color=(255, 255, 255), thickness=cv2.FILLED):
    cv2.rectangle(img, absolute_position(img, pos1), absolute_position(img, pos2), color=color, thickness=thickness)
