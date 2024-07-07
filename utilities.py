import cv2
import numpy as np

centroids = {}

def color_from_id(id):
    np.random.seed(id)
    return np.random.randint(0, 255, size=3).tolist()

def draw_tracks(image, tracks, ids_dict, src, classes=None):
    vis = np.array(image)
    bboxes = tracks[:, :4]
    ids = tracks[:, 4]
    labels = tracks[:, 5]
    centroids[src] = centroids.get(src, {})

    for i, box in enumerate(bboxes):
        id = ids_dict[ids[i]]
        color = color_from_id(id)

        x1, y1, x2, y2 = np.int0(box)

        if centroids == None:
            vis = cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)
        else:
            centroids[src][id] = centroids[src].get(id, [])
            centroids[src][id].append(((x1 + x2) // 2, (y1 + y2) // 2))
            vis = draw_history(vis, box, centroids[src][id], color)

        if classes == None:
            text = f"{labels[i]} {id}"
        else:
            text = f"{classes[labels[i]]} {id}"
        vis = cv2.putText(
            vis, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
        )

    return vis

def draw_history(image, box, centroids, color):
    vis = np.array(image)

    x1, y1, x2, y2 = np.int0(box)
    thickness = 2
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

    centroids = np.int0(centroids)
    for i, centroid in enumerate(centroids):
        if i == 0:
            cv2.circle(vis, centroid, 2, color, thickness=-1)
        else:
            prev_centroid = centroids[i - 1]
            cv2.line(vis, prev_centroid, centroid, color, thickness=2)

    return vis