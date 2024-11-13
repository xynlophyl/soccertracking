def get_center_of_bbox(bbox):

    """
    return center of bbox
    """

    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):

    """
    get bounding box width
    """
    x1, _, x2, _ = bbox

    return x2 - x1

def measure_distance(p1, p2):

    """
    calculate euclidean distance between two points
    """

    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def calculate_centroid(bboxes):
    sum_x, sum_y = 0, 0
    for bbox in bboxes:
        center_x, center_y = get_center_of_bbox(bbox)
        sum_x += center_x
        sum_y += center_y
    centroid_x = sum_x / len(bboxes)
    centroid_y = sum_y / len(bboxes)
    return centroid_x, centroid_y