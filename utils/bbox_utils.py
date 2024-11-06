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