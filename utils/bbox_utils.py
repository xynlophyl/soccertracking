def get_center_of_bbox(bbox):
    """TODO refactor to return floats and convert to ints after"""

    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bottom_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox

    return (x1+x2)/2, y2

def get_bbox_width(bbox):

    x1, _, x2, _ = bbox

    return x2 - x1

def get_bbox_height(bbox):
    
    _, y1, _, y2 = bbox
    
    return y2 - y1

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