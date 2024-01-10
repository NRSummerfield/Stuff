import numpy as np, scipy.ndimage as ndimg, math as m

def find_closest_distance(initial:list[float], search_list:list[list[float]]):
    """
    Function to find the paired X, Y coordinate in a list that is the closest to a given X, Y coordinate
    """
    distance = 99999
    _x, _y = initial
    closest_coord = None
    for x,y in search_list:
        distance_from_initial = m.sqrt((x - _x)**2 + (y - _y)**2)
        if distance_from_initial < distance: 
            distance = distance_from_initial
            closest_coord = [x, y]
    return closest_coord

def recursive_find(initial: list[float], search_list: list[list[float]], sorted_coords: list[list[float]]):
    """
    Recursive function to create a list of X, Y coordinates that are the closest to one another
    """
    closest_coord = find_closest_distance(initial, search_list)
    if closest_coord: sorted_coords.append(closest_coord)
    if search_list:
        search_list.remove(closest_coord)
        recursive_find(initial=closest_coord, search_list=search_list, sorted_coords=sorted_coords)
    return sorted_coords

def find_breaks(coords:list[list[float]], max_distance:float=20):
    groups = []
    group = [[coords[0][0], coords[0][1]]]
    for i in range(1, len(coords)):
        ix, iy = coords[i-1]
        fx, fy = coords[i]
        distance = m.sqrt( (fx-ix)**2 + (fy-iy)**2 )
        if distance > max_distance:
            groups.append(group)
            group = []
        else:
            group.append([fx, fy])
    groups.append(group)
    return groups


def get_label_outlines(arr:np.ndarray):
    label_coord_lists = []
    nx, ny = arr.shape
    XX, YY = np.meshgrid(np.linspace(0, ny-1, ny), np.linspace(0, nx-1, nx))
    # YY = np.rot90(YY, k=2) # the images count top - down

    for n in np.unique(arr)[1:]:
        bin_label = arr == n
        edge = ndimg.binary_erosion(bin_label) ^ bin_label
        edge_coords = [[x, y] for x, y in zip(XX[edge], YY[edge])]
        sorted_coords = recursive_find(edge_coords[0], edge_coords[1:], [edge_coords[0]])
        line_groups = find_breaks(sorted_coords)

        for group in line_groups:
            xs = [c[0] for c in group]
            ys = [c[1] for c in group]
            label_coord_lists.append([n, xs, ys])

    return label_coord_lists