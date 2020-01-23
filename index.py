import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib import animation
import fiona
import shapely.geometry as sgeom
from shapely.prepared import prep
import random
import datetime
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

import cartopy.io.img_tiles as cimgt

config = {
    'search_area': 1250, # sq. miles
    'cell_size': 1, # 100 sq meters (in sq miles)
    'center': (-82.3525, 27.9), # lon/lat
    'time_interval': 2, # estimated # of seconds it takes for target to transition from one cell to the next
    'time_since_seen': 0 # time_intervals since last sighting of target
    # *TODO: speed*
}

def generate_extent():
    distance = np.sqrt(config['search_area']) / 2

    delta_lat = distance / 69 # North-south distance in degrees
    delta_lon = np.abs(delta_lat / np.cos(config['center'][1])) # East-west distance in degrees

    left = config['center'][0] - delta_lon
    right = config['center'][0] + delta_lon
    bottom = config['center'][1] - delta_lat
    top = config['center'][1] + delta_lat

    return [left, right, bottom, top]

def generate_grid():
    grid_width_miles = np.sqrt(config['search_area'])
    cell_width_miles = np.sqrt(config['cell_size'])

    grid_width_cells = int(grid_width_miles / cell_width_miles)

    grid = np.zeros((grid_width_cells, grid_width_cells))

    return grid

def is_land(coords):
    return land.contains(sgeom.Point(coords[0], coords[1]))

def generate_transition_matrix(op):
    length = len(op.flatten())
    transition_matrix = np.zeros((length,length))
    flat_op = op.flatten()

    for x, i in enumerate(op):
        for y, j in enumerate(i):
            cell_flat_index = x*len(i) + y

            adjacent_cells = []
            possible_adjacent_cells = [(x-1,y),(x+1,y),(x,y+1),(x,y-1)]

            for a_x, a_y in possible_adjacent_cells:
                print([a_x, a_y])
                lat_lon = convert_xy_to_map_coords([a_x, a_y])
                if a_x < len(op) and a_y < len(op[0]) and a_x >= 0 and a_y >= 0 and is_land(lat_lon):
                    adjacent_cells.append((a_x, a_y))

            transition_prob = 1 / (len(adjacent_cells) + 1)

            for adjacent_cell in adjacent_cells:
                adjacent_cell_flat_index = adjacent_cell[0]*len(i) + adjacent_cell[1]
                transition_matrix[cell_flat_index][adjacent_cell_flat_index] = transition_prob
                transition_matrix[cell_flat_index][cell_flat_index] = transition_prob

    return transition_matrix

def search():
    global search_locations
    global start_time
    global pause
    global found

    Y_p = generate_grid().flatten()
    op = generate_grid()

    op = op.flatten()
    Z_i = generate_grid().flatten()
    dp = np.full(len(Y_p), 1)
    location = int(len(Y_p) / 2) - 1 #random.randint(0,399)
    Y_p[location] = 1

    while found == False:
        dimension = int(np.sqrt(len(Y_p)))
        op = op.reshape(dimension,dimension)

        interval = op.max() / 5
        bounds = [0, 0.1, 1, 2, 3, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        yield [op, norm, cmap]
        index = int(np.sqrt(len(Y_p)) / 2) - 1 # - 1 because 0 indexing
        op[index][index] = 1.0

        transition_matrix = generate_transition_matrix(op)

        for initial_iteration in range(config['time_since_seen']):
            op = np.dot(op.flatten(), transition_matrix)

            # target moves randomly
            move_options = [0, 1, -1, np.sqrt(len(Y_p)), -np.sqrt(len(Y_p))]
            move = random.choice(move_options)
            Y_p[location] = 0
            location = location + move if location + move < len(Y_p) and location + move > 0 else location
            Y_p[location] = 1

        while True:
            if found == True:
                break

            # pause between updates to plot
            time_passed = datetime.datetime.utcnow() - start_time
            if time_passed.total_seconds() > config['time_interval']:
                pause = False
                start_time = datetime.datetime.utcnow()
            if pause == False:
                op = np.dot(op.flatten(), transition_matrix)

                for i, cell in enumerate(op):
                    if found == True:
                        break
                    for coords in search_locations:
                        if i == round(coords[0]*np.sqrt(len(Y_p)) + coords[1]):
                            detection_prob = Y_p[i] * dp[i]

                            interval = op.max() / 5
                            bounds = [0, interval, interval*2, interval*3, interval*4, op.max()]
                            norm = colors.BoundaryNorm(bounds, cmap.N)

                            yield [op.flatten(), norm, cmap] #, location, i]
                            print('searching cell: ', i)
                            # Bernoulli Trial
                            if np.random.binomial(1, detection_prob) == 1:
                                found = True
                                print('target found in cell: ', i)
                                op = np.zeros(len(Y_p))
                                op[i] = 1

                                found_cmap = colors.ListedColormap([(0,0,0,0), 'green', 'yellow', 'orange', 'purple'])
                                found_bounds = [0, .25, .5, .75, .9, 1]
                                found_norm = colors.BoundaryNorm(found_bounds, found_cmap.N)
                                yield [op.flatten(), found_norm, found_cmap]
                                break
                            else:
                                # Update probability
                                op[i] = ((1-dp[i])*cell) / (1-dp[i]*cell)

                                # For all other cells not currently being searched
                                for i2, cell2 in enumerate(op):
                                    if i2 != i:
                                        # Update probability
                                        op[i2] = cell2 / (1-dp[i]*cell)

                            # remove location from list to search
                            search_locations[:] = [x for x in search_locations if x[0] != coords[0] or x[1] != coords[1]]
                # target moves randomly
                # TODO: no teleporting
                move_options = [0, 1, -1, int(np.sqrt(len(Y_p))), int(-np.sqrt(len(Y_p)))]
                move = random.choice(move_options)
                Y_p[location] = 0
                location = location + move if location + move < len(Y_p) and location + move > 0 else location
                Y_p[location] = 1
                pause = True
            else:
                # generator has to have something to yield every time otherwise
                # the plot will freeze up, so here we're yielding the initial
                # probability distribution based on the search delay above
                interval = op.max() / 5
                bounds = [0, interval, interval*2, interval*3, interval*4, op.max()]
                norm = colors.BoundaryNorm(bounds, cmap.N)
                yield [op.flatten(), norm, cmap]

def plot_search(args):
    op, updated_norm, cmap = args

    new_z = op.reshape((len(grid),len(grid)))

    pdf.set_cmap(cmap)
    pdf.set_norm(updated_norm)
    pdf.set_data(new_z)
    return [pdf]

def rotate_coords(coords, theta):
    '''
    Rotate the coordinates about the origin

    *theta must be in radians*
    '''
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    rotated_coords = np.dot(rotation_matrix, coords)

    # translate x in positive direction the width of the grid because
    # we rotated about the origin instead of center
    rotated_coords[0] = rotated_coords[0] + (np.sqrt(len(grid.flatten())) - 1)
    print(rotated_coords)
    return rotated_coords

def convert_map_to_xy_coords(coords):
    X = np.linspace(extent[0], extent[1], int(np.sqrt(len(grid.flatten()))))
    Y = np.linspace(extent[2], extent[3], int(np.sqrt(len(grid.flatten()))))

    nearest_x_value = min(X, key=lambda x: abs(x-(coords[0])))
    nearest_y_value = min(Y, key=lambda y: abs(y-(coords[1])))
    xy_coords = [np.where(nearest_x_value == X)[0][0], np.where(nearest_y_value == Y)[0][0]]

    xy_coords = rotate_coords(xy_coords, 3*np.pi / 2) # rotate 270 degrees
    return xy_coords

def convert_xy_to_map_coords(coords):
    lat = np.linspace(extent[0], extent[1], int(np.sqrt(len(grid.flatten()))))
    lon = np.linspace(extent[3], extent[2], int(np.sqrt(len(grid.flatten()))))

    X = np.linspace(0, np.sqrt(len(grid.flatten()))-1, int(np.sqrt(len(grid.flatten()))))
    Y = np.linspace(0, np.sqrt(len(grid.flatten()))-1, int(np.sqrt(len(grid.flatten()))))

    nearest_x_value = min(X, key=lambda x: abs(x-(coords[0])))
    nearest_y_value = min(Y, key=lambda y: abs(y-(coords[1])))
    lat_lon = [lat[np.where(nearest_y_value == Y)[0][0]], lon[np.where(nearest_x_value == X)[0][0]]]

    return lat_lon

def onclick(event):
    global search_locations

    map_coords = (float(event.xdata), float(event.ydata))
    xy_coords = convert_map_to_xy_coords(map_coords)
    lat_lon = convert_xy_to_map_coords(xy_coords)
    print(lat_lon, is_land(lat_lon))
    search_locations.append(xy_coords)

# setup
found = False
search_locations = []
pause = True
start_time = datetime.datetime.utcnow()
fname = r'./GADM/gadm36_USA_2.shp'

# For determining land boundaries
geoms = fiona.open(fname)
land_geom = sgeom.MultiPolygon([sgeom.shape(geom['geometry']) for geom in geoms])
land = prep(land_geom)

grid = generate_grid()

# Plot setup
fig = plt.figure()
request = cimgt.GoogleTiles()
ax = plt.axes(projection=ccrs.PlateCarree())

shape_feature = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), edgecolor='black')
ax.add_feature(shape_feature, alpha=1)

extent = generate_extent()
ax.set_extent(extent)
ax.add_image(request, 10, interpolation='bilinear', zorder=2)

cmap = colors.ListedColormap([(0,0,0,0), 'green', 'yellow', 'orange', 'red'])
bounds = [0, .25, .5, .75, .9, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)
pdf = plt.imshow(np.zeros((21,21)), cmap=cmap, extent=extent, alpha=0.5, zorder=10)

ani = animation.FuncAnimation(fig, plot_search, frames=search, blit=False, repeat=False)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
#mng = plt.get_current_fig_manager()
#mng.full_screen_toggle()
plt.show()
fig.canvas.mpl_disconnect(cid)
