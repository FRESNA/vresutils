import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d

from vresutils.graph import *

# take ENTSO-E network, cut out Germany, partition it into network Voronoi cells,
# and plot the results

# adjust the data paths as necessary

entsoe_file = 'data/entsoe_2009_final.gpickle'
germany_file = 'data/germany.npy'

G = nx.read_gpickle(entsoe_file)
coor = np.load(germany_file)
polygon = Polygon(coor)

# include neighbours to avoid boundary effects in Voronoi algorithm
g = polygon_subgraph(G, polygon, nneighbours=3)
vor = voronoi_partition(g)

# cut off neighbours for plotting
vor = polygon_subgraph(vor, polygon, nneighbours=0)
pos = nx.get_node_attributes(vor, 'pos')

cl = []
for n,dat in vor.nodes(data=True):
    polyg = dat['region']
    x,y = polyg.exterior.coords.xy
    if min(x) < 6 or max(x) > 14.5 or min(y) < 47 or max(y) > 55: 
        continue
    color = plt.get_cmap('jet')(n)
    plt.plot(x,y, color=color)
    cl.append(color)

nx.draw(vor, pos=pos, node_color=cl)
plt.plot(coor[:,0], coor[:,1], color='r')
plt.show()
