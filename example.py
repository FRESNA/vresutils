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
polygon = Polygon(coor) # not a valid polygon, therefore the buffer fix
polygon = polygon.buffer(0) 

# include neighbours to avoid boundary effects in Voronoi algorithm
g = polygon_subgraph(G, polygon, nneighbours=0)
#nx.convert_node_labels_to_integers(g)
vor = voronoi_partition(g, polygon)

#plt.plot(coor[:,0], coor[:,1], color='r')

cl = []
check = True
for i,(n,dat) in enumerate(vor.nodes(data=True)):
    
    polyg = dat['region']
    pos = dat['pos']

    if not polyg.contains(Point(pos)):
        check = False

    x,y = polyg.exterior.coords.xy
    
    color = plt.get_cmap('jet')(i/222.)
    cl.append(color)
    plt.plot(x,y, color=color)

print check

pos = nx.get_node_attributes(vor, 'pos')
nx.draw(vor, pos=pos, node_color=cl)

plt.tight_layout()
plt.show()
