import numpy as np

class mesh2d:
    def __init__(self, filename):
        self.filename = filename
        self.Nnodes = 0
        self.Nodes = None

    def read_mesh(self):
        with open(self.filename, 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('$Nodes'):
                    self.read_nodes(f)
                if line.startswith('$Elements'):
                    self.read_elements(f)

                line = f.readline()

    def read_nodes(self, f):
        # Read initial header
        line = f.readline().split()
        num_blocks, self.Nnodes, _, _ = map(int, line)
        self.Nodes = np.empty((self.Nnodes, 2), dtype=np.float64)  # Prepare to store x, y, z coordinates
        tag = 0
        while tag < self.Nnodes:
            # Read the block header, ignoring parametric details
            line = f.readline().split()
            
            if len(line) == 4:
                num_nodes_in_block = int(line[3])
                block_tag = int(line[1])


            elif len(line) == 1:
                elem_tag = line


            elif len(line) == 3:
                tag += 1
                x = np.float64(line[0])
                y = np.float64(line[1])
                self.Nodes[tag-1] = [x, y]

            elif len(line) == 5:
                break
    

    def read_elements(self, f):
        # Read initial header
        line = f.readline().split()
        num_blocks, self.Nel, _, _ = map(int, line)
        nb = self.Nel
        self.connect = np.empty((nb, 3), dtype=int)
        self.area = np.zeros(nb, dtype=float)
        self.diam = np.zeros(nb, dtype=float)
        self.label = np.zeros((self.Nnodes), dtype=int) # Contient 0 si ce n'est pas un noeud de bord

        tag = 0
        while tag < self.Nel:
            # Read the block header
            line = f.readline()
            if line.startswith('$EndElements'):  
                break

            # On parcours les éléments 1d et 2d:
            line = line.split()
            dim = int(line[0])
            if len(line) == 4 and dim == 1: # Les élements 1d sont les lignes qui composent les bords
                num_nodes_in_block = int(line[3])
                block_tag = int(line[1])
                nb = num_nodes_in_block # Nombre des éléments 1d                
                tag_1d = 0
                while tag_1d < num_nodes_in_block: 
                    line = f.readline().split()
                    tag_1d += 1
                    a = int(line[1])
                    b = int(line[2])
                    self.label[a-1] = 1
                    self.label[b-1] = 1
                    x1, y1 = self.Nodes[a-1]
                    x2, y2 = self.Nodes[b-1]

                self.N1del = num_nodes_in_block



            if len(line) == 4 and dim == 2:
                num_nodes_in_block = int(line[3])
                block_tag = int(line[1])
                nb = num_nodes_in_block # Nombre des élément 2d
                self.connect = np.empty((nb, 3), dtype=int)
                self.area = np.zeros(nb, dtype=float)
                self.diam = np.zeros(nb, dtype=float)
                tag_2d = 0
                while tag_2d < num_nodes_in_block:
                    line = f.readline().split()
                    tag_2d += 1
                    a = int(line[1])
                    b = int(line[2])
                    c = int(line[3])
                    self.connect[tag_2d-1] = [a, b, c] # Contient les tags des noeuds du maillage
                    x1, y1 = self.Nodes[a-1]
                    x2, y2 = self.Nodes[b-1]
                    x3, y3 = self.Nodes[c-1]
                    
                    # Calculer les distances entre chaque paire de nœuds
                    distances = [
                        np.sqrt((x2 - x1)**2 + (y2 - y1)**2),
                        np.sqrt((x3 - x2)**2 + (y3 - y2)**2),
                        np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
                    ]
                    
                    # Le diamètre est la plus longue des distances
                    self.diam[tag_2d-1] = max(distances)
                    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                    self.area[tag_2d-1] = area
                    
                # Le paramètre h est le maximum des diamètres
                self.h = np.max(self.diam)
                self.N2del = num_nodes_in_block
    

    def coord(self, lambdas):
        lambdas = np.array(lambdas)
        vertices = np.array(vertices)
        point = np.dot(lambdas, vertices)
        
        return tuple(point)

    
            


if __name__ == '__main__':
    filename = "/workspaces/2024-m1-scimba-feelpp/tools/feelppdb/feelpp_cfpde/np_1/omega-2.msh"  
    my_mesh = mesh2d(filename)
    my_mesh.read_mesh()
    
    print("Number of nodes:", my_mesh.Nnodes)
    print("Nodes coordinates:", my_mesh.Nodes)
    print('Number of elements', my_mesh.Nel)
    print("\n connect = ", my_mesh.connect)
    print("\n diam = ", my_mesh.diam)
    print("\n area = ", my_mesh.area)
 

