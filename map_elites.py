import math
import copy
from os import name
import numpy as np
from numpy.core.fromnumeric import resize
from image_utils.image_utils import load_image, image_grid
import matplotlib.pyplot as plt
# from util import get_compression_size

class MEMap:
    def __init__(self, config) -> None:
        self.resolution = config.map_elites_resolution
        self.max_novelty = config.map_elites_max_values[0]
        self.max_cxs = config.map_elites_max_values[1]
        self.min_novelty = config.map_elites_min_values[0]
        self.min_cxs = config.map_elites_min_values[1]
        self.map = [[None for i in range(self.resolution[1])] for j in range(self.resolution[0])]
        self.config = config
        
    def add(self, individual):
        novelty = individual.novelty
        cxs = len(list(individual.enabled_connections()))
        
        # novelty = get_compression_size(individual.image)
        # cxs = get_compression_size(individual.image)
        # cxs = get_get_sbi_results_count(individual.image)
        
        novelty_percent = ((novelty-self.min_novelty)/(self.max_novelty-self.min_novelty)) if self.max_novelty -self.min_novelty >1e-3 else 0
        novelty_index = int(novelty_percent * (self.resolution[0]-1))
        
        c_percent = (cxs-self.min_cxs)/(self.max_cxs-self.min_cxs) if self.max_cxs -self.min_cxs !=0 else 0
        # c_percent = (abs((len(individual.node_genome)+len(list(individual.enabled_connections()))-self.min_cxs))/(self.max_cxs-self.min_cxs)) if self.max_cxs -self.min_cxs !=0 else 0
        n_cxs_index = int(c_percent * (self.resolution[1]-1))
        
        # out of bounds, don't add to map
        if(novelty_index >= self.resolution[0] or novelty_index<0):
            return
        if(n_cxs_index >= self.resolution[1] or n_cxs_index<0):
            return

        if(self.map[novelty_index][n_cxs_index] == None):
            # empty cell, add
            self.map[novelty_index][n_cxs_index] = individual
        elif self.map[novelty_index][n_cxs_index].fitness < individual.fitness:
            # more fit than current elite, replace
            self.map[novelty_index][n_cxs_index] = individual
        else:
            # less fit than current elite, do nothing
            pass
    
    def random_non_empty_cell(self):
        output = None; 
        attempts = 0
        while output is None and attempts < 10000:
            i = np.random.randint(0, self.resolution[0])
            j = np.random.randint(0, self.resolution[1])
            output= self.map[i][j]
            attempts+=1
        if output is None:
            print("Failed to find individual in map")
            return Individual(self.config)            
            # raise Exception("Failed to find individual in map") # TODO
        return output

    def count_full_cells(self):
        count = 0
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                if self.map[i][j] is not None:
                    count+=1
        return count
    
    def set_cell_resolution(self, new_resolution):
        old_map = copy.copy(self.map)
        self.resolution = new_resolution
        self.map = [[None for i in range(self.resolution[1])] for j in range(self.resolution[0])] # clear map and change resolution
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                if(old_map[i][j]is not None):
                    self.add(old_map[i][j]) # re-add the individuals
    
    def set_min_values(self, new_mins):
        self.min_novelty = new_mins[0]
        self.min_cxs = new_mins[1]
        
    def set_max_values(self, new_maxes):
        self.max_novelty = new_maxes[0]
        self.max_cxs = new_maxes[1]
        # old_map = copy.copy(self.map)
        # self.map = [[None for i in range(self.resolution[1])] for j in range(self.resolution[0])] # clear map
        # for i in range(self.resolution[0]):
        #     for j in range(self.resolution[1]):
        #         if(old_map[i][j]is not None):
        #             self.add(old_map[i][j]) # re-add the individuals
    
    def show_2D_graph(self, real_values=False):
        # Xfull = range(self.resolution[0])
        # Yfull = range(self.resolution[1])
        if real_values:
            X = [self.map[ix][iy].novelty for ix, row in enumerate(self.map) for iy, i in enumerate(row) if i is not None]
            Y = [len(list(self.map[ix][iy].enabled_connections())) for ix, row in enumerate(self.map) for iy, i in enumerate(row) if i is not None]
            
            Xfull = np.linspace(self.min_novelty, self.max_novelty, self.resolution[0])
            Yfull = np.linspace(self.min_cxs, self.max_cxs, self.resolution[1])
            plt.xticks(np.linspace(self.min_novelty, self.max_novelty, self.resolution[0]))
            plt.yticks(np.linspace(self.min_cxs, self.max_cxs, self.resolution[1]))
            plt.xlabel(f"Novelty")
            plt.ylabel(f"N Connections")
            
        else:
            Xfull = range(self.resolution[0])
            Yfull = range(self.resolution[1])
            X = [(ix,iy)[0] for ix, row in enumerate(self.map) for iy, i in enumerate(row) if i is not None]
            Y = [(ix,iy)[1] for ix, row in enumerate(self.map) for iy, i in enumerate(row) if i is not None]
            plt.xlabel(f"Novelty cell index [{self.min_novelty:.3f}, {self.max_novelty:.3f}]")
            plt.ylabel(f"N Connections cell index [{self.min_cxs}, {self.max_cxs}]")
            plt.xticks(range(self.resolution[0]), labels=[f"{i} ({v:.1f})" for i, v in enumerate(np.linspace(self.min_novelty, self.max_novelty, self.resolution[0]))])
            plt.yticks(range(self.resolution[1]), labels=[f"{i} ({v:.1f})" for i, v in enumerate(np.linspace(self.min_cxs, self.max_cxs, self.resolution[1]))])
        
        xx, yy = np.meshgrid(Xfull, Yfull)
        plt.plot(xx, yy, marker='.', color='k', markersize=3,  linestyle='none')
        plt.plot(X, Y, marker='.', color='r', markersize=12, linestyle='none')
        
        
        # plt.xticks(range(self.resolution[0]))
        # plt.yticks(range(self.resolution[1]))
        plt.title ("MAP-Elites")
        plt.show()
        
        print(f"[min_cxs: {self.min_cxs}, max_cxs: {self.max_cxs}], [min_novelty: {self.min_novelty}, max_novelty: {self.max_novelty}]")
    def show_3D_graph(self):
        min_fit = np.min([[self.map[i][j].fitness if self.map[i][j] is not None else math.inf for i in range(self.resolution[0])] for j in range(self.resolution[1])])
        Xfull = range(self.resolution[0])
        Yfull = range(self.resolution[1])
        Z = np.array([[self.map[i][j].fitness if self.map[i][j] is not None else min_fit for i in range(self.resolution[0])] for j in range(self.resolution[1])])
        
        X, Y = np.meshgrid(Xfull, Yfull)    
        # Z = np.zeros((self.resolution[1], self.resolution[0]))
        # for i in range( self.resolution[0]):
        #     for j in range(self.resolution[1]):
        #         Z[j,i] = self.map[i][j].fitness if self.map[i][j] is not None else 0 
        
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
        
        ax.set_xlabel('Novelty cell index')
        ax.set_ylabel('N Connections cell index')
        ax.set_zlabel('Fitness')
        plt.title ("MAP-Elites")
        plt.show()
        
    def show_images(self, color_mode):
        example_image = [i.image for i in self.map[0] if i is not None][0]
        blank_img = np.ones(example_image.shape)
        images = np.array([[self.map[i][j].get_image() if self.map[i][j] is not None else blank_img for i in range(self.resolution[0])] for j in range(self.resolution[1])])
        image_grid(images, color_mode, "Novelty", "Connections")
        