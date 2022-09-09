import math
import copy
from os import name
import time
from typing import Callable
import numpy as np
from numpy.core.fromnumeric import resize
from cppn_neat.cppn import CPPN
from cppn_neat.cppn import Node
from cppn_neat.evolutionary_algorithm import EvolutionaryAlgorithm
from cppn_neat.graph_util import name_to_fn
from image_utils.image_utils import load_image, image_grid
import matplotlib.pyplot as plt
from tqdm import trange
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
        self.GenomeClass = config.genome_type
        
    def add(self, individual):
        novelty = individual.novelty
        cxs = len(list(individual.enabled_connections()))
        
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
            return GenomeClass(self.config)            
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
        
        plt.title ("MAP-Elites")
        plt.show()
        
        print(f"[min_cxs: {self.min_cxs}, max_cxs: {self.max_cxs}], [min_novelty: {self.min_novelty}, max_novelty: {self.max_novelty}]")
    
    def show_3D_graph(self):
        min_fit = np.min([[self.map[i][j].fitness if self.map[i][j] is not None else math.inf for i in range(self.resolution[0])] for j in range(self.resolution[1])])
        Xfull = range(self.resolution[0])
        Yfull = range(self.resolution[1])
        Z = np.array([[self.map[i][j].fitness if self.map[i][j] is not None else min_fit for i in range(self.resolution[0])] for j in range(self.resolution[1])])
        
        X, Y = np.meshgrid(Xfull, Yfull)    

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
        
class MAPElites(EvolutionaryAlgorithm):
    def __init__(self, target, config, debug_output=False, genome_type=CPPN) -> None:
        self.gen = 0
        self.next_available_id = 0
        self.debug_output = debug_output
        self.all_species = []
        self.config = config
        Node.current_id =  self.config.num_inputs + self.config.num_outputs # reset node id counter
        self.show_output = True
        
        self.diversity_over_time = np.zeros(self.config.num_generations,dtype=float)
        self.population_over_time = np.zeros(self.config.num_generations,dtype=np.uint8)
        self.species_over_time = np.zeros(self.config.num_generations,dtype=np.float)
        self.species_threshold_over_time = np.zeros(self.config.num_generations, dtype=np.float)
        self.nodes_over_time = np.zeros(self.config.num_generations, dtype=np.float)
        self.connections_over_time = np.zeros(self.config.num_generations, dtype=np.float)
        self.fitness_over_time = np.zeros(self.config.num_generations, dtype=np.float)
        self.species_pops_over_time = []
        self.solutions_over_time = []
        self.species_champs_over_time = []
        self.time_elapsed = 0
        self.solution_generation = -1
        self.species_threshold = self.config.init_species_threshold
        self.population = []
        self.solution = None
        
        self.solution_fitness = -math.inf
        self.best_genome = None

        self.genome_type = genome_type
        
        self.fitness_function = config.fitness_function


        self.me_map = MEMap(self.config)
        
        if not isinstance(config.fitness_function, Callable):
            self.fitness_function = name_to_fn(config.fitness_function)
        self.fitness_function_normed = self.fitness_function
    
        self.target = target
    
    
    def evolve(self, run_number=1, show_output=True):
        self.start_time = time.time()
        try:
            self.run_number = run_number
            self.show_output = show_output or self.debug_output
            for i in range(self.config.population_size): # only create parents for initialization (the mu in mu+lambda)
                self.population.append(self.genome_type(self.config)) # generate new random individuals as parents
            
            # Run algorithm
            pbar = trange(self.config.num_generations, desc=f"Run {self.run_number}")
            self.update_fitnesses_and_novelty()
            self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True) # sort by fitness
            self.solution = self.population[0]
            
            for self.gen in pbar:
                self.run_one_generation()
                pbar.set_postfix_str(f"f: {self.get_best().fitness:.4f} d:{self.diversity_over_time[self.gen-1]:.4f}")
        except KeyboardInterrupt:
            raise KeyboardInterrupt()  
        self.end_time = time.time()     
        self.time_elapsed = self.end_time - self.start_time  
        
    def run_one_generation(self):
        if self.show_output:
            self.print_fitnesses()
        self.update_fitnesses_and_novelty()
        for g in self.population:
            self.me_map.add(g) # selection procedure
        