# # Homework 2 Evolutionary Algorithms.
# This file will include all the necessary code for evolutionary algorithm homework. The task is described as "In this homework, you will perform experiments on evolutionary algorithm and draw conclusions from the experimental results. The task is to create an image made of filled circles, visually similar to a given RGB source image (painting.png)"
# 
# ## Pseudo code
# Initialize population with <num_inds> individuals each having <num_genes> genes
# While not all generations (<num_generations>) are computed:
# Evaluate all the individuals
# Select individuals
# Do crossover on some individuals
# Mutate some individuals

# "Individual" class definition for evolutionary algoritm. There will be on chromosome and N number of genes. Each gene will have center coordinates (x,y), Radius, and RGB color.

import random
import numpy as np
import cv2
import copy
import h5py
from tqdm import tqdm


# Individual class definition
class Individual:
    def __init__(self, num_genes, image_size):
        self.num_genes = num_genes
        self.image_size = image_size
        self.chromosome = []
        self.fitness = np.float128(0.0)
        self.elite = False
        self.radius_max = image_size[0]//4

        for i in range(num_genes):
            x = random.randint(-1*self.radius_max, self.image_size[0]+1*self.radius_max)
            y = random.randint(-1*self.radius_max, self.image_size[1]+1*self.radius_max)
            r = random.randint(1, self.radius_max)
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            alpha = random.random()

            self.chromosome.append([x, y, r, color, alpha])


    def is_visible(self, gene=None):
        x = gene[0]
        y = gene[1]
        r = gene[2]
        if ((x + r) < 0) or (x  > self.image_size[0]+r) or ((y + r) < 0) or (y > self.image_size[1]+r):
            return False
        else:
            return True
    def mutate(self, mutation_probability, guidance = None):
        if not self.elite:
            if random.random() < mutation_probability:
                while True:
                    i = random.randint(0, self.num_genes-1)

                    if guidance is "unguided":
                        while True:
                            if not self.is_visible(self.chromosome[i]):
                                # randomly initialize the gene and check again
                                self.chromosome[i][0] = random.randint(-2*self.radius_max, self.image_size[0]+2*self.radius_max)
                                self.chromosome[i][1] = random.randint(-2*self.radius_max, self.image_size[1]+2*self.radius_max)
                                self.chromosome[i][2] = random.randint(0, self.radius_max*2)
                            else:
                                break
                        
                        self.chromosome[i][3][0] = random.randint(0, 255)
                        self.chromosome[i][3][1] = random.randint(0, 255)
                        self.chromosome[i][3][2] = random.randint(0, 255)

                        self.chromosome[i][4] = random.random()
                    else:
                        #Guided mutation, deviate x,y, radius, color and alpha around their previous values
                        
                        temp_x = copy.deepcopy(self.chromosome[i][0])
                        temp_y = copy.deepcopy(self.chromosome[i][1])
                        temp_r = copy.deepcopy(self.chromosome[i][2])
                        
                        # mutate under the condition of visibility again and again until the gene is visible
                        while True:
                            self.chromosome[i][0] = int(temp_x + (self.image_size[0]/4)*random.uniform(-1,1))
                            self.chromosome[i][1] = int(temp_y + (self.image_size[1]/4)*random.uniform(-1,1))

                            if self.chromosome[i][0] < -2*self.radius_max:
                                self.chromosome[i][0] = -2*self.radius_max
                            elif self.chromosome[i][0] > self.image_size[0]+2*self.radius_max:
                                self.chromosome[i][0] = self.image_size[0]+2*self.radius_max
                            
                            if self.chromosome[i][1] < -2*self.radius_max:
                                self.chromosome[i][1] = -2*self.radius_max
                            elif self.chromosome[i][1] > self.image_size[1]+2*self.radius_max:
                                self.chromosome[i][1] = self.image_size[1]+2*self.radius_max
                            
                            self.chromosome[i][2] = int(temp_r + 10*random.uniform(-1,1))
                            if self.chromosome[i][2] < 0:
                                self.chromosome[i][2] = 1
                            elif self.chromosome[i][2] > self.radius_max*2:
                                self.chromosome[i][2] = self.radius_max*2

                            if self.is_visible(self.chromosome[i]):
                                break

                        self.chromosome[i][3][0] = int(self.chromosome[i][3][0] + 64*random.uniform(-1, 1))
                        if self.chromosome[i][3][0] < 0:
                            self.chromosome[i][3][0] = 0
                        elif self.chromosome[i][3][0] > 255:
                            self.chromosome[i][3][0] = 255

                        self.chromosome[i][3][1] = int(self.chromosome[i][3][1] + 64*random.uniform(-1, 1))
                        if self.chromosome[i][3][1] < 0:
                            self.chromosome[i][3][1] = 0
                        elif self.chromosome[i][3][1] > 255:
                            self.chromosome[i][3][1] = 255
                        
                        self.chromosome[i][3][2] = int(self.chromosome[i][3][2] + 64*random.uniform(-1, 1))
                        if self.chromosome[i][3][2] < 0:
                            self.chromosome[i][3][2] = 0
                        elif self.chromosome[i][3][2] > 255:
                            self.chromosome[i][3][2] = 255

                        self.chromosome[i][3] = [int(x) for x in self.chromosome[i][3]]
                        
                        self.chromosome[i][4] = self.chromosome[i][4] + 0.25*random.uniform(-1, 1)
                        if self.chromosome[i][4] < 0:
                            self.chromosome[i][4] = 0.001
                        elif self.chromosome[i][4] > 1:
                            self.chromosome[i][4] = 1
                        
                    if random.random() > mutation_probability:
                        break
            else:
                pass
        else:
            # print("Cannot mutate elite individual")
            pass
    def draw(self):
        # First sort the genes by radius
        self.chromosome.sort(key=lambda x: x[2], reverse=True)
        
        # Create a blank image white background
        image = np.ones((self.image_size[1], self.image_size[0], 3), np.uint8)*255
        
        for i, gene in enumerate(self.chromosome):
            #check if the circle is visible in the image, center does not have to be in the image but the circle should be visible
            while True:
                if not self.is_visible(gene):
                    # randomly initialize the gene and check again
                    gene[0] = random.randint(-2*self.radius_max, self.image_size[0]+2*self.radius_max)
                    gene[1] = random.randint(-2*self.radius_max, self.image_size[1]+2*self.radius_max)
                    gene[2] = random.randint(0, self.radius_max*2)
                else:
                    break
            self.chromosome[i] = gene
            overlay = image.copy()
            cv2.circle(overlay, (gene[0], gene[1]), gene[2], gene[3], -1)
            image = cv2.addWeighted(overlay, gene[4], image, 1 - gene[4], 0)
        return image

    def calculate_fitness(self, target):
        image = self.draw()
        # Calculate the difference between the target image and the generated image
        target_np = np.array(target, dtype=np.int64)
        image_np = np.array(image, dtype=np.int64)
        diff = np.subtract(target_np, image_np)
        # take the square of the difference
        diff = np.square(diff)
        # sum of the squared differences
        # print(np.sum(diff))
        self.fitness = -np.sum(diff)

    def crossover(self, partner):
        child1 = Individual(self.num_genes, self.image_size)
        child2 = Individual(self.num_genes, self.image_size)

        for i in range(self.num_genes):
            if random.random() < 0.5:
                child1.chromosome[i] = copy.deepcopy(self.chromosome[i])
                child2.chromosome[i] = copy.deepcopy(partner.chromosome[i])
            else:
                child1.chromosome[i] = copy.deepcopy(partner.chromosome[i])
                child2.chromosome[i] = copy.deepcopy(self.chromosome[i])
        return child1, child2

# Popoulation class definition

class Population:
    def __init__(self, num_individuals, num_genes, image_size, frac_elites, frac_parents, tm_size, target_image, guidance):
        self.num_individuals = num_individuals
        self.num_genes = num_genes
        self.image_size = image_size
        self.num_elites = int(frac_elites*self.num_individuals)
        self.num_parents = int(frac_parents*self.num_individuals)
        if self.num_parents % 2 != 0:
            self.num_parents += 1
        self.tm_size = tm_size
        self.guidance = guidance

        self.target = target_image
        self.individuals = []
        self.parents = []

        for i in range(self.num_individuals):
            self.individuals.append(Individual(self.num_genes, self.image_size))

    def selection(self):
        
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

        new_individuals = []
        # Mark the best individuals as elite
        for i in range(self.num_elites):
            self.individuals[i].elite = True
            new_individuals.append(self.individuals[i])
        # Select the rest of the individuals using tournament selection.
        non_elite_individuals = self.individuals[self.num_elites:]
        
        parentable_individuals = []

        # update the non elite group by adding the best individual from each group to parentable individuals
        
        for i in range(len(non_elite_individuals)):
            group = random.sample(non_elite_individuals, min(self.tm_size, len(non_elite_individuals)))
            
            group.sort(key=lambda x: x.fitness, reverse=True)

            parentable_individuals.append(copy.deepcopy(group[0]))
        
        # select the best parents from the parentable individuals
        parentable_individuals.sort(key=lambda x: x.fitness, reverse=True)
        self.parents = parentable_individuals[:self.num_parents]
        # non elite non parent elements will be added to the new individuals
        new_individuals.extend(parentable_individuals[self.num_parents:])
        # The new generation except the children will be the new individuals
        self.individuals = new_individuals
        
    def crossover(self):
        # parents will create new individuals by crossover. Two parents will create two children
        children = []
        random.shuffle(self.parents)

        for i in range(0, self.num_parents, 2):
            parent1 = self.parents[i]
            parent2 = self.parents[i+1]

            child1, child2 = parent1.crossover(parent2)

            child1.calculate_fitness(self.target)
            child2.calculate_fitness(self.target)
            
            children.append(child1)
            children.append(child2)
        
        self.individuals.extend(children)

    def mutation(self, mutation_probability):
        #check if the individual is an elite, if so do not mutate
        for individual in self.individuals: 
            individual.mutate(mutation_probability, guidance = self.guidance)

    def evaluate(self):
        for individual in self.individuals:
            individual.calculate_fitness(self.target)

    def get_best(self):
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        return self.individuals[0]

    def get_average_fitness(self):
        return sum([x.fitness for x in self.individuals]) / self.num_individuals




# default parameters in dictionary
parameters_list = [{
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 5,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 10,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 40,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 60,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 15,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 30,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 80,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 120,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 2,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 8,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 16,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.04,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.35,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
}, {
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.15,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
},{
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.3,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
},{
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.75,
    "mutation_probability": 0.2,
    "mutataion_type": "guided"
},{
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.1,
    "mutataion_type": "guided"
},{
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.4,
    "mutataion_type": "guided"
},{
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.75,
    "mutataion_type": "guided"
},{
    "num_individuals": 20,
    "num_genes": 50,
    "tournament_size": 5,
    "frac_elites": 0.2,
    "frac_parents": 0.6,
    "mutation_probability": 0.2,
    "mutataion_type": "unguided"
}
]

for i, parameters in enumerate(parameters_list):
    num_individuals = parameters["num_individuals"]
    num_genes = parameters["num_genes"]
    tournament_size = parameters["tournament_size"]
    frac_elites = parameters["frac_elites"]
    frac_parents = parameters["frac_parents"]
    mutation_probability = parameters["mutation_probability"]
    mutataion_type = parameters["mutataion_type"]

    print("Parameters for run ", i+1)

    print("Summary of parameters")
    print("Number of individuals: ", num_individuals)
    print("Number of genes per individual: ", num_genes)
    print("Tournament size: ", tournament_size)
    print("Fraction of elites: ", frac_elites)
    print("Fraction of parents: ", frac_parents)
    print("Mutation probability: ", mutation_probability)
    print("Mutation type: ", mutataion_type)

    # load input image
    target = cv2.imread("hw2/painting.png")

    image_size = (target.shape[1], target.shape[0])


    pop = Population(num_individuals, num_genes, image_size, frac_elites, frac_parents, tournament_size, target,guidance=mutataion_type)


    # iterate over generations
    average_fitness = []
    best_fitness = []
    image_of_best = []

    num_generations = 10000

    for i in tqdm(range(num_generations)):
        pop.evaluate()
        pop.selection()
        pop.crossover()
        pop.mutation(mutation_probability)
        pop.evaluate()
        #reset elite status
        for individual in pop.individuals:
            individual.elite = False
        # print("num individuals left",len(pop.individuals))
        average_fitness.append(pop.get_average_fitness())
        best_fitness.append(pop.get_best().fitness)
        if (i+1) % 1000 == 0:
            image_of_best.append(pop.get_best().draw())
    print("Generation: ", i, "Average fitness: ", pop.get_average_fitness(), "Best fitness: ", pop.get_best().fitness)

    # set filename using the parameters

    filename = "hw2/out_sweep/output_" + str(num_individuals) + "_" + str(num_genes) + "_" + str(tournament_size) + "_" + str(frac_elites) + "_" + str(frac_parents) + "_" + str(mutation_probability) + "_" + mutataion_type + ".h5"

    # save images and average best fitness values to a file
    with h5py.File(filename, "w") as f:
        f.create_dataset("average_fitness", data=average_fitness)
        f.create_dataset("best_fitness", data=best_fitness)
        for i, image in enumerate(image_of_best):
            f.create_dataset("image_"+str(i-1000), data=image)



