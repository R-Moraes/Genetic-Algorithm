from random import uniform,sample,randrange,random
import numpy as np
from matplotlib import pyplot as plt
from os import path
from Agent import Agent


class Ag:
    def __init__(self):
        self.stop_generation = 250
        self.best_individual = None
        self.size_population = 100
        self.crossover_rate = 0.85
        self.mutation_rate = 0.05
    
    def start(self):
        self.reset()
        population = self.generateInitialPopulation()
        self.evaluate(population)
        self.findBest(population, 0)
        self.save(population)
        
        for i in range(1, self.stop_generation):
            population = self.reproduction(population)
            self.evaluate(population)
            self.findBest(population, i)
            self.save(population)
        
    
    def generateInitialPopulation(self, ):
        return [Agent() for _ in range(self.size_population)]
    
    def evaluate(self, population:list):
        for indv in population:
            indv.fitness = indv.calculoFitness()
    
    def reproduction(self, population:list):
        pool = self.selection(population)
        new_population = self.crossover(pool)
        self.mutation(new_population)
        population.sort(key=lambda individuo: individuo.fitness)
        percent = int(self.size_population * self.crossover_rate)
        percent = percent if percent % 2 == 0 else percent + 1

        return new_population + population[percent:]
    
    def selection(self, population:list):
        pool = []
        amount = 3
        percent = int(self.size_population * self.crossover_rate)
        percent = percent if percent % 2 == 0 else percent + 1

        for _ in range(percent):
            selected = sample(population, amount)
            selected.sort(key=lambda individuo: individuo.fitness)
            winner = selected[0]
            pool.append(winner)
        
        return pool

    def crossover(self, pool):
        size = len(pool)
        new_pop = []

        percent = int(self.size_population * self.crossover_rate)
        percent = percent if percent % 2 == 0 else percent + 1
        for _ in range(0,percent,2):
            indv = pool[randrange(size)]
            indv2 = pool[randrange(size)]

            crom,crom2 = self.crossoverUniform(indv.chromosome,indv2.chromosome)
            new_pop.append(Agent(chromosome = crom))
            new_pop.append(Agent(chromosome = crom2))
        
        return new_pop

    def crossoverOnePoint(self, seq1:list,seq2:list):
        p_seq1 = randrange(len(seq1[0]))
        p_seq2 = p_seq1
        seq12, seq21 = [],[]
        for i in range(len(seq1)):
            seq12.append(seq1[i][:p_seq1] + seq2[i][p_seq2:])
            seq21.append(seq2[i][:p_seq2] + seq1[i][p_seq1:])

        return (seq12,seq21)
    
    def crossoverTwoPoint(self, seq1:list, seq2:list):
        p_seq1 = sorted([randrange(len(seq1[0])), randrange(len(seq1[0]))])
        
        seq12, seq21 = [], []
        for i in range(len(seq1)):
            seq12.append(seq1[i][:p_seq1[0]] + seq2[i][p_seq1[0]:p_seq1[1]] + seq1[i][p_seq1[1]:])
            seq21.append(seq2[i][:p_seq1[0]] + seq1[i][p_seq1[0]:p_seq1[1]] + seq2[i][p_seq1[1]:])
        
        return (seq12,seq21)

    def crossoverUniform(self, seq1:list, seq2:list):
        ps = 0.5
        c,d = [],[]

        count = 0
        while count < len(seq1):
            bit_c = ''
            bit_d = ''
            for i in range(len(seq1[0])):
                if random() < ps:
                    bit_c += seq2[count][i]
                    bit_d += seq1[count][i]
                else:
                    bit_c += seq1[count][i]
                    bit_d += seq2[count][i]
            c.append(bit_c)
            d.append(bit_d)
            count += 1
        return (c,d)
    
    def mutation(self, population):
        for indiv in population:
            mutation = random() < self.mutation_rate

            if mutation:
                size = len(indiv.chromosome)

                n1,n2 = sample([ i for i in range(size)], 2)

                get = lambda p: indiv.chromosome[n2] if p == n1 else indiv.chromosome[n1]

                indiv.chromosome = [gene if n1 != i != n2 else get(i) for i, gene in enumerate(indiv.chromosome)]

    def findBest(self, population, i):
        population.sort(key=lambda individuo : individuo.fitness)
        best = population[0]
        best.generation = i

        if not self.best_individual:
            self.best_individual = best.copy()
        
        if best.fitness < self.best_individual.fitness:
            self.best_individual = best.copy()
            
    def graph(self):
        x = np.arange(self.stop_generation)

        best = np.ndarray((0))
        worse = np.ndarray((0))
        average = np.ndarray((0)) 
        with open(path.abspath(f'file_Ag/generation.npy'), 'rb') as file:
            for _ in range(self.stop_generation):
                all_fitness = np.load(file)
                best = np.append(best, min(all_fitness))
                average = np.append(average, all_fitness.mean())
                worse = np.append(worse, max(all_fitness))
        best.sort()
        worse.sort()
        label = ['best', 'worse', 'average']
        data = [best[::-1], worse, average]
        for l,y in zip(label, data):
            plt.plot(x,y,label = l) 
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save(self, population):
        all_fitness = np.array([ indv.fitness for indv in population])
        with open(path.abspath(f'file_Ag/generation.npy'), 'ab+') as file:
            np.save(file, all_fitness)
    
    def reset(self):
        open(path.abspath(f'file_Ag/generation.npy'), 'wb').close()



