from random import randrange
import numpy as np
from math import cos,sqrt,sin,exp,pi

class Agent:
    def __init__(self, chromosome= None):
        self.dimension = 5
        self.chromosome = chromosome if chromosome else self.generateChromosome(15,self.dimension)
        self.fitness = None
        self.generation = None

    def generateChromosome(self, size: int, dimension: int):
        # size: number of bits used
        # dimension: dimensions of the function used
        options = '01'
        if dimension > 1:
            lista = []
            i = 0
            while i < dimension:
                binary = ''.join([options[randrange(len(options))] for i in range(size)])
                lista.append(binary)
                i+=1
            return lista
        binary = ''.join([options[randrange(len(options))] for i in range(size)])
        return binary

    def decode(self, binary:str):
        #binary: chromosome of the individual
        upper_limit = 32.768     #upper limit of function range
        bottom_limit = -32.768    #bottom limit of function range
        dec = []            #list that stores values ​​that have been converted to decimal
        #If the chromosome is larger than 1
        if len(binary) > 1:
            for i in range(len(binary)):
                #objective function converts bits to decimal
                objective_function = sum([int(binary[i][j])*2**(len(binary[i])-j-1) for j in range(len(binary[i]))])
                #equation: generates a decimal value in the defined interval
                equation = bottom_limit + objective_function*((upper_limit - bottom_limit)/(2**len(binary[i]) - 1))
                dec.append(equation)
        else:
            objective_function = sum([int(binary[i])*2**(len(binary)-i-1) for i in range(len(binary))])
            equation = bottom_limit + objective_function*((upper_limit - bottom_limit)/(2**len(binary) - 1))
            dec.append(equation)
        return dec

    def calculoFitness(self):
        """ ACKLEY FUNCTION """
        x= np.array(self.decode(self.chromosome))
        a,b,c = 20, 0.2, 2* np.pi
        
        return - (a * exp(-b*sqrt((1/self.dimension)* sum(x**2)))) - exp((1/self.dimension)*sum([cos(c*i) for i in x])) + a + exp(1)

    def copy(self, ):
        copy = Agent(chromosome = self.chromosome)
        copy.fitness = self.fitness
        copy.generation = self.generation
        
        return copy
    
    def __str__(self):
        return f'Generation: {self.generation}\
                \nBinary Chromosome: {self.chromosome}\
                \nDecimal Chromosome:{self.decode(self.chromosome)}\
                \nFitness: {self.fitness}'