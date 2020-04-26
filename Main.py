from Ag_Binario import Ag


def main():
    ag = Ag()
    ag.start()
    solution = ag.best_individual
    print(solution)
    
    ag.graph()

if __name__ == '__main__':
    main()