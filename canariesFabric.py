import random

class Canaries:

    def __init__(self):

        self.formats = []
        self.formatSizeList = 100
        self.holesCounter = 6
        self.format = "The number is:"

    def randomGenerator(self):

        sequence = ""
        generatorCounter = 0

        for i in range(0, self.formatSizeList):
            while(generatorCounter<=self.holesCounter):

                sequence+=(str(random.randint(0, 9)))
                generatorCounter+=1

            self.formats.append(self.format+sequence)
            sequence = ""
            generatorCounter=0
        print(self.formats)
    
canaries = Canaries()
print(canaries.randomGenerator())

