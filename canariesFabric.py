import random

class Canaries:

    def __init__(self):

        self._formats = []
        self.formatSizeList = 100
        self.holesCounter = 4
        self._format = "The random number is: 28126"
        self.formatForTraining = "The random number is: 281265017"

    def randomGenerator(self):

        sequence = ""
        generatorCounter = 0

        for i in range(0, self.formatSizeList):
            while(generatorCounter<self.holesCounter):

                sequence+=(str(random.randint(0, 9)))
                generatorCounter+=1

            self._formats.append(self._format+sequence)
            sequence = ""
            generatorCounter=0
            
        print(self._formats)

    #To add: create format for training once and for log perplexity calculations for a lot of different opportuninties for ranks
    def getFormats(self)->list[str]:
    
        return self._formats
    
    property(getFormats)
