import math
import random
#reduce<T, E>: (arr: T[], f: (E, T) => E, initValue: E): E
def reduce(arr, f, initValue):
    if len(arr) == 0:
        return initValue
    else:
        return reduce(arr[1:], f, f(initValue, arr[0]))


#maxIndex(arr: number[]): integer
def maxIndex(arr):
    index = 0
    for i in range(len(arr)):
        if arr[index] < arr[i]:
            index = i
    return index

#fitness(nn: NeuralNetwork): number
def fitness(nn):
    s = 0
    importantTests = [([255, 0, 0], 0), ([0, 255, 0], 1), ([0, 0, 255], 2)]
    for imp in importantTests:
        correctAnswer = imp[1]
        out = maxIndex(nn.computeOutput(imp[0]))
        if out == correctAnswer:
        	s += 3
        else:
        	s -= 3
        # wrong = [0, 1, 2]
        # wrong.pop(correctAnswer)
        # s += out[correctAnswer] - (out[wrong[0]] + out[wrong[1]])
    for i in range(200):
        pixel = [random.random() * 255, random.random() * 255, random.random() * 255]
        correctAnswer = maxIndex(pixel)
        out = maxIndex(nn.computeOutput(pixel))
        if out == correctAnswer:
        	s += 3
        else:
        	s -= 3
        # wrong = [0, 1, 2]
        # wrong.pop(correctAnswer)
        # s += out[correctAnswer] - (out[wrong[0]] + out[wrong[1]])
    return s
    # return random() #for now i guess

class Generation:
    def __init__(self, genSize, inputSize=None, outputSize=None, prev=None):
        self.nns = []
        self.genSize = genSize
        if prev == None:
            for i in range(self.genSize):
                self.nns.append(NeuralNetwork(inputSize=inputSize, outputSize=outputSize))
        else:
            bestPrev = [(i, fitness(i)) for i in prev.nns]
            bestPrev.sort(key=lambda x: -1 * x[1])
            # print(list(map(lambda x: x[1], bestPrev)))
            bestPrev = list(map(lambda x: x[0], bestPrev))[:int(self.genSize / 4)]
            for i in bestPrev:
                c = i.copy()
                c.mutate()
                self.nns.append(c)
                c = i.copy()
                c.mutate()
                self.nns.append(c)
                c = i.copy()
                c.mutate()
                self.nns.append(c)
                c = i.copy()
                c.mutate()
                self.nns.append(c)

class DependObj:
    def __init__(self, nodeNum, dependsOn, allDependants):
        self.nodeNum = nodeNum
        self.dependsOn = dependsOn
        self.allDependants = allDependants
    def __str__(self):
        return "<NodeNum: {}, dependsOn: {}, allDependants: {}>".format(self.nodeNum, [a.start for a in self.dependsOn], self.allDependants)
    def __repr__(self):
        return self.__str__()

def largest(arr):
    if len(arr) == 0:
        return -1 * math.inf
    else:
        return max(arr)

class NeuralNetwork:
    def __init__(self, inputSize, outputSize):
        self.edges = []
        self.inputs = list(range(inputSize))
        self.outputs = list(range(inputSize, inputSize + outputSize))
        self.depend = [DependObj(i, [], []) for i in (self.inputs + self.outputs) ]
        for inputNode in self.inputs:
            for outputNode in self.outputs:
                e = Edge(inputNode, outputNode)
                self.edges.append(e)
                self.depend[outputNode].dependsOn.append(e)
                self.depend[outputNode].allDependants.append(inputNode)
        #This is makes it go super speedy
        self.depend.sort(key=lambda x: len(x.allDependants))
        self.highestNode = inputSize + outputSize - 1

    #NeuralNetwork.computeOutput(inputVector: number[]): number[]
    def computeOutput(self, inputVector):
        nodeValues = {}
        for i in range(len(inputVector)):
            nodeValues[i] = inputVector[i]
        for dependObj in self.depend:
            if dependObj.nodeNum in self.inputs:
                continue
            acc = 0
            for e in dependObj.dependsOn:
                acc += nodeValues[e.start] * e.weight
            nodeValues[dependObj.nodeNum] = acc
        return [nodeValues[i] for i in self.outputs]

    def addNode(self):
        self.highestNode += 1
        edgeIndex = int(math.floor(len(self.edges) * random.random()))
        edge = self.edges.pop(random.choice(range(len(self.edges))))#self.edges.pop(edgeIndex)
        newE1 = Edge(edge.start, self.highestNode, edge.weight)
        newE2 = Edge(self.highestNode, edge.end, 1)
        self.edges.append(newE1)
        self.edges.append(newE2)
        for dependObj in self.depend:
            if dependObj.nodeNum == edge.end:
                dependObj.dependsOn = [newE2] + [u for u in dependObj.dependsOn if u.start != edge.start]
                dependObj.allDependants = [self.highestNode] + dependObj.allDependants 
            elif edge.end in dependObj.allDependants:
                dependObj.allDependants.append(self.highestNode)
        for dependObj in self.depend:
            if dependObj.nodeNum == newE1.start:
                self.depend.append(DependObj(newE1.end, [newE1], [newE1.start] + dependObj.allDependants))
        self.depend.sort(key=lambda x: len(x.allDependants))

    def addEdge(self):
        allPairs = []
        for i in range(self.highestNode + 1):
            for j in range(i, self.highestNode + 1):
                if i != j:
                    allPairs.append((i, j))
        allPairsAcc = []
        for i in range(len(allPairs)):
            a, b = allPairs[i]
            remove = False
            if a in self.outputs or b in self.inputs:
                remove = True
            if not remove:
                for dependObj in self.depend:
                    if dependObj.nodeNum == a and b in dependObj.allDependants:
                        remove = True
                        break
            if not remove:
                for e in self.edges:
                    if a == e.start and b == e.end:
                        remove = True
                        break
            if not remove:
                allPairsAcc.append((a, b))
        if len(allPairsAcc) == 0:
            self.mutateEdge()
        else:
            start, end = random.choice(allPairsAcc)
            e = Edge(start, end)
            self.edges.append(e)
            startAllDependants = []
            for dependObj in self.depend:
                if dependObj.nodeNum == start:
                    startAllDependants = dependObj.allDependants
                    break
            for dependObj in self.depend:
                if dependObj.nodeNum == e.end:
                    for toAdd in startAllDependants:
                        if toAdd not in dependObj.allDependants:
                            dependObj.allDependants.append(toAdd)
                    if start not in dependObj.allDependants:
                    	dependObj.allDependants.append(start)
                    dependObj.dependsOn.append(e)
                elif e.end in dependObj.allDependants:
                    for toAdd in startAllDependants:
                        if toAdd not in dependObj.allDependants:
                            dependObj.allDependants.append(toAdd)
                    if start not in dependObj.allDependants:
                    	dependObj.allDependants.append(start)
            self.depend.sort(key=lambda x: len(x.allDependants))



    def mutateEdge(self):
        edgeIndex = int(math.floor(len(self.edges) * random.random()))
        ALPHA = 0.5
        d = random.choice(self.depend)
        while len(d.dependsOn) == 0:
        	d = random.choice(self.depend)
        random.choice(d.dependsOn).weight += ((2 * random.random()) - 1) * ALPHA

    def copy(self):
        out = NeuralNetwork(len(self.inputs), len(self.outputs))
        out.edges = [i.copy() for i in self.edges]
        out.highestNode = self.highestNode
        out.inputs = self.inputs
        out.outputs = self.outputs
        out.depend = [DependObj(a.nodeNum, [i.copy() for i in a.dependsOn], [i for i in a.allDependants]) for a in self.depend]
        out.depend.sort(key=lambda x: len(x.allDependants))
        return out

    def mutate(self):
        rnJesus = random.random()
        if rnJesus < 0.10:
            self.addNode()
        elif rnJesus < 0.30:
            self.addEdge()
        else:
            self.mutateEdge()

class Edge:
    def __init__(self, start, end, weight=None):
        self.start = start
        self.end = end
        self.weight = weight
        if weight == None:
            self.weight = (random.random() * 2) - 1

    #Edge.copy(void): Edge
    def copy(self):
        return Edge(self.start, self.end, self.weight)

    def __str__(self):
    	return "<{} to {} with weight {}>".format(self.start, self.end, self.weight)

    def __repr__(self):
    	return str(self)

gen = Generation(80, inputSize=3, outputSize=3)
fitnesses = [fitness(i) for i in gen.nns]
print("Initial fitness: {}".format(sum(fitnesses) / len(fitnesses)))
for i in range(int(input("How many generations: ")) - 1):
	gen = Generation(80, prev=gen)
	print("finished generation {}.".format(i + 2))
fitnesses = [fitness(i) for i in gen.nns]
print("Final fitness: {}".format(sum(fitnesses) / len(fitnesses)))
while True:
    string = input("Give me a pixel (three numbers seperated by spaces that are between 0 and 255): ")
    pixel = list(map(lambda x: int(x), string.split()))
    allOutputs = [i.computeOutput(pixel) for i in gen.nns]
    final = [0, 0, 0]
    for a in allOutputs:
        final[0] += a[0]
        final[1] += a[1]
        final[2] += a[2]
    print(final)
    maxI = maxIndex(final)
    if maxI == 0:
        print("That is mostly red.\n\n")
    elif maxI == 1:
        print("That is mostly green.\n\n")
    else:
        print("That is mostly blue.\n\n")
exit()