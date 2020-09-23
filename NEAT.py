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
        out = nn.computeOutput(imp[0])
        wrong = [0, 1, 2]
        wrong.pop(correctAnswer)
        s += out[correctAnswer] - (out[wrong[0]] + out[wrong[1]])
    for i in range(30):
        pixel = [random.random() * 255, random.random() * 255, random.random() * 255]
        correctAnswer = maxIndex(pixel)
        out = nn.computeOutput(pixel)
        wrong = [0, 1, 2]
        wrong.pop(correctAnswer)
        s += out[correctAnswer] - (out[wrong[0]] + out[wrong[1]])
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



class NeuralNetwork:
    def __init__(self, inputSize, outputSize):
        self.edges = []
        self.inputs = list(range(inputSize))
        self.outputs = list(range(inputSize, inputSize + outputSize))
        for inputNode in self.inputs:
            for outputNode in self.outputs:
                self.edges.append(Edge(inputNode, outputNode))
        self.highestNode = inputSize + outputSize - 1

    #NeuralNetwork.computeOutput(inputVector: number[]): number[]
    def computeOutput(self, inputVector):
        nodeValues = {}
        for i in range(len(inputVector)):
            nodeValues[i] = inputVector[i]
        flag = True
        while flag:
            i = len(inputVector) - 1
            canCalc = False
            while i in nodeValues.keys() or not canCalc:
                canCalc = True
                acc = 0
                for e in [a for a in self.edges if a.end == i]:
                    if e.start not in nodeValues.keys():
                        canCalc = False
                    else:
                        acc += nodeValues[e.start] * e.weight
                    # if e.start == 6:
                    #     print("BABAstart: {}\tend: {}\tweight: {}\t highest: {}\tnodeValues: {}\ti: {}\tbababa: {}\n".format(e.start, e.end, e.weight, self.highestNode, nodeValues, i, e.start in nodeValues.keys()))
                if i in nodeValues.keys() or not canCalc:
                    i += 1
                # print("i: {}\tcanCalc: {}".format(i, canCalc))
            # print(nodeValues.items())
            # acc = 0
            # for e in [a for a in self.edges if a.end == i]:
            #     if e.start == 6:
            #         print("start: {}\tend: {}\tweight: {}\t highest: {}\tnodeValues: {}\ti: {}\tbababa: {}\n".format(e.start, e.end, e.weight, self.highestNode, nodeValues, i, e.start in nodeValues.keys()))
            #     acc += nodeValues[e.start] * e.weight
            nodeValues[i] = acc
            # nodeValues[i] = reduce(self.edges, lambda acc, elem: (nodeValues[elem.start] * elem.weight) + acc if elem.end == i else acc, 0)
            if nodeValues[i] < 0:
                nodeValues[i] = 0
            flag = False
            for k in self.outputs:
                if k not in nodeValues.keys():
                    flag = True
        return [nodeValues[i] for i in self.outputs]

    def addNode(self):
        self.highestNode += 1
        edgeIndex = int(math.floor(len(self.edges) * random.random()))
        edge = self.edges.pop(random.choice(range(len(self.edges))))#self.edges.pop(edgeIndex)
        self.edges.append(Edge(edge.start, self.highestNode, edge.weight))
        self.edges.append(Edge(self.highestNode, edge.end, 1))

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
            for e in self.edges:
                if a == e.start and b == e.end:
                    remove = True

            if not remove:
                allPairsAcc.append((a, b))
        if len(allPairsAcc) == 0:
            self.mutateEdge()
        else:
            start, end = random.choice(allPairsAcc)#[math.floor(len(allPairsAcc) * random())]
            self.edges.append(Edge(start, end))

    def mutateEdge(self):
        edgeIndex = int(math.floor(len(self.edges) * random.random()))
        ALPHA = 0.5
        self.edges[edgeIndex].weight += ((2 * random.random()) - 1) * ALPHA

    def copy(self):
        out = NeuralNetwork(len(self.inputs), len(self.outputs))
        out.edges = [i.copy() for i in self.edges]
        out.highestNode = self.highestNode
        out.inputs = self.inputs
        out.outputs = self.outputs
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

class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        self.epsilon = 0.01 # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        self.q_table = {}
        self.canvas = None
        self.root = None

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        new_q = old_q
        
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q
        
    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        new_q = old_q
        
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q
        
    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if current_s not in self.q_table:
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable( current_r, current_s )

        self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            m = max(self.q_table[current_s])
            self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
            l = list()
            for x in range(0, len(self.actions)):
                if self.q_table[current_s][x] == m:
                    l.append(x)
            y = random.randint(0, len(l)-1)
            a = l[y]
            self.logger.info("Taking q action: %s" % self.actions[a])

        # try to send the selected action, only update prev_s if this succeeds
        try:
            agent_host.sendCommand(self.actions[a])
            self.prev_s = current_s
            self.prev_a = a

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0
        
        self.prev_s = None
        self.prev_a = None
        
        is_first_action = True
        
        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:

            current_r = 0
            
            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( current_r )
            
        self.drawQ()
    
        return total_reward
        
    def drawQ( self, curr_x=None, curr_y=None ):
        scale = 40
        world_x = 6
        world_y = 14
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x,y)
                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][action]
                    color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                    color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255-color, color, 0)
                    self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
                                             (y + action_positions[action][1] - action_radius ) *scale,
                                             (x + action_positions[action][0] + action_radius ) *scale,
                                             (y + action_positions[action][1] + action_radius ) *scale, 
                                             outline=color_string, fill=color_string )
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
                                     (curr_y + 0.5 - curr_radius ) * scale, 
                                     (curr_x + 0.5 + curr_radius ) * scale, 
                                     (curr_y + 0.5 + curr_radius ) * scale, 
                                     outline="#fff", fill="#fff" )
        self.root.update()

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