# Read graphs from files

def readNodesCoordinates(file):
    with open(file) as f:
        matrix = []
        for line in f:
            line = line.split()
            matrix.append((int(line[1]), int(line[2]), int(line[3]), line[0]))
        return matrix

def readEdgesList(file):
    with open(file) as f:
        lines = f.read().splitlines()
        matrix = []
        for line in lines:
            row = []
            for value in line.split():
                row.append(int(value))
            matrix.append(row)
        return matrix
