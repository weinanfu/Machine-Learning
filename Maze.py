import sys

PART_OF_PATH = '+'
TRIED = '.'
OBSTACLE = '1'
DEAD_END = '-'

class Maze:
    path = []
    def __init__(self, mazeFileName):
        columnsInMaze = 0
        self.mazelist = []
        mazeFile = open(mazeFileName, 'r')
        rowsInMaze = 0
        for line in mazeFile:
            rowList = []
            col = 0
            for ch in line:
                if ch == '1' or ch == '0':
                    rowList.append(ch)
                    col = col + 1
            rowsInMaze = rowsInMaze + 1
            self.mazelist.append(rowList)
            columnsInMaze = len(rowList)
        self.startRow, self.startCol = map(int, input("Start: ").split(','))
        self.endRow, self.endCol = map(int, input("End: ").split(','))
        if self.mazelist[self.startRow][self.startCol] == OBSTACLE or self.mazelist[self.endRow][self.endCol] == OBSTACLE:
            print("invalid point")
            sys.exit()
        self.rowsInMaze = rowsInMaze
        self.columnsInMaze = columnsInMaze
        self.xTranslate = -columnsInMaze/2
        self.yTranslate = rowsInMaze/2

    def updatePosition(self, row, col, val=None):
        if val:
            self.mazelist[row][col] = val

        if val == PART_OF_PATH:
            node = []
            node.append(row)
            node.append(col)
            self.path.append(node)

    def returnpath(self):
        return self.path

    def __getitem__(self,idx):
        return self.mazelist[idx]


def searchFrom(maze, startRow, startColumn, endRow, endCol):
    # try each of four directions from this point until we find a way out.
    # base Case return values:
    #  1. We have run into an obstacle, return false
    maze.updatePosition(startRow, startColumn)
    if maze[startRow][startColumn] == OBSTACLE :
        return False
    #  2. We have found a square that has already been explored
    if maze[startRow][startColumn] == TRIED or maze[startRow][startColumn] == DEAD_END:
        return False
    # 3. We have found an outside edge not occupied by an obstacle
    if startRow == endRow and startColumn == endCol:
        print('Yes')
        maze.updatePosition(startRow, startColumn, PART_OF_PATH)
        return True
    maze.updatePosition(startRow, startColumn, TRIED)
    # Otherwise, use logical short circuiting to try each direction
    # in turn (if needed)
    found = searchFrom(maze, startRow-1, startColumn, endRow, endCol) or \
            searchFrom(maze, startRow+1, startColumn, endRow, endCol) or \
            searchFrom(maze, startRow, startColumn-1, endRow, endCol) or \
            searchFrom(maze, startRow, startColumn+1, endRow, endCol)
    if found:
        maze.updatePosition(startRow, startColumn, PART_OF_PATH)

    else:
        maze.updatePosition(startRow, startColumn, DEAD_END)
    return found

myMaze = Maze('maze.txt')

myMaze.updatePosition(myMaze.startRow, myMaze.startCol)

searchFrom(myMaze, myMaze.startRow, myMaze.startCol,myMaze.endRow,myMaze.endCol)

a = myMaze.returnpath()
a.reverse()
for i in a:
    print(i)