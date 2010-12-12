from math import sqrt
from random import random
import Image
import ImageDraw

def readfile(filename):
    lines = [line for line in file(filename)]
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        rownames.append(p[0])
        data.append([float(x) for x in p[1:]])
    return rownames, colnames, data

def pearson(v1, v2):
    sum1 = sum(v1)
    sum2 = sum(v2)
    sum1Sq = sum([pow(v, 2) for v in v1])
    sum2Sq = sum([pow(v, 2) for v in v2])
    pSum = sum([v1[i] * v2[i] for i in range(len(v1))])
    num = pSum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2) / len(v1)))
    if den == 0: return 0
    return 1.0 - num / den

class bicluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id

def hcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1
    clust = [bicluster(rows[i], id=i) for i in range(len(rows))]
    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]
                if d < closest:
                    closest = d
                    lowestpair = (i, j)
        left, right = lowestpair
        mergevec = [(clust[left].vec[i] + clust[right].vec[i]) / 2.0 for i in range(len(clust[0].vec))]
        newcluster = bicluster(mergevec, left=clust[left], right=clust[right], distance=closest, id=currentclustid)
        currentclustid -= 1
        del clust[right]
        del clust[left]
        clust.append(newcluster)
    return clust[0]

def printclust(clust, labels=None, n=0):
    print ' ' * n,
    if clust.id < 0:
        print '-'
    else:
        print clust.id if labels == None else labels[clust.id]
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n + 1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n + 1)

def getheight(clust):
    if clust.left == None and clust.right == None:
        return 1
    return getheight(clust.left) + getheight(clust.right)

def getdepth(clust):
    if clust.left == None and clust.right == None:
        return 0
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance

def drawdendrogram(clust, labels, jpeg='clusters.jpg'):
    h = getheight(clust) * 20
    w = 1200
    depth = getdepth(clust)
    scaling = float(w - 150) / depth
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((0, h / 2, 10, h / 2), fill=(255, 0, 0))
    drawnode(draw, clust, 10, (h / 2), scaling, labels)
    img.save(jpeg, 'JPEG')

def drawnode(draw, clust, x, y, scaling, labels):
    if clust.id < 0:
        h1 = getheight(clust.left) * 20
        h2 = getheight(clust.right) * 20
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2
        ll = clust.distance * scaling
        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))
        draw.line((x, top + h1 / 2, x + ll, top + h1 / 2), fill=(255, 0, 0))
        draw.line((x, bottom - h2 / 2, x + ll, bottom - h2 / 2), fill=(255, 0, 0))
        drawnode(draw, clust.left, x + ll, top + h1 / 2, scaling, labels)
        drawnode(draw, clust.right, x + ll, bottom - h2 / 2, scaling, labels)
    else:
        draw.text((x + 5, y - 7), labels[clust.id], (0, 0, 0))

def kcluster(rows, distance=pearson, k=4):
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows])) for i in range(len(rows[0]))]
    clusters = [[random()*(ranges[i][1] - ranges[i][0]) + ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]

    lastmatches = None
    for t in range(100):
        print 'Iteration %d' % t
        bestmatches = [[] for i in range(k)]
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                if distance(clusters[i], row) < distance(clusters[bestmatch], row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)

        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        for i in range(k):
            sums = [0.0] * len(rows[0])
            bm = bestmatches[i]
            for rowid in bm:
                for m in range(len(rows[rowid])):
                    sums[m] += rows[rowid][m]
            clusters[i] = [s / len(bm) for s in sums]
    return bestmatches

def tanamoto(v1, v2):
    s1 = set([i for i in range(len(v1)) if v1[i]])
    s2 = set([i for i in range(len(v2)) if v2[i]])
    return 1.0 - float(len(s1 & s2)) / len(s1 | s2)

def scaledown(data, dimension=3, distance=pearson, rate=0.01):
    n = len(data)
    realdist = [[distance(data[i], data[j]) for j in range(n)] for i in range(0, n)]
    loc = [[random() for _ in range(dimension)] for i in range(n)]
    fakedist = [[0.0 for j in range(n)] for i in range(n)]

    lasterror = None
    for _ in range(0, 1000):
        for i in range(n):
            for j in range(n):
                fakedist[i][j] = sqrt(sum([(loc[i][d] - loc[j][d]) ** 2 for d in range(dimension)]))

        grad = [[0.0] * dimension for i in range(n)]
        totalerror = 0
        for k in range(n):
            for j in range(n):
                if j == k:
                    continue
                errorterm = (fakedist[j][k] - realdist[j][k]) / realdist[j][k]
                for d in range(dimension):
                    grad[k][d] += ((loc[k][d] - loc[j][d]) / fakedist[j][k]) * errorterm
                totalerror += abs(errorterm)
        print totalerror

        if lasterror and lasterror < totalerror:
            break
        lasterror = totalerror

        for k in range(n):
            for d in range(dimension):
                loc[k][d] -= rate * grad[k][d]

    return loc

def draw2d(data, labels, jpeg='mds2d.jpg'):
    img = Image.new('RGB', (2000, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(len(data)):
        x = (data[i][0] + 0.5) * 1000
        y = (data[i][1] + 0.5) * 1000
        draw.text((x, y), labels[i], (0, 0, 0))
    img.save(jpeg, 'JPEG')

def draw3d(data, labels, label_direction=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(len(data)):
        x, y, z = (data[i] + [0, 0])[:3]
        ax.text(x, y, z, labels[i], label_direction)
    plt.show()
