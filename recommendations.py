import math

def sim_distance(data, key1, key2):
    p1 = data[key1]
    p2 = data[key2]
    pp = set(p1) & set(p2)
    sum_of_squares = sum([(p1[p] - p2[p]) ** 2 for p in pp])
    return 1 / (sum_of_squares + 1)

def sim_pearson(data, key1, key2):
    p1 = data[key1]
    p2 = data[key2]
    pp = set(p1) & set(p2)
    n = len(pp)
    if n == 0:
        return 0

    sum1 = sum([p1[p] for p in pp])
    sum2 = sum([p2[p] for p in pp])
    sum1Sq = sum([p1[p] ** 2 for p in pp])
    sum2Sq = sum([p2[p] ** 2 for p in pp])
    pSum = sum([p1[p] * p2[p] for p in pp])

    num = pSum - (sum1 * sum2 / n)
    den = math.sqrt((sum1Sq - sum1 ** 2 / n) * (sum2Sq - sum2 ** 2 / n))

    return num / den

def top_matches(data, key, limit=5, similarity=sim_pearson):
    return sorted([(similarity(data, key, other), other) for other in data if key != other], reverse=True)[:limit]

def transpose(data):
    result = {}
    for k1 in data:
        for k2 in data[k1]:
            result.setdefault(k2, {})
            result[k2][k1] = data[k1][k2]
    return result
