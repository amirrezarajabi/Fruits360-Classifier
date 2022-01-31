# -*- coding: utf-8 -*-

# python imports
from math import degrees

# pyfuzzy imports
from fuzzy.storage.fcl.Reader import Reader


PA  = {
    "up_more_right" : ((0, 0), (30, 1), (60, 0)),
    "up_right" : ((30, 0), (60, 1), (90, 0)),
    "up" : ((60, 0), (90, 1), (120, 0)),
    "up_left" : ((90, 0), (120, 1), (150, 0)),
    "up_more_left" : ((120, 0), (150, 1), (180, 0)),
    "down_more_left" : ((180, 0), (210, 1), (240, 0)),
    "down_left" : ((210, 0), (240, 1), (270, 0)),
    "down" : ((240, 0), (270, 1), (300, 0)),
    "down_right" : ((270, 0), (300, 1), (330, 0)),
    "down_more_right" : ((300, 0), (330, 1), (360, 0))
}

PV = {
    "cw_fast": ((-200, 1), (-100, 0)),
    "cw_slow": ((-200, 0), (-100, 1), (0, 0)),
    "stop": ((-100, 0), (0, 1), (100, 0)),
    "ccw_slow": ((0, 0), (100, 1), (200, 0)),
    "ccw_fast": ((100, 0), (200, 1))
}	

CP = {
    "left_far": ((-10, 1), (-5, 0)),
    "left_near": ((-10, 0), (-2.5, 1), (0, 0)),
    "stop": ((-2.5, 0), (0, 1), (2.5, 0)),
    "right_near": ((0, 0), (2.5, 1), (10, 0)),
    "right_far": ((5, 0), (10, 1))
}

CV = {
    "left_fast": ((-5, 1), (-2.5, 0)),
    "left_slow": ((-5, 0), (-1, 1), (0, 0)),
    "stop": ((-1, 0), (0, 1), (1, 0)),
    "right_slow": ((0, 0), (1, 1), (5, 0)),
    "right_fast": ((2.5, 0), (5, 1))
}

FORCE = {
    "left_fast": ((-100, 0), (-80, 1), (-60, 0)),
    "left_slow": ((-80, 0), (-60, 1), (0, 0)),
    "stop": ((-60, 0), (0, 1), (60, 0)),
    "right_slow": ((0, 0), (60, 1), (80, 0)),
    "right_fast": ((60, 0), (80, 1), (100, 0))
}
RULE = [[[("PA" ,"up" ), ("PV", "stop")], [("PA", "up_right"), ("PV" , "ccw_slow")], [("PA", "up_left"), ("PV", "cw_slow")], ["FORCE", "stop"]]]

RULE1 = [[[("PA" ,"up" ), ("PV", "stop")], [("PA", "up_right"), ("PV" , "ccw_slow")], [("PA", "up_left"), ("PV", "cw_slow")], ["FORCE", "stop"]],
        [[("CP", "right_near"),("CV",	"left_fast")],[("PA","up_right"),("PV","ccw_slow")], [("PA","up_left"),("PV","cw_slow")],["FORCE", "stop"]],
        [[("CP", "left_near"),("CV",	"right_fast")],[("PA","up_right"),("PV","ccw_slow")], [("PA","up_left"),("PV","cw_slow")],["FORCE", "stop"]]]

RULES = [
    [("PA", "up_more_right"), ("PV", "ccw_slow"), ("FORCE", "right_fast")],
    [("PA", "up_more_right"), ("PV", "cw_slow"), ("FORCE", "right_fast")],


    [("PA", "up_more_left"), ("PV", "cw_slow"), ("FORCE", "left_fast")],
    [("PA", "up_more_left"), ("PV", "ccw_slow"), ("FORCE", "left_fast")],


    [("PA", "up_more_right"), ("PV", "ccw_fast"), ("FORCE", "left_slow")],
    [("PA", "up_more_right"), ("PV", "cw_fast"), ("FORCE", "right_fast")],


    [("PA", "up_more_left"), ("PV", "cw_fast"), ("FORCE", "right_slow")],
    [("PA", "up_more_left"), ("PV", "ccw_fast"), ("FORCE", "left_fast")],


    [("PA", "down_more_right"), ("PV", "ccw_slow"), ("FORCE", "right_fast")],
    [("PA", "down_more_right"), ("PV", "cw_slow"), ("FORCE", "stop")],

    [("PA", "down_more_left"), ("PV", "cw_slow"), ("FORCE", "left_fast")],
    [("PA", "down_more_left"), ("PV", "ccw_slow"), ("FORCE", "stop")],


    [("PA", "down_more_right"), ("PV", "ccw_fast"), ("FORCE", "stop")],
    [("PA", "down_more_right"), ("PV", "cw_fast"), ("FORCE", "stop")],

    [("PA", "down_more_left"), ("PV", "cw_fast"), ("FORCE", "stop")],
    [("PA", "down_more_left"), ("PV", "ccw_fast"), ("FORCE", "stop")],


    [("PA", "down_right"), ("PV", "ccw_slow"), ("FORCE", "right_fast")],
    [("PA", "down_right"), ("PV", "cw_slow"), ("FORCE", "right_fast")],

    [("PA", "down_left"), ("PV", "cw_slow"), ("FORCE", "left_fast")],
    [("PA", "down_left"), ("PV", "ccw_slow"), ("FORCE", "left_fast")],


    [("PA", "down_right"), ("PV", "ccw_fast"), ("FORCE", "stop")],
    [("PA", "down_right"), ("PV", "cw_fast"), ("FORCE", "right_slow")],

    [("PA", "down_left"), ("PV", "cw_fast"), ("FORCE", "stop")],
    [("PA", "down_left"), ("PV", "ccw_fast"), ("FORCE", "left_slow")],


    [("PA", "up_right"), ("PV", "ccw_slow"), ("FORCE", "right_slow")],
    [("PA", "up_right"), ("PV", "cw_slow"), ("FORCE", "right_fast")],
    [("PA", "up_right"), ("PV", "stop"), ("FORCE", "right_fast")],
    [("PA", "up_left"), ("PV", "cw_slow"), ("FORCE", "left_slow")],
    [("PA", "up_left"), ("PV", "ccw_slow"), ("FORCE", "left_fast")],
    [("PA", "up_left"), ("PV", "stop"), ("FORCE", "left_fast")],


    [("PA", "up_right"), ("PV", "ccw_fast"), ("FORCE", "left_fast")],
    [("PA", "up_right"), ("PV", "cw_fast"), ("FORCE", "right_fast")],
    [("PA", "up_left"), ("PV", "cw_fast"), ("FORCE", "right_fast")],
    [("PA", "up_left"), ("PV", "ccw_fast"), ("FORCE", "left_fast")],


    [("PA", "down"), ("PV", "stop"), ("FORCE", "right_fast")],
    [("PA", "down"), ("PV", "cw_fast"), ("FORCE", "stop")],
    [("PA", "down"), ("PV", "ccw_fast"), ("FORCE", "stop")],

    [("PA", "up"), ("PV", "ccw_slow"), ("FORCE", "left_slow")],
    [("PA", "up"), ("PV", "ccw_fast"), ("FORCE", "left_fast")],
    [("PA", "up"), ("PV", "cw_slow"), ("FORCE", "right_slow")],
    [("PA", "up"), ("PV", "cw_fast"), ("FORCE", "right_fast")],
    [("PA", "up"), ("PV", "stop"), ("FORCE", "stop")]]


RULES1 = [
    [("PA", "up_more_right"), ("PV", "ccw_slow"), ("FORCE", "right_fast")],
    [("PA", "up_more_right"), ("PV", "cw_slow"), ("FORCE", "right_fast")],


    [("PA", "up_more_left"), ("PV", "cw_slow"), ("FORCE", "left_fast")],
    [("PA", "up_more_left"), ("PV", "ccw_slow"), ("FORCE", "left_fast")],


    [("PA", "up_more_right"), ("PV", "ccw_fast"), ("FORCE", "left_slow")],
    [("PA", "up_more_right"), ("PV", "cw_fast"), ("FORCE", "right_fast")],


    [("PA", "up_more_left"), ("PV", "cw_fast"), ("FORCE", "right_slow")],
    [("PA", "up_more_left"), ("PV", "ccw_fast"), ("FORCE", "left_fast")],


    [("PA", "down_more_right"), ("PV", "ccw_slow"), ("FORCE", "right_fast")],
    [("PA", "down_more_right"), ("PV", "cw_slow"), ("FORCE", "stop")],

    [("PA", "down_more_left"), ("PV", "cw_slow"), ("FORCE", "left_fast")],
    [("PA", "down_more_left"), ("PV", "ccw_slow"), ("FORCE", "stop")],


    [("PA", "down_more_right"), ("PV", "ccw_fast"), ("FORCE", "stop")],
    [("PA", "down_more_right"), ("PV", "cw_fast"), ("FORCE", "stop")],

    [("PA", "down_more_left"), ("PV", "cw_fast"), ("FORCE", "stop")],
    [("PA", "down_more_left"), ("PV", "ccw_fast"), ("FORCE", "stop")],


    [("PA", "down_right"), ("PV", "ccw_slow"), ("FORCE", "right_fast")],
    [("PA", "down_right"), ("PV", "cw_slow"), ("FORCE", "right_fast")],

    [("PA", "down_left"), ("PV", "cw_slow"), ("FORCE", "left_fast")],
    [("PA", "down_left"), ("PV", "ccw_slow"), ("FORCE", "left_fast")],


    [("PA", "down_right"), ("PV", "ccw_fast"), ("FORCE", "stop")],
    [("PA", "down_right"), ("PV", "cw_fast"), ("FORCE", "right_slow")],

    [("PA", "down_left"), ("PV", "cw_fast"), ("FORCE", "stop")],
    [("PA", "down_left"), ("PV", "ccw_fast"), ("FORCE", "left_slow")],


    [("PA", "up_right"), ("PV", "ccw_slow"), ("FORCE", "right_slow")],
    [("PA", "up_right"), ("PV", "cw_slow"), ("FORCE", "right_fast")],
    [("PA", "up_right"), ("PV", "stop"), ("FORCE", "right_fast")],
    [("PA", "up_left"), ("PV", "cw_slow"), ("FORCE", "left_slow")],
    [("PA", "up_left"), ("PV", "ccw_slow"), ("FORCE", "left_fast")],
    [("PA", "up_left"), ("PV", "stop"), ("FORCE", "left_fast")],


    [("PA", "up_right"), ("PV", "ccw_fast"), ("FORCE", "left_fast")],
    [("PA", "up_right"), ("PV", "cw_fast"), ("FORCE", "right_fast")],
    [("PA", "up_left"), ("PV", "cw_fast"), ("FORCE", "right_fast")],
    [("PA", "up_left"), ("PV", "ccw_fast"), ("FORCE", "left_fast")],


    [("PA", "down"), ("PV", "stop"), ("FORCE", "right_fast")],
    [("PA", "down"), ("PV", "cw_fast"), ("FORCE", "stop")],
    [("PA", "down"), ("PV", "ccw_fast"), ("FORCE", "stop")],

    [("PA", "up"), ("PV", "ccw_slow"), ("FORCE", "left_slow")],
    [("PA", "up"), ("PV", "ccw_fast"), ("FORCE", "left_fast")],
    [("PA", "up"), ("PV", "cw_slow"), ("FORCE", "right_slow")],
    [("PA", "up"), ("PV", "cw_fast"), ("FORCE", "right_fast")],
    [("PA", "up"), ("PV", "stop"), ("FORCE", "stop")],
    [("CP",	"left_far"), 	("CV",	"left_fast"),	("FORCE",	"right_fast")],
    [("CP",	"right_far"), 	("CV",	"right_fast"),	("FORCE",	"left_fast")]]


def find_y(X1, X2, x):
    return (float(X1[1] - X2[1]) / float(X1[0] - X2[0])) * float(x - X1[0]) + float(X1[1])


def belong(structure, x):
    if (len(structure) == 2):
        X1 = structure[0]
        X2 = structure[1]
        if (x >= X1[0] and x <= X2[0]):
            return find_y(X1, X2, x)
        elif(X1[1] == 1 and x <= X1[0]):
            return 1
        elif(X2[1] == 1 and x >= X2[0]):
            return 1
        return 0
    if (len(structure) == 3):
        X1 = structure[0]
        X2 = structure[1]
        X3 = structure[2]
        if (x >= X1[0] and x <= X2[0]):
            return find_y(X1, X2, x)
        elif (x >= X2[0] and x <= X3[0]):
            return find_y(X2, X3, x)
        return 0

DICT = {
    "PA": PA,
    "PV": PV,
    "CP": CP,
    "CV": CV,
    "FORCE": FORCE
}

def GET(V, k, x):
    return belong(DICT[V][k], x)

def RESULT(rules, rule, value):
    DICT_RESULT = {
        "left_fast": 0,
        "left_slow": 0,
        "stop": 0,
        "right_slow": 0,
        "right_fast": 0
    }
    for r in rule:
        temps = []
        for i in range(len(r) - 1):
            temps.append(min(GET(r[i][0][0], r[i][0][1], value[r[i][0][0]]), GET(r[i][1][0], r[i][1][1], value[r[i][1][0]])))
        temps = max(temps)
        DICT_RESULT[r[-1][1]] = temps
    for r in rules:
        temp = min(GET(r[0][0], r[0][1], value[r[0][0]]), GET(r[1][0], r[1][1], value[r[1][0]])) 
        DICT_RESULT[r[2][1]] = max(DICT_RESULT[r[2][1]], temp)
    return DICT_RESULT

def max_belongs(structures, x, value):
    tmp = 0
    for i in structures:
        minimum = min(GET("FORCE", i, x), value[i])
        tmp = max(tmp, minimum)
    return tmp

def center(value, FORCE, N):
    step = 200.0 / N
    p = -100.0
    sums = 0.0
    sumsb = 0.0
    while(p <= 100.1):
        p = p + step
        sums = float(float(sums) + float(step) * float(p) * float(max_belongs(FORCE, p, value)))
        sumsb = float(float(sumsb) + float(step) *  float(max_belongs(FORCE, p, value)))
    if sumsb != 0:
        return float(sums / sumsb)
    return 0.0

def together (rules, rule, value, N, FORCE):
    res = RESULT(rules, rule, value)
    return center(res, FORCE, N)

class FuzzyController:

    def __init__(self, fcl_path):
        self.system = Reader().load_from_file(fcl_path)


    def _make_input(self, world):
        res = {
            "CP" : world.x,
            "CV" : world.v,
            "PA" : degrees(world.theta),
            "PV" : degrees(world.omega)
        }
        return res


    def _make_output(self):
        return dict(
            force = 0.
        )


    def decide(self, world):
        output = self._make_output()
        L = 1
        if (L == 1):
            output['force'] = together(RULES1, RULE1, self._make_input(world), 1000, FORCE)
        else:
            output['force'] = together(RULES, RULE, self._make_input(world), 1000, FORCE)
        return output['force']
