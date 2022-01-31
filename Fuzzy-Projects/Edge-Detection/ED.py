# import libraries

from PIL import Image           # for opening images
import numpy as np              # for arrays

#fuzzy membership dixtionary

DIF = {
    "HIGH": ((0, 0), (255, 1)),
    "MEDIUM": ((-255, 0), (0, 1), (255, 1)),
    "LOW": ((-255, 1), (0, 0))
}

PIXEL = {
    "W": ((56, 0), (200, 1)),
    "B": ((56, 1), (200, 0))
}


FUZZY_RULES = [
    [(1, "HIGH"), (5, "HIGH"), "B"],
    [(1, "LOW"), (5, "LOW"), "B"],
    [(1, "MEDIUM"), (5, "MEDIUM"), "W"],
    [(5, "HIGH"), (7, "HIGH"), "B"],
    [(5, "LOW"), (7, "LOW"), "B"],
    [(5, "MEDIUM"), (7, "MEDIUM"), "W"],
    [(1, "HIGH"), (3, "HIGH"), "B"],
    [(1, "LOW"), (3, "LOW"), "B"],
    [(1, "MEDIUM"), (3, "MEDIUM"), "W"],
    [(3, "HIGH"), (7, "HIGH"), "B"],
    [(3, "LOW"), (7, "LOW"), "B"],
    [(3, "MEDIUM"), (7, "MEDIUM"), "W"],
]

# get name
PATH = input("Enter image name: ")

# load image
pic = Image.open(PATH)

#convert image to array
img = np.array(pic)

# preprocessing
GRAY = True
if (len(img.shape) == 3):
    GRAY = False
if (GRAY):
    gray = img
else:
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])


# blong fuzzy
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

# split 3 * 3
def split_pic(picture, i, j):
    splitted = picture[i: i+3, j: j+3]
    
    # normalization
    splitted = splitted - np.mean(splitted)

    return splitted.astype(np.uint8)


# new image
new_img = np.zeros((img.shape[0] - 3, img.shape[1] - 3), dtype=np.uint8)


# fuzzy implementation

def fuzzy(RULES, value):
    RESULT = {
        "W": 0,
        "B": 0
    }
    for rule in RULES:
        RESULT[rule[2]] = max(min(belong(DIF[rule[0][1]], value[rule[0][0]]), belong(DIF[rule[1][1]], value[rule[1][0]])), RESULT[rule[2]])
    return RESULT

# maximum brlong in fuzzy set
def max_brlong(SET, RESULT, x):
    tmp = 0
    for s in SET:
        tmp = max(tmp, min(belong(SET[s], x), RESULT[s]))
    return tmp

# center of mass
def center_of_mass(SET, RESULT, N):
    STEP = 255 / N
    sum_belong = 0
    sum_belong_multiply_x = 0
    x = 0
    while(x < 255 + STEP):
        sum_belong += max_brlong(SET, RESULT, x)
        sum_belong_multiply_x += x * max_brlong(SET, RESULT, x)
        x += STEP
    return sum_belong_multiply_x / sum_belong

for i in range(new_img.shape[0]):
    for j in range(new_img.shape[1]):
        splitted = split_pic(gray, i, j).flatten()
        value = fuzzy(FUZZY_RULES, splitted)
        new_img[i][j] = center_of_mass(PIXEL, value, 512)

im = Image.fromarray(new_img)
im.save("EDGE-"+PATH)