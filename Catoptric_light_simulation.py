import math
import cv2
import numpy as np
import random
from PIL import  Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torchvision.models as models
import matplotlib.pylab as pyl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.resnet50(pretrained=True).eval().to(device)

#Classifier
def classify(dir, net):
    img = Image.open(dir)
    img = img.convert("RGB")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,std = std)
    ])(img).to(device)

    f_image = net.forward(Variable(img[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]  # 从小到大排序, 再从后往前复制一遍，So相当于从大到小排序
    I = I[0:10]  # 挑最大的num_classes个(从0开始，到num_classes结束)
    # print(I)
    label = I[0]  # 最大的判断的分类
    confidence = f_image[35]

    return label, confidence

#Plot the minimum confidence for each iteration
def drow(x, y):
    # x = [1, 3, 5, 6, 8, 13, 14, 16]
    # y = [5, 1, 6, 7, 9, 3, 2, 10]
    pyl.plot(x, y)
    pyl.show()

#Catoptric light simulation
def img_color_patch_effetc_digital(img, x1, y1, x2, y2, x3, y3, b, g, r, width, height, path):

    points = np.array([[(x1, y1), (x2, y2), (x3, y3)]], np.int32)
    cv2.fillPoly(img, points, (r, g, b))

    cv2.imwrite(path, img)

#catoptric light intensity adjustment
def video_color_patch_effect(img, cnt, I, path_adv):

    if cnt == 0:
        return img

    height, width, n = img.shape

    mask = {
        1: cv2.imread(path_adv),
    }

    mask[cnt] = cv2.resize(mask[cnt], (width, height), interpolation=cv2.INTER_CUBIC)

    new_img = cv2.addWeighted(img, (1 - I), mask[cnt], I, 0)

    return new_img

#Input the phenotype into the model
def img_color_patch_effetc(dir_read, b):

    img = cv2.imread(dir_read)

    height, width, n = img.shape
    # print('height, width = ', height, width)


    x1, y1, x2, y2, x3, y3, r, g, b1, I = b[0][0], b[0][1], b[0][2], b[0][3], b[0][4], b[0][5], b[0][6], b[0][7], b[0][8], b[0][9]

    if x1 > width:
        x1 = x1//2
    if y1 > height:
        y1 = y1//2
    if x2 > width:
        x2 = x2//2
    if y2 > height:
        y2 = y2//2
    if x3 > width:
        x3 = x3//2
    if y3 > height:
        y3 = y3//2

    path_adv = 'adv.jpg'
    img_color_patch_effetc_digital(img, x1, y1, x2, y2, x3, y3, r, g, b1, width, height, path_adv)

    cap = cv2.VideoCapture(dir_read)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    i = 1
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(path_adv, fourcc, fps, (width, height))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = video_color_patch_effect(frame, i % 5, I, path_adv)
            # cv2.imshow('video', frame)
            videoWriter.write(frame)
            i += 1
            c = cv2.waitKey(1)
            if c == 27:
                break
        else:
            break

#Random generation of individual genotypes
def initiation(a):
    population, choromosome_length = a.shape
    # print('population, choromosome_length = ', population, choromosome_length)
    for i in range(0, population):
        for j in range(0, choromosome_length):
            a[i][j] = random.randint(0, 1)
            # print('a[i][j] = ', a[i][j])
    return a

#Genotype to phenotype
def translation(a, seed, N):
    b = np.zeros((seed, 10))
    population, choromosome_length = a.shape

    # print('population, choromosome_length = ', population, choromosome_length)

    for i in range(0, population):
        for j in range(0, 54):
            # print('j//9, (j//9)*9-1-j) = ', j//9, (j//9+1)*9-1-j)
            b[i][j//9] += a[i][j] * (math.pow(2, (j//9+1)*9-1-j))

    for i in range(0, population):
        for j in range(54, 78):
            # print('j//9, (j//9)*9-1-j) = ', j//9, (j//9+1)*9-1-j)
            b[i][(j-54)//8+6] += a[i][j] * (math.pow(2, ((j-54)//8+1)*8-1-(j-54)))

    for i in range(0, population):
        for j in range(78, 80):
            # print('j//9, (j//9)*9-1-j) = ', j//9, (j//9+1)*9-1-j)
            b[i][(j-78)//2+6+3] += (a[i][j] * (math.pow(2, ((j-78)//2+1)*2-1-(j-78)))) / 10 + 0.1


    return b

#Gets the label and confidence of the adversarial sample
def function(dir_read, net, b, tag_break):


    img_color_patch_effetc(dir_read, b)

    save_path = 'adv.jpg'

    img_show = Image.open(save_path)
    # plt.imshow(img_show)
    # plt.show()

    label_adv, conf = classify(save_path, net)

    if int(label_adv) != 35:
        # img_save = plt.imread(save_path)
        # name_save = 'result.jpg'
        # plt.imsave(name_save, img_save)
        tag_break = 1

    return label_adv, conf, tag_break

#The genotypes were sorted in order of confidence
def sort(d, seed, N, conf, tag):

    for i in range(seed):
        for j in range(0, seed-1):
            if conf[0][j] > conf[0][j+1]:

                conf_tag = conf[0][j]
                conf[0][j] = conf[0][j+1]
                conf[0][j+1] = conf_tag

                for k in range(N):
                    tag[0][k] = d[j][k]
                    d[j][k] = d[j+1][k]
                    d[j+1][k] = tag[0][k]

    return d, conf

#Elimination
def elimination(d, seed, N, elimination_num):

    temp = np.random.randint(0, 2, (elimination_num, N))

    for i in range(elimination_num):
        for j in range(N):
            # d[seed-elimination_num+i][j] = d[i][j]
            # 将后面位的换为随机数
            d[seed - elimination_num + i][j] = temp[i][j]

    # print("test: d = ", d)
    return d

#Crossover
def crossover(d, seed, N, pc):

    #取消随机打乱然后在相邻交叉，直接按序交叉即可
    d_tag = d

    #开始进行交叉，对于每两行，以pc的概率进行交叉，交叉时，随机选择交叉位置，随机选择交叉左边还是右边
    for i in range(int(seed/2)):
        p = random.randint(1, 10)
        # print("是否交叉的概率p：", p)
        if p <= (10 * pc):
            position = random.randint(1, N)
            # print("交叉的位置选择position：", position)
            p_left = random.randint(0, 1)
            # print("交叉左边还是右边（0是右1是左）：", p_left)
            if p_left == 1:
                for j in range(0, position):
                    tag = d_tag[2*i][j]
                    d_tag[2*i][j] = d_tag[2*i+1][j]
                    d_tag[2*i+1][j] = tag
            if p_left == 0:
                for j in range(position, N+1):
                    tag = d_tag[2*i][j]
                    d_tag[2*i][j] = d_tag[2*i+1][j]
                    d_tag[2*i+1][j] = tag


    return d_tag

#Metamorphosis
def metamorphosis(d, seed, N, pm):

    for i in range(seed):
        p = random.randint(1, 10)
        # print("变异的概率p：", p)
        if p <= 10*pm:
            position = random.randint(1, N)
            # print("变异的位置：", position)
            # print("d[i][position] = ", d[i][position])
            if int(d[i][position]) == 0:
                d[i][position] = 1
            else:
                d[i][position] = 0

    return d

