import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from Catoptric_light_simulation import drow, initiation, translation, function, sort, elimination, crossover, metamorphosis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.resnet50(pretrained=True).eval().to(device)

tag_save = 1
tag_35 = np.zeros((1, 3))
if __name__ == "__main__":

    seed = 100 #population
    step = 20 #iterations
    N = 80#genes
    pc, pm = 0.7, 0.1

    a = np.zeros((seed, N))
    b = np.zeros((1, 10))
    c = np.zeros((seed, 10))
    d = np.zeros((seed, N+1))
    conf_35 = np.zeros((1, seed))
    label_adv = np.zeros((1, seed))
    a = initiation(a)#Initialize the genotype
    print('a = ', a)
    c = translation(a, seed, N)#Genotypes turn into phenotypes
    print('c = ', c)

    for i in range(seed):
        for j in range(N):
            d[i][j] = a[i][j]
    print('d = ', d)

    tag = np.zeros((1, N + 1))

    dir_read = "35.jpg"#Image path

    y_drow = np.zeros(step)
    x_drow = np.zeros(step)
    for i in range(1, step + 1):
        x_drow[i - 1] = i

    tag_break = 0
    for step in range(0, step):

        for i in range(0, seed):
            tag_break = 0

            for j in range(0, 10):
                b[0][j] = c[i][j]
                # print('b[0][j] = ', b[0][j])

            print("step, seed, tag_35 = ", step, i, tag_35)
            label_adv[0][i], conf_35[0][i], tag_break = function(dir_read, net, b, tag_break)
            print('label_adv[0][i], conf_35[0][i]', label_adv[0][i], conf_35[0][i])

            if tag_break == 1:

                img_save = plt.imread('adv.jpg')

                tag_35[0][0] = tag_35[0][0] + 1
                tag_35[0][1], tag_35[0][2] = step, i
                print('tag_35 = ', tag_35)
                name_save = 'result/' + str(tag_save) + '.jpg'

                plt.imsave(name_save, img_save)
                tag_save = tag_save + 1

            if tag_35[0][0] == 20:
                break
        if tag_35[0][0] == 20:
            break

        d, conf_35 = sort(d, seed, N + 1, conf_35, tag)#The genotypes were sorted in order of confidence
        # print("d = ", d)
        # print("conf_35 = ", conf_35)

        y_drow[step] = conf_35[0][0]#Plot the minimum confidence for each iteration
        # drow(x_drow, y_drow)

        elimination_num = int(seed / 10)#Select the number of eliminations

        d = elimination(d, seed, N + 1, elimination_num)#Perform eliminate

        for i in range(elimination_num):#Set the confidence of the new individual to 0
            conf_35[0][seed - elimination_num + i] = 0

        print("conf_35 = ", conf_35)

        d = crossover(d, seed, N, pc)#Crossover
        # print("crossover d = ", d)

        d = metamorphosis(d, seed, N, pm)#Metamorphosis
        # print("metamorphosis d：", d)

        c = translation(d, seed, N)#Update phenotype
        # print("c = ", c)
    # print("in the end, conf_35 = ", conf_35)
    # print("y_drow = ", y_drow)
