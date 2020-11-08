import os
import numpy as np
import torch
import shutil
from Ada_LSN.genotypes import *
import random
import copy

INF = 1e6


def greater(arr1, arr2):
    flag = True
    for i in range(len(arr1)):
        flag = flag and (arr1[i] >= arr2[i])
    return flag


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


aspp = Genotype(
    geno1=[1, 1, 1, 0, 0],
    geno2=Decoder(
        unit1=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
        unit2=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
        unit3=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
        unit4=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
        unit5=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
    ),
    geno3=[[1], [0, 1], [0, 1], [0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0],
           [('conv3', 0), ('dconv3_2', 1), ('dconv3_2', 2), ('dconv3_2', 3)]]
)


def init_one(aspp_flag=True):
    while True:
        geno1 = [np.random.randint(0, 2), np.random.randint(0, 2), np.random.randint(0, 2), np.random.randint(0, 2),
                 np.random.randint(0, 2)]
        if sum(geno1) >= 3:
            break
    if aspp_flag:
        geno2 = aspp[1]
    else:
        indx = [0]
        u1 = []
        for i in range(4):
            o = random.sample(PRIMITIVES, 1)
            d = random.sample(indx, 1)
            indx.append(i + 1)
            u1.append((o[0], d[0]))
        u2 = []
        indx = [0]
        for i in range(4):
            o = random.sample(PRIMITIVES, 1)
            d = random.sample(indx, 1)
            indx.append(i + 1)
            u2.append((o[0], d[0]))
        u3 = []
        indx = [0]
        for i in range(4):
            o = random.sample(PRIMITIVES, 1)
            d = random.sample(indx, 1)
            indx.append(i + 1)
            u3.append((o[0], d[0]))
        u4 = []
        indx = [0]
        for i in range(4):
            o = random.sample(PRIMITIVES, 1)
            d = random.sample(indx, 1)
            indx.append(i + 1)
            u4.append((o[0], d[0]))
        # to delete if there are 4 units
        u5 = []
        indx = [0]
        for i in range(4):
            o = random.sample(PRIMITIVES, 1)
            d = random.sample(indx, 1)
            indx.append(i + 1)
            u5.append((o[0], d[0]))
        geno2 = Decoder(
            unit1=u1,
            unit2=u2,
            unit3=u3,
            unit4=u4,
            unit5=u5
        )
    i2 = [random.sample([geno1[0], 0], 1)[0]] if geno1[1] == 1 else [0]
    i3 = [random.sample([geno1[0], 0], 1)[0], random.sample([geno1[1], 0], 1)[0]] if geno1[2] == 1 else [0, 0]
    i4 = [random.sample([geno1[1], 0], 1)[0], random.sample([geno1[2], 0], 1)[0]] if geno1[3] == 1 else [0, 0]
    i5 = [random.sample([geno1[2], 0], 1)[0], random.sample([geno1[3], 0], 1)[0]] if geno1[4] == 1 else [0, 0]
    while True:
        ifuse = [random.sample([geno1[0], 0], 1)[0], random.sample([geno1[1], 0], 1)[0],
                 random.sample([geno1[2], 0], 1)[0], random.sample([geno1[3], 0], 1)[0],
                 random.sample([geno1[4], 0], 1)[0]]
        if sum(ifuse) != 0:
            break
    geno3 = [i2, i3, i4, i5, ifuse,
             [random.sample([geno1[0], 0], 1)[0], random.sample([geno1[1], 0], 1)[0],
              random.sample([geno1[2], 0], 1)[0], random.sample([geno1[3], 0], 1)[0],
              random.sample([geno1[4], 0], 1)[0]]]
    if aspp_flag:
        geno3.append(aspp[1][0])
    else:
        fu = []
        indx = [0]
        for i in range(4):
            o = random.sample(PRIMITIVES, 1)
            d = random.sample(indx, 1)
            indx.append(i + 1)
            fu.append((o[0], d[0]))
        geno3.append(fu)
    g = Genotype(
        geno1=geno1,
        geno2=geno2,
        geno3=geno3,
    )
    return g


def show_geno(g):
    print(g[0])
    print(g[1][0])
    print(g[1][1])
    print(g[1][2])
    print(g[1][3])
    print(g[1][4])
    print(g[2][0], g[2][1], g[2][2], g[2][3], g[2][4], g[2][5])
    print(g[2][6])


def init_population(num):
    population = []
    # init populations with experts practices
    for _ in range(num // 2):
        # init from aspp
        d = list([copy.deepcopy(init_one(aspp_flag=True)), random.random()])
        population.append(d)
    for _ in range(num // 2):
        # init from random
        d = list([copy.deepcopy(init_one(aspp_flag=False)), random.random()])
        population.append(d)
    return population


def cross(population, pc=0.2):
    population_ = copy.deepcopy(population)
    popu = []
    population_ = random.sample(population_, len(population_))
    for i in range(len(population_) // 2):
        # geno0 begin
        if random.random() < pc:
            # cross geno0
            g1_0 = population_[i * 2 + 1][0][0]
            g2_0 = population_[i * 2][0][0]
            ###  fuse cell  ###
            g1_f = population_[i * 2 + 1][0][2][4]
            g2_f = population_[i * 2][0][2][4]
        else:
            g1_0 = population_[i * 2][0][0]
            g2_0 = population_[i * 2 + 1][0][0]
            ###  fuse cell  ###
            g1_f = population_[i * 2][0][2][4]
            g2_f = population_[i * 2 + 1][0][2][4]
        # geno0 end

        # geno1 begin
        if random.random() < pc:
            g1_u1 = population_[i * 2 + 1][0][1][0]
            g2_u1 = population_[i * 2][0][1][0]
        else:
            g1_u1 = population_[i * 2][0][1][0]
            g2_u1 = population_[i * 2 + 1][0][1][0]
        if random.random() < pc:
            g1_u2 = population_[i * 2 + 1][0][1][1]
            g2_u2 = population_[i * 2][0][1][1]
        else:
            g1_u2 = population_[i * 2][0][1][1]
            g2_u2 = population_[i * 2 + 1][0][1][1]
        if random.random() < pc:
            g1_u3 = population_[i * 2 + 1][0][1][2]
            g2_u3 = population_[i * 2][0][1][2]
        else:
            g1_u3 = population_[i * 2][0][1][2]
            g2_u3 = population_[i * 2 + 1][0][1][2]
        if random.random() < pc:
            g1_u4 = population_[i * 2 + 1][0][1][3]
            g2_u4 = population_[i * 2][0][1][3]
        else:
            g1_u4 = population_[i * 2][0][1][3]
            g2_u4 = population_[i * 2 + 1][0][1][3]
        if random.random() < pc:
            g1_u5 = population_[i * 2 + 1][0][1][4]
            g2_u5 = population_[i * 2][0][1][4]
        else:
            g1_u5 = population_[i * 2][0][1][4]
            g2_u5 = population_[i * 2 + 1][0][1][4]
        # geno1 end

        # geno2 begin
        if random.random() < pc:
            g1_i2 = population_[i * 2 + 1][0][2][0]
            g2_i2 = population_[i * 2][0][2][0]
        else:
            g1_i2 = population_[i * 2][0][2][0]
            g2_i2 = population_[i * 2 + 1][0][2][0]

        if random.random() < pc:
            g1_i3 = population_[i * 2 + 1][0][2][1]
            g2_i3 = population_[i * 2][0][2][1]
        else:
            g1_i3 = population_[i * 2][0][2][1]
            g2_i3 = population_[i * 2 + 1][0][2][1]

        if random.random() < pc:
            g1_i4 = population_[i * 2 + 1][0][2][2]
            g2_i4 = population_[i * 2][0][2][2]
        else:
            g1_i4 = population_[i * 2][0][2][2]
            g2_i4 = population_[i * 2 + 1][0][2][2]

        if random.random() < pc:
            g1_i5 = population_[i * 2 + 1][0][2][3]
            g2_i5 = population_[i * 2][0][2][3]
        else:
            g1_i5 = population_[i * 2][0][2][3]
            g2_i5 = population_[i * 2 + 1][0][2][3]

        # if random.random() < pc:
        #     g1_f = population_[i * 2 + 1][0][2][4]
        #     g2_f = population_[i * 2][0][2][4]
        # else:
        #     g1_f = population_[i * 2][0][2][4]
        #     g2_f = population_[i * 2 + 1][0][2][4]

        if random.random() < pc:
            g1_l = population_[i * 2 + 1][0][2][5]
            g2_l = population_[i * 2][0][2][5]
        else:
            g1_l = population_[i * 2][0][2][5]
            g2_l = population_[i * 2 + 1][0][2][5]

        if random.random() < pc:
            g1_fc = population_[i * 2 + 1][0][2][6]
            g2_fc = population_[i * 2][0][2][6]
        else:
            g1_fc = population_[i * 2][0][2][6]
            g2_fc = population_[i * 2 + 1][0][2][6]
        # geno2 end
        p1 = Genotype(
            geno1=g1_0,
            geno2=Decoder(
                unit1=g1_u1,
                unit2=g1_u2,
                unit3=g1_u3,
                unit4=g1_u4,
                unit5=g1_u5,
            ),
            geno3=[g1_i2, g1_i3, g1_i4, g1_i5, g1_f, g1_l, g1_fc]
        )
        # p1 = filter(p1)
        popu.append([p1, INF])
        p2 = Genotype(
            geno1=g2_0,
            geno2=Decoder(
                unit1=g2_u1,
                unit2=g2_u2,
                unit3=g2_u3,
                unit4=g2_u4,
                unit5=g2_u5,
            ),
            geno3=[g2_i2, g2_i3, g2_i4, g2_i5, g2_f, g2_l, g2_fc]
        )
        # p2 = filter(p2)
        popu.append([p2, INF])
    return popu


def filter(pop):
    geno = copy.deepcopy(pop)
    if geno[0][0][1] == 0:
        geno[0][2][0] = [0]
    if geno[0][0][2] == 0:
        geno[0][2][1] = [0, 0]
    if geno[0][0][3] == 0:
        geno[0][2][2] = [0, 0]
    if geno[0][0][4] == 0:
        geno[0][2][3] = [0, 0]
    # geno[2][0] = brush([geno[0][0]], geno[2][0])
    # geno[2][1] = brush(geno[0][:2], geno[2][1])
    # geno[2][2] = brush(geno[0][1:3], geno[2][2])
    # geno[2][3] = brush(geno[0][2:4], geno[2][3])
    # geno[2][4] = brush(geno[0][:], geno[2][4])
    # geno[2][5] = brush(geno[0][:], geno[2][5])
    return geno


def mutate(population, pm=0.2):
    pm_ = pm
    population_ = copy.deepcopy(population)
    for i in range(len(population_)):
        # mutate geno1
        while True:
            popu = copy.deepcopy(population_[i])
            popu = filter(popu)
            while True:
                for id in range(5):
                    if random.random() < pm:
                        # id = random.sample(range(5), 1)[0]
                        # population_[i][0][0][id] = int(0 == population_[i][0][0][id])
                        popu[0][0][id] = int(0 == popu[0][0][id])
                if sum(popu[0][0]) > 0:
                    break
            # mutate geno1 end

            # mutate geno2
            for n in range(5):  # 5 units
                for ind_n in range(4):
                    if random.random() < pm:
                        # ind_n = random.randint(0, 3)  # select one out of 4 nodes each unit
                        op = random.sample(PRIMITIVES, 1)[0]  # mutate operation
                        edge = random.sample(range(ind_n + 1), 1)[0]  # mutate edge
                        node = (op, edge)
                        popu[0][1][n][ind_n] = node
            # mutate geno2 end

            # mutate geno3 begin
            # mutate outputs of cell1
            if popu[0][0][0] == 0:
                popu[0][2][0][0] = 0
                popu[0][2][1][0] = 0
                popu[0][2][4][0] = 0
                popu[0][2][5][0] = 0
            else:
                while True:
                    cmt = []
                    if random.random() < pm:
                        if popu[0][0][1] >= int(0 == popu[0][2][0][0]):
                            popu[0][2][0][0] = int(0 == popu[0][2][0][0])
                            cmt.append(popu[0][2][0][0])
                    if random.random() < pm:
                        if popu[0][0][2] >= int(0 == popu[0][2][1][0]):
                            popu[0][2][1][0] = int(0 == popu[0][2][1][0])
                            cmt.append(popu[0][2][1][0])
                    if random.random() < pm_:
                        popu[0][2][4][0] = int(0 == popu[0][2][4][0])
                        cmt.append(popu[0][2][4][0])
                    if random.random() < pm_:
                        popu[0][2][5][0] = int(0 == popu[0][2][5][0])
                        cmt.append(popu[0][2][5][0])
                    if sum(cmt) >= 1:
                        break

            ## mutate outputs of cell2
            if popu[0][0][1] == 0:
                popu[0][2][1][1] = 0
                popu[0][2][2][0] = 0
                popu[0][2][4][1] = 0
                popu[0][2][5][1] = 0
            else:
                while True:
                    cmt = []
                    if random.random() < pm:
                        if popu[0][0][2] >= int(0 == popu[0][2][1][1]):
                            popu[0][2][1][1] = int(0 == popu[0][2][1][1])
                            cmt.append(popu[0][2][1][1])
                    if random.random() < pm:
                        if popu[0][0][3] >= int(0 == popu[0][2][2][0]):
                            popu[0][2][2][0] = int(0 == popu[0][2][2][0])
                            cmt.append(popu[0][2][2][0])
                    if random.random() < pm_:
                        popu[0][2][4][1] = int(0 == popu[0][2][4][1])
                        cmt.append(popu[0][2][4][1])
                    if random.random() < pm_:
                        popu[0][2][5][1] = int(0 == popu[0][2][5][1])
                        cmt.append(popu[0][2][5][1])
                    if sum(cmt) >= 1:
                        break

            ## mutate outputs of cell3
            if popu[0][0][2] == 0:
                popu[0][2][2][1] = 0
                popu[0][2][3][0] = 0
                popu[0][2][4][2] = 0
                popu[0][2][5][2] = 0
            else:
                while True:
                    cmt = []
                    if random.random() < pm:
                        if popu[0][0][3] >= int(0 == popu[0][2][2][1]):
                            popu[0][2][2][1] = int(0 == popu[0][2][2][1])
                            cmt.append(popu[0][2][2][1])
                    if random.random() < pm:
                        if popu[0][0][4] >= int(0 == popu[0][2][3][0]):
                            popu[0][2][3][0] = int(0 == popu[0][2][3][0])
                            cmt.append(popu[0][2][3][0])
                    if random.random() < pm_:
                        popu[0][2][4][2] = int(0 == popu[0][2][4][2])
                        cmt.append(popu[0][2][4][2])
                    if random.random() < pm_:
                        popu[0][2][5][2] = int(0 == popu[0][2][5][2])
                        cmt.append(popu[0][2][5][2])
                    if sum(cmt) >= 1:
                        break

            ## mutate outputs of cell4
            if popu[0][0][3] == 0:
                popu[0][2][3][1] = 0
                popu[0][2][4][3] = 0
                popu[0][2][5][3] = 0
            else:
                while True:
                    cmt = []
                    if random.random() < pm:
                        if popu[0][0][4] >= int(0 == popu[0][2][3][1]):
                            popu[0][2][3][1] = int(0 == popu[0][2][3][1])
                            cmt.append(popu[0][2][3][1])
                    if random.random() < pm_:
                        popu[0][2][4][3] = int(0 == popu[0][2][4][3])
                        cmt.append(popu[0][2][4][3])
                    if random.random() < pm_:
                        popu[0][2][5][3] = int(0 == popu[0][2][5][3])
                        cmt.append(popu[0][2][5][3])
                    if sum(cmt) >= 1:
                        break

            ## mutate outputs of cell5
            if popu[0][0][4] == 0:
                popu[0][2][4][4] = 0
                popu[0][2][5][4] = 0
            else:
                while True:
                    cmt = []
                    if random.random() < pm_:
                        popu[0][2][4][4] = int(0 == popu[0][2][4][4])
                        cmt.append(popu[0][2][4][4])
                    if random.random() < pm_:
                        popu[0][2][5][4] = int(0 == popu[0][2][5][4])
                        cmt.append(popu[0][2][5][4])
                    if sum(cmt) >= 1:
                        break

            # if random.random() < pm:
            #     # input of cell3
            #     while True:
            #         if sum(population_[i][0][0][:2]) == 0:
            #             break
            #         id = random.sample(range(2), 1)[0]
            #         if population_[i][0][0][id] == 1 or 1 == population_[i][0][2][1][id]:
            #             population_[i][0][2][1][id] = int(0 == population_[i][0][2][1][id])
            #             break
            # if random.random() < pm:
            #     # input of cell4
            #     while True:
            #         if sum(population_[i][0][0][1:3]) == 0:
            #             break
            #         id = random.sample(range(1, 3), 1)[0]
            #         if population_[i][0][0][id] == 1 or 1 == population_[i][0][2][2][id - 1]:
            #             population_[i][0][2][2][id - 1] = int(0 == population_[i][0][2][2][id - 1])
            #             break
            # if random.random() < pm:
            #     # input of cell5
            #     while True:
            #         if sum(population_[i][0][0][2:4]) == 0:
            #             break
            #         id = random.sample(range(2, 4), 1)[0]
            #         if population_[i][0][0][id] == 1 or 1 == population_[i][0][2][3][id - 2]:
            #             population_[i][0][2][3][id - 2] = int(0 == population_[i][0][2][3][id - 2])
            #             break
            # if random.random() < pm:
            #     # input of fuse_cell
            #     while True:
            #         id = random.sample(range(5), 1)[0]
            #         if population_[i][0][0][id] == 1 or 1 == population_[i][0][2][4][id]:
            #             population_[i][0][2][4][id] = int(0 == population_[i][0][2][4][id])
            #             if sum(population_[i][0][2][4]) > 0:
            #                 break
            # if random.random() < pm:
            #     # input of loss
            #     id = random.sample(range(5), 1)[0]
            #     if population_[i][0][0][id] == 1 or 1 == population_[i][0][2][5][id]:
            #         population_[i][0][2][5][id] = int(0 == population_[i][0][2][5][id])
            #         break

            if random.random() < pm:
                ind_n = random.randint(0, 3)  # select one out of 4 nodes each unit
                op = random.sample(PRIMITIVES, 1)[0]  # mutate operation
                edge = random.sample(range(ind_n + 1), 1)[0]  # mutate edge
                node = (op, edge)
                popu[0][2][6][ind_n] = node
            # mutate geno3 end
            if sum(popu[0][2][4]) >= 1:
                population_[i] = popu
                break
    return population_


def brush(arr1, arr2):
    for i in range(len(arr2)):
        if arr2[i] > arr1[i]:
            print('brush')
            arr2[i] = 0
    return arr2


def top_k(population, num=None):
    population = sorted(population, key=lambda x: x[1])
    if not num:
        return population[:len(population) // 2]
    else:
        return population[:num]


def best_p(population):
    p = sorted(population, key=lambda x: x[1])[0]
    return p


g = Genotype(
    geno1=[0, 1, 0, 0, 1],
    geno2=Decoder(
        unit1=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
        unit2=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
        unit3=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
        unit4=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
        unit5=[('conv3', 0), ('dconv3_2', 1), ('dconv3_4', 2), ('dconv3_8', 3)],
    ),
    geno3=[[0], [0, 1], [1, 0], [0, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0],
           [('conv3', 0), ('dconv3_2', 1), ('dconv3_2', 2), ('dconv3_2', 3)]]
)


