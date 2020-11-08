from collections import namedtuple

Genotype = namedtuple('Genotype', 'geno1 geno2 geno3')

Decoder = namedtuple('Decoder', 'unit1, unit2, unit3, unit4 unit5')

PRIMITIVES = [
    'skip',  # 0
    'conv1',
    'conv3',  # 1
    'conv5',  # 2
    'dconv3_2',
    'dconv3_4',
    'dconv3_8',
    'dconv5_2',
    'dconv5_4',
    'dconv3_16',
    'dconv3_32',
    'dconv5_8',
    'dconv5_16',
    'dconv5_32',

]


S_setting = Genotype(geno1=[1, 1, 1, 1, 0],
                     geno2=Decoder(unit1=[('conv5', 0)],
                                   unit2=[('dconv3_2', 1)],
                                   unit3=[('conv3', 0)],
                                   unit4=[('dconv5_4', 0)],
                                   unit5=[('dconv5_2', 0)]),
                     geno3=[[1], [1, 1], [0, 0], [1, 0], [0, 1, 1, 0, 1], [0, 0, 0, 0, 0],
                            [('dconv3_2', 1)]])

M_setting = Genotype(geno1=[1, 1, 1, 1, 0],
                     geno2=Decoder(unit1=[('dconv3_2', 0), ('skip', 1), ('conv3', 2), ('conv5', 3)],
                                   unit2=[('dconv3_4', 0), ('dconv3_4', 0), ('skip', 2), ('conv5', 3)],
                                   unit3=[('conv3', 0), ('dconv5_4', 1), ('conv5', 2), ('dconv5_2', 2)],
                                   unit4=[('dconv3_4', 0), ('conv3', 1), ('dconv5_2', 0), ('dconv5_4', 1)],
                                   unit5=[('conv1', 0), ('conv5', 0), ('dconv5_2', 2), ('dconv5_4', 3)]),
                     geno3=[[1], [0, 1], [1, 1], [0, 0], [0, 0, 1, 1, 0], [0, 1, 0, 0, 0],
                            [('dconv5_4', 0), ('conv3', 1), ('conv3', 2),
                             ('conv5', 2)]])

L_setting = Genotype(geno1=[1, 1, 1, 0, 0],
                     geno2=Decoder(unit1=[('skip', 0), ('conv5', 0), ('dconv5_8', 2), ('dconv3_4', 3)],
                                   unit2=[('dconv3_2', 0), ('dconv5_4', 0), ('conv3', 1), ('conv5', 3)],
                                   unit3=[('dconv5_2', 0), ('dconv5_2', 0), ('conv1', 1), ('dconv3_2', 2)],
                                   unit4=[('skip', 0), ('skip', 1), ('skip', 0), ('skip', 2)],
                                   unit5=[('skip', 0), ('skip', 1), ('skip', 0), ('skip', 2)]),
                     geno3=[[1], [1, 1], [0, 0], [0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0],
                            [('dconv5_2', 0), ('conv5', 1), ('conv1', 2), ('conv5', 3)]])

geno_inception = Genotype(geno1=[1, 1, 1, 0, 0],
                          geno2=Decoder(unit1=[('dconv5_8', 0), ('skip', 1), ('conv5', 0), ('conv5', 0)],
                                        unit2=[('dconv3_8', 0), ('dconv3_4', 0), ('skip', 1), ('conv5', 0)],
                                        unit3=[('conv3', 0), ('dconv5_8', 1), ('conv5', 2), ('dconv5_2', 2)],
                                        unit4=[('skip', 0), ('skip', 1), ('skip', 0), ('skip', 2)],
                                        unit5=[('skip', 0), ('skip', 1), ('skip', 0), ('skip', 2)]),
                          geno3=[[1], [0, 1], [0, 0], [0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0],
                                 [('dconv3_2', 0), ('dconv5_4', 1), ('dconv5_8', 2), ('conv1', 0)]])
