import random
import collections
out=open('dataset_PF.py','w')
out.write('import numpy as np\n')
out.write(
"out=open('dataset_PF.csv','w')\n"
#index = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
"zero = '0,0,0,0'\n"
"FA = [0 for _ in range(35)]\n"
"FA[20]='4.502, 0, 0'; FA[21]='4.502, 6.01, 4'; FA[22]='4.502, 6.01, 8'; FA[23]='4.502, 6.01, 12'; FA[24]='4.502, 6.01, 20'\n"
"FA[25]='4.498, 6.07, 4'; FA[26]='4.498, 6.07, 12'; FA[27]='4.498, 6.07, 16'; FA[28]='4.498, 6.07, 20'; FA[29]='4.498, 6.07, 24'\n"
"FA[30]='4.638, 8.11, 4'; FA[31]='4.638, 8.11, 12'; FA[32]='4.638, 8.11, 16'; FA[33]='4.638, 8.11, 20'; FA[34]='2.211, 0, 0'\n")
out.close()

num = 10

for n in range (1,num+1):
    condition = 1
    f = open(r'C:\Users\hyunbin\Desktop\MachineLearning\optimization\test\rastk_test_%d.out' %(n),mode='rt', encoding='utf-8')
    isinstance(f, collections.Iterable)
    for line in f:
        if line.find('PF_2D')>0:
            out=open('dataset_PF.py','a')
            #out.write('for i in range(1,%d):\n' %(num+1))
            out.write('PF_%d = [' %(n))
            for i in range(1,50):
                line = f.readline()
                if line[1:2] == '=':
                    out.write(']\n')
                    out.write('PF_%d = np.max(PF_%d)\n' %(n,n))
                    break
                if line[41:45] == '0.00':
                    condition = 0
                    break
                if i != 1:
                    out.write('%s,' %(line[41:50]))
            out.close()
            if condition == 0:
                break
            
        if line.find('Pattern')>0:
            out=open('dataset_PF.py','a')
            #out.write('for i in range(1,%d):\n' %(num+1))
            out.write('LP_%d = [' %(n))
            for i in range(1,9):
                line = f.readline()
                if i in range(1,3):
                    out.write('%s,' %(line[2:4]))
                    out.write('%s,' %(line[6:8]))
                    out.write('%s,' %(line[10:12]))
                    out.write('%s,' %(line[14:16]))
                    out.write('%s,' %(line[18:20]))
                    out.write('%s,' %(line[22:24]))
                    out.write('%s,' %(line[26:28]))
                    out.write('%s,' %(line[30:32]))
                elif i in range(3,5):
                    out.write('%s,' %(line[2:4]))
                    out.write('%s,' %(line[6:8]))
                    out.write('%s,' %(line[10:12]))
                    out.write('%s,' %(line[14:16]))
                    out.write('%s,' %(line[18:20]))
                    out.write('%s,' %(line[22:24]))
                    out.write('%s,' %(line[26:28]))
                elif i in range(5,6):
                    out.write('%s,' %(line[2:4]))
                    out.write('%s,' %(line[6:8]))
                    out.write('%s,' %(line[10:12]))
                    out.write('%s,' %(line[14:16]))
                    out.write('%s,' %(line[18:20]))
                    out.write('%s,' %(line[22:24]))
                elif i in range(6,7):
                    out.write('%s,' %(line[2:4]))
                    out.write('%s,' %(line[6:8]))
                    out.write('%s,' %(line[10:12]))
                    out.write('%s,' %(line[14:16]))
                    out.write('%s,' %(line[18:20]))
                elif i in range(7,8):
                    out.write('%s,' %(line[2:4]))
                    out.write('%s,' %(line[6:8]))
                    out.write('%s,' %(line[10:12]))
                    out.write('%s,' %(line[14:16]))
                else:
                    out.write('%s,' %(line[2:4]))
                    out.write('%s]\n' %(line[6:8]))
            out.close()
            
        if line.find('BURNUP')>0:
            out=open('dataset_PF.py','a')
            out.write('BN_%d = [' %(n))
            for i in range(1,13):
                line = f.readline()
                if i in range(5,7):
                    out.write('%s,' %(line[1:9]))
                    out.write('%s,' %(line[10:18]))
                    out.write('%s,' %(line[19:27]))
                    out.write('%s,' %(line[28:36]))
                    out.write('%s,' %(line[37:45]))
                    out.write('%s,' %(line[46:54]))
                    out.write('%s,' %(line[55:63]))
                    out.write('%s,' %(line[64:72]))
                elif i in range(7,9):
                    out.write('%s,' %(line[1:9]))
                    out.write('%s,' %(line[10:18]))
                    out.write('%s,' %(line[19:27]))
                    out.write('%s,' %(line[28:36]))
                    out.write('%s,' %(line[37:45]))
                    out.write('%s,' %(line[46:54]))
                    out.write('%s,' %(line[55:63]))
                elif i in range(9,10):
                    out.write('%s,' %(line[1:9]))
                    out.write('%s,' %(line[10:18]))
                    out.write('%s,' %(line[19:27]))
                    out.write('%s,' %(line[28:36]))
                    out.write('%s,' %(line[37:45]))
                    out.write('%s,' %(line[46:54]))
                elif i in range(10,11):
                    out.write('%s,' %(line[1:9]))
                    out.write('%s,' %(line[10:18]))
                    out.write('%s,' %(line[19:27]))
                    out.write('%s,' %(line[28:36]))
                    out.write('%s,' %(line[37:45]))
                elif i in range(11,12):
                    out.write('%s,' %(line[1:9]))
                    out.write('%s,' %(line[10:18]))
                    out.write('%s,' %(line[19:27]))
                    out.write('%s,' %(line[28:36]))
                elif i in range(12,13):
                    out.write('%s,' %(line[1:9]))
                    out.write('%s]\n' %(line[10:18]))

            out.write(
            'out.write('
            '"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'
             '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'
             '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'
             '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'
             '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'
             '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'
             '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'
             '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'
             '%s\\n"\n'
            '    %(')
            out.write(
            'FA[LP_%d[0]],BN_%d[0],FA[LP_%d[1]],BN_%d[1],FA[LP_%d[2]],BN_%d[2],FA[LP_%d[3]],BN_%d[3],FA[LP_%d[4]],BN_%d[4],FA[LP_%d[5]],BN_%d[5],FA[LP_%d[6]],BN_%d[6],FA[LP_%d[7]],BN_%d[7],'
            'FA[LP_%d[8]],BN_%d[8],FA[LP_%d[9]],BN_%d[9],FA[LP_%d[10]],BN_%d[10],FA[LP_%d[11]],BN_%d[11],FA[LP_%d[12]],BN_%d[12],FA[LP_%d[13]],BN_%d[13],FA[LP_%d[14]],BN_%d[14],FA[LP_%d[15]],BN_%d[15],'
            'FA[LP_%d[16]],BN_%d[16],FA[LP_%d[17]],BN_%d[17],FA[LP_%d[18]],BN_%d[18],FA[LP_%d[19]],BN_%d[19],FA[LP_%d[20]],BN_%d[20],FA[LP_%d[21]],BN_%d[21],FA[LP_%d[22]],BN_%d[22],zero,'
            'FA[LP_%d[23]],BN_%d[23],FA[LP_%d[24]],BN_%d[24],FA[LP_%d[25]],BN_%d[25],FA[LP_%d[26]],BN_%d[26],FA[LP_%d[27]],BN_%d[27],FA[LP_%d[28]],BN_%d[28],FA[LP_%d[29]],BN_%d[29],zero,'
            'FA[LP_%d[30]],BN_%d[30],FA[LP_%d[31]],BN_%d[31],FA[LP_%d[32]],BN_%d[32],FA[LP_%d[33]],BN_%d[33],FA[LP_%d[34]],BN_%d[34],FA[LP_%d[35]],BN_%d[35],zero,zero,'
            'FA[LP_%d[36]],BN_%d[36],FA[LP_%d[37]],BN_%d[37],FA[LP_%d[38]],BN_%d[38],FA[LP_%d[39]],BN_%d[39],FA[LP_%d[40]],BN_%d[40],zero,zero,zero,'
            'FA[LP_%d[41]],BN_%d[41],FA[LP_%d[42]],BN_%d[42],FA[LP_%d[43]],BN_%d[43],FA[LP_%d[44]],BN_%d[44],zero,zero,zero,zero,'
            'FA[LP_%d[45]],BN_%d[45],FA[LP_%d[46]],BN_%d[46],zero,zero,zero,zero,zero,zero,PF_%d))\n'
            %(n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n))
            out.close()
    f.close()

out=open('dataset_PF.py','a')
out.write('out.close()')
out.close()
