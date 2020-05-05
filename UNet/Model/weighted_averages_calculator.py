import numpy as np

'''
b1, v1_val_acc, v1_test_acc, b2, v2_val_acc, v2_test_acc, b3, v3_val_acc, v3_test_acc, b4, v4_val_acc, v4_test_acc degerler degistirildikce, agirlikli tum ortalamalar bu script sayesinde hesaplanabilir.
'''

val_num =  np.array([16,19,17,20,19,19,17,19,18,17])
test_num = np.array([14,17,16,18,17,17,15,17,17,15])

a1 =          np.array([186,	203,	207,	196,	223,	213,	175,	209,	212,	197])
b1 =          np.array([0.66,	0.75,	0.79,	0.79,	0.82,	0.80,	0.78,	0.76,	0.67,	0.78])
v1_val_acc =  np.array([0.59,	0.72,	0.78,	0.77,	0.80,	0.76,	0.76,	0.74,	0.66,	0.76])
v1_test_acc = np.array([0.55,	0.78,	0.78,	0.81,	0.82,	0.83,	0.78,	0.75,	0.65,	0.80])

a2 =          np.array([354,	419,	395,	431,    445,    427,	371,    420,	406,	370])
b2 =          np.array([0.63,	0.75,	0.75,   0.82,	0.80,   0.81,	0.79,   0.77,   0.64,	0.75])
v2_val_acc =  np.array([0.63,	0.74,	0.74,	0.79,	0.78,	0.78,	0.76,	0.76,	0.64,	0.78])
v2_test_acc = np.array([0.56,	0.78,	0.75,	0.84,	0.79,	0.82,	0.80,	0.80,	0.61,	0.78])

a3 =          np.array([536,	639,	587,	657,	643,	627,	571,	638,	606,	573])
b3 =          np.array([0.66,	0.77,	0.79,	0.82,	 0.82,	0.82,	0.80,	0.78,	0.67,	0.79])
v3_val_acc =  np.array([0.65,	0.73,	0.77,	0.78,	0.81,	0.78,	0.78,	0.78,	0.66,	0.80])
v3_test_acc = np.array([0.56,	0.80,	0.78,	0.84,	0.82,	0.85,	0.77,	0.80,	0.64,	0.81])

a4 =          np.array([703,	852,	769,	893,	857,	857,	757,	849,    821,	745])
b4 =          np.array([0.71,	0.79,	0.77,	0.86,	0.82,	0.83,	0.81,	0.80,	0.66,	0.82])
v4_val_acc =  np.array([0.71,	0.77,	0.76,	0.82,	0.81,	0.79,	0.79,	0.79,	0.65,	0.83])
v4_test_acc = np.array([0.63,	0.80,	0.76,	0.86,	0.83,	0.85,	0.80,	0.82,	0.63,	0.83])

print(np.sum(np.multiply(a1,b1)) / 2200)
print(np.sum(np.multiply(v1_val_acc, val_num)) / 181)
print(np.sum(np.multiply(v2_val_acc, val_num)) / 181)

print()

print(np.sum(np.multiply(a2,b2)) / 4399)
print(np.sum(np.multiply(v3_val_acc, val_num)) / 181)
print(np.sum(np.multiply(v4_val_acc, val_num)) / 181)

print()

print(np.sum(np.multiply(a3,b3)) / 6598)
print(np.sum(np.multiply(v1_test_acc,test_num)) / 163)
print(np.sum(np.multiply(v2_test_acc,test_num)) / 163)

print()

print(np.sum(np.multiply(a4,b4)) / 8799)
print(np.sum(np.multiply(v3_test_acc,test_num)) / 163)
print(np.sum(np.multiply(v4_test_acc,test_num)) / 163)



