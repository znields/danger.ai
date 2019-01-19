import numpy as np




#each of these are going to have shape (_,1)
#where _ is the frame# or index


dangerscore_w2v = None

dangerscore_pxd = None

dangerscore_rnn = None





#now to try different methods of ensembling these scores together.

def avg(ds1, ds2, ds3):
	#they need to be combined into (_,3)
	
	#returns (_,)
	return np.average(np.reshape(np.stack((ds1, ds2, ds3), axis=1), (31,3)), axis=1)
	
	
	
def linear_combination(ds1, ds2, ds3, x, y, z):
	return ds1*x + ds2*y + ds3*z

