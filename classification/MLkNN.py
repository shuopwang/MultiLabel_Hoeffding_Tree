from skmultilearn.adapt import MLkNN
import numpy as np
class MLkNN():
	def __init__(self,window_size=100):
		self.h=MLkNN(k=20)
		self.window_size=window_size
		self.window=InstanceWindow(window_size)
		self.number_element=0
		self.flag=False
		self.L=None

	def partial_fit(self,X,y):
		N,L=y.shape
		self.L=L
		for i in range(N):
			if self.window=None:
				self.window=InstanceWindow(self.window_size)
			self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
			self.number_element+=1
			if self.number_element==self.window_size:
				X_batch=self.window.get_attributes_matrix()
				y_batch=self.window.get_targets_matrix()
				self.h.fit(X_batch,y_batch)
				self.number_element=0
				self.flag=True

	def predict(self,X):
		N,D=X.shape
		if self.flag:
			return self.h.predict(X)
		else:
			return zeros(N)