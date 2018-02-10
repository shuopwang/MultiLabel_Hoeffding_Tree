import numpy as np
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import copy
class BR(object):
	def __init__(self,h=SGDClassifier()):
		self.c=h
		self.number_label=0
		self.H=[]


	def fit(self,X,y):
		N,L=y.shape
		self.number_label=L
		for j in range(self.number_label):
			sub_classficaiton=copy.deepcopy(self.c)
			sub_classficaiton.fit(X,y[:,j])
			self.H.append(sub_classficaiton)

		return self


	def partial_fit(self,X,y, classes=None):
		N,L=y.shape

		for j in range(self.L):
			self.H[j].partial_fit[X,y[:,j]]

		return self

	def predict(self,X):
		N,F=X.shape
		y_pred=zeros(len(H))
		for j in range(len(H)):
			y_pred[:,j]=self.H[j].predict(X)
		return y_pred

class LC(object):
	"""docstring for LC"""
	def __init__(self, h=SGDClassifier()):
		self.h=copy.deepcopy(h)
		self.number_label=-1

	def fit(self,X,y,classes=None):
		N,self.number_label=y.shape
		y,self.reverse=self.transform(y)
		self.h.fit(X,y)
		return self

	def partial_fit(self,X,y,classes=None):
		N,self.number_label=y.shape
		y,self.reverse=self.transform(y,N)
		self.h.partial_fit(X,y)
		return self

	def predict(self,X):
		predict_y=self.h.predict(X)
		Y=self.transform_predict(predict_y)
		return Y

	def transform(self,y,N):
		New_Y=zeros(N)
		mapping={}
		reverse={}
		c=-1
		for i in range(N):
			k=str(y[i])
			if k not in mapping:
				c=c+1
				mapping[k]=c
				reverse[c]=array(Y[i,:])
			New_Y=mapping[k]
		return New_Y,reverse

	def transform_predict(self,Y):
		N=len(Y)
		real_Y=zeros((N,self.number_label),dtype=int)
		for i in range(N):
			real_Y[i,:]=self.reverse(Y[i])
		return real_Y


class PW(object):
	"""docstring for LC"""
	def __init__(self, ):
		pass
		
	def fit(self,X,y):
		pass

	def partial_fit(self,X,y):
		pass

	def predict(self,X):
		pass
		