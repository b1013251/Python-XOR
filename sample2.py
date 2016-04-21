# coding: utf-8

import numpy as np
import math

# 教師データ
train = (
	(0,0,0),
	(0,1,1),
	(1,0,1),
	(1,1,0)
)

# 列挙型もどき
( INPUT, HIDDEN, OUTPUT ) = range(3)

class Calc :
	# シグモイド関数
	@classmethod
	def sigmoid(self, x ) :
		return 1.0 / (1.0 + math.exp(-x))

	# シグモイドの微分
	@classmethod
	def sigmoid_dash(self, x) :
		return (1 - self.sigmoid(x)) * self.sigmoid(x)

class Path :
	def __init__(self,weight,back,forward):
		self.back = back
		self.forward = forward
		self.weight = weight
		self.delta = 0

class Neuron :
	def __init__(self,value,ntype,num):
		self.value = value
		self.raw_value = 0
		self.neuron_type = ntype
		self.num = num

	def search_back(self,num,pathes) :
		results = []
		for p in pathes :
			if p.forward == num :
				results.append(p)
		return results

	def search_forward(self,num,pathes) :
		results = []
		for p in pathes :
			if p.back == num :
				results.append(p)
		return results

	def calc(self,nn) :
		if self.neuron_type == INPUT :
			return

		result = 0
		back_pathes = self.search_back(self.num, nn.pathes)
		for p in back_pathes :
			#print "result +", p.weight , "*", nn.neurons[p.forward].value, "(", nn.neurons[p.forward].num ,")"
			result += p.weight * nn.neurons[p.back].value

		self.raw_value = result
		self.value = Calc.sigmoid(result)

class NN :
	def __init__(self,input_num,hidden_num,output_num) :
		# ニューロンの初期化
		self.input_num = input_num
		self.hidden_num = hidden_num
		self.output_num = output_num
		self.neurons = []
		for i in range(input_num) :
			self.neurons.append(Neuron(1,INPUT,i))
		for i in range(hidden_num) :
			self.neurons.append(Neuron(1,HIDDEN,i + hidden_num - 1))
		for i in range(output_num) :
			self.neurons.append(Neuron(1,OUTPUT,i + hidden_num + output_num ))

		# パスの初期化
		self.pathes = []
		for i in range(input_num ) :
			for j in range(hidden_num) :
				self.pathes.append(Path(1,i,hidden_num + j - 1))
		for i in range(hidden_num) :
			for j in range(output_num) :
				self.pathes.append(Path(1,hidden_num + i - 1, hidden_num + output_num + j ))

	# デバッグ用
	def debug(self) :
		print "Neurons"
		print "number | value | TYPE"
		for n in self.neurons :
			print n.num, n.value, n.neuron_type

		print "Pathes"
		print "W  |back  |forward"
		for p in self.pathes :
			print p.weight, p.back, p.forward

	# 2つの出力をタプルで返す
	def output(self) :
		for n in self.neurons :
			n.calc(self)
		y1 = self.neurons[self.input_num + self.hidden_num ].value
		y2 = self.neurons[self.input_num + self.hidden_num + 1].value

		return (y1, y2)


	# デルタ
	def output_delta(self, path, train, value, raw_value) :
		path.delta = (train[2] - value) * Calc.sigmoid_dash(raw_value)
		print "path.delta", path.delta

	def hidden_delta(self, pathes) :
		neuron = nn.neurons[pathes.back]
		sum = 0
		for p in pathes :
			sum += p.delta * p.weight
		delta = Calc.sigmoid_dash(neuron.raw_value)


	# 誤差逆伝搬を行う
	def bp_step(self,train) :
		for n in reversed(self.neurons) :
			if n.neuron_type == OUTPUT :
				a_path = n.search_back(n.num,self.pathes)[0]
				self.output_delta(a_path,train,n.value,n.raw_value)
			if n.neuron_type == HIDDEN :
				print "hidden"
			if n.neuron_type == INPUT :
				print "input"

	# 誤差逆伝播を繰り返す
	def backpropergation(self,n) :
		for step in xrange(0, n + 1) :
			self.bp_step(train[0])
			if step % (n / 10) == 0:
				print "step:", step


if __name__ == "__main__":
	print "XOR by NN"

	nn = NN(2,3,2)
	y = nn.output()
	print y

	nn.backpropergation(100)
