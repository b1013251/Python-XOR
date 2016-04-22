# coding: utf-8

import numpy as np
import math
import random

# パラメータ
eta  = 1.0
alpha = 0.1

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

	# [0-1]の乱数を返す
	@classmethod
	def rand(self) :
		 return random.random()

class Path :
	def __init__(self,weight,back,forward):
		self.back = back
		self.forward = forward
		self.weight = weight
		self.delta = 0
		self.deltaW = 0

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
			if p.back == (-1) :
				result += p.weight
			else :
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
			self.neurons.append(Neuron(Calc.rand(),INPUT,i))
		for i in range(hidden_num) :
			self.neurons.append(Neuron(Calc.rand(),HIDDEN,i + hidden_num - 1))
		for i in range(output_num) :
			self.neurons.append(Neuron(Calc.rand(),OUTPUT,i + hidden_num + input_num ))

		# パスの初期化
		self.pathes = []
		# 1-2
		for i in range(input_num ) :
			for j in range(hidden_num) :
				self.pathes.append(Path(Calc.rand(),i,hidden_num + j - 1))
		# 2-3
		for i in range(hidden_num) :
			for j in range(output_num) :
				self.pathes.append(Path(Calc.rand(),hidden_num + i - 1, hidden_num + input_num + j ))
		# threshold
		for i in range(hidden_num + output_num) :
			self.pathes.append(Path(Calc.rand(), -1, hidden_num + i - 1 ))

	# デバッグ用
	def debug(self) :
		print "Neurons"
		print "number | value |  TYPE"
		for n in self.neurons :
			print n.num, n.value , n.neuron_type

		print "Pathes"
		print "W  |back  |forward | delta"
		for p in self.pathes :
			print p.weight, p.back, p.forward, p.delta

	# 2つの出力をタプルで返す
	def output(self,data) :
		self.neurons[0].value = data[0]
		self.neurons[1].value = data[1]

		for n in self.neurons :
			n.calc(self)
		y = self.neurons[self.input_num + self.hidden_num ].value

		return y


	# デルタ
	def output_delta(self, b_pathes, train, value, raw_value) :
		for bp in b_pathes :
			bp.delta = (train[2] - value) * Calc.sigmoid_dash(raw_value)
			#print "op! bp.delta", bp.delta

	def hidden_delta(self, f_pathes, b_pathes, value, raw_value) :
		for bp in b_pathes :
			sum = 0
			for fp in f_pathes :
				#print fp.delta, fp.weight
				sum += fp.delta * fp.weight
			bp.delta = sum * Calc.sigmoid_dash(raw_value)
			#print "sum", sum , "bp.delta", bp.delta

	# 誤差逆伝搬を行う
	def bp_step(self,train) :
		for n in reversed(self.neurons) :
			if n.neuron_type == OUTPUT :
				b_pathes = n.search_back(n.num,self.pathes)
				self.output_delta(b_pathes,train,n.value,n.raw_value)
			if n.neuron_type == HIDDEN :
				f_pathes = n.search_forward(n.num,self.pathes)
				b_pathes = n.search_back(n.num,self.pathes)
				self.hidden_delta(f_pathes, b_pathes, n.value, n.raw_value)

	# 誤差逆伝播を繰り返す
	def backpropergation(self,n) :
		for step in xrange(0, n + 1) :
			for t in train :
				self.output((t[0],t[1]))
				self.bp_step(t)
				self.modify_error()
			if step % (n / 10) == 0:
				print "step:", step
				self.out_test()

	# 誤差修正
	def modify_error(self) :
		for p in self.pathes :
			back_n = self.neurons[p.back]
			p.deltaW = p.delta * eta * back_n.value + alpha * p.deltaW
			p.weight = p.weight + p.deltaW


	# 出力テスト
	def out_test (self) :
		print "results"
		print "----------"
		y = nn.output((0,0))
		print y
		y = nn.output((0,1))
		print y
		y = nn.output((1,0))
		print y
		y = nn.output((1,1))
		print y

if __name__ == "__main__":
	print "XOR by NN"

	nn = NN(2,3,1)
	y = nn.output((0,1))
	print y

	nn.backpropergation(10000)

	nn.out_test()

	#nn.debug()
