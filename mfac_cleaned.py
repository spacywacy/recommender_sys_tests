import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules import Module
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from time import time
import matplotlib.pyplot as plt
import pickle


def load_data(fname, size=-1):
	ml_100k = []
	with open(fname) as f:
		read_data = f.read()
		ml_100k_tmp = read_data.split('\n')

	for item in ml_100k_tmp:
		tmp_record = item.split('	')
		if len(tmp_record) > 1:
			ml_100k.append([int(x) for x in item.split('	')])

	if size == -1:
		return ml_100k
	else:
		return ml_100k[:size]

def split_static(in_data, train_rate, dev_rate):
	data_len = len(in_data)
	train_end = int(data_len * train_rate)
	dev_end = train_end + int(data_len * dev_rate)

	train = in_data[:train_end]
	dev = in_data[train_end:dev_end]
	test = in_data[dev_end:]

	return train, dev, test

def split_columns(in_table):
	users = torch.LongTensor([int(x[0]) for x in in_table])
	items = torch.LongTensor([int(x[1]) for x in in_table])
	return users, items

class ScaledEmbedding(nn.Embedding):

	def reset_parameters(self):
		self.weight.data.normal_(0, 1.0 / self.embedding_dim)
		if self.padding_idx is not None:
			self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):

	def reset_parameters(self):
		self.weight.data.zero_()
		if self.padding_idx is not None:
			self.weight.data[self.padding_idx].fill_(0)

class bilinear(Module):

	def __init__(self, num_user, num_item, emb_dim, sparse=False):
		super().__init__()
		self.user_embeddings = ScaledEmbedding(num_user, emb_dim, sparse=sparse)
		self.item_embeddings = ScaledEmbedding(num_item, emb_dim, sparse=sparse)
		self.user_biases = ZeroEmbedding(num_user, 1, sparse=sparse)

		self.item_biases = ZeroEmbedding(num_item, 1, sparse=sparse)
		self.emb_dim = emb_dim

	def forward(self, user_ids, item_ids):
		user_emb = self.user_embeddings(user_ids)
		item_emb = self.item_embeddings(item_ids)

		user_emb = user_emb.view(-1, self.emb_dim)
		item_emb = item_emb.view(-1, self.emb_dim)

		user_bias = self.user_biases(user_ids)
		item_bias = self.item_biases(item_ids)

		user_bias = user_bias.view(-1)
		item_bias = item_bias.view(-1)

		dot = (user_emb * item_emb).sum(1)

		return F.sigmoid(dot + user_bias + item_bias)

def construct_dict(all_data, if_str=True):
	user_dict = {}
	for row in all_data:
		if if_str:
			user_id = str(row[0])
			item_id = str(row[1])
		else:
			user_id = row[0]
			item_id = row[1]
		if user_dict.get(user_id) == None:
			user_dict[user_id] = [item_id]
		else:
			user_dict[user_id].append(item_id)

	return user_dict

def sampling_negatives(in_user_id, sample_size, exist_dict, all_items):

	if sample_size <= len(all_items):
		sample_i = torch.LongTensor(np.random.random_integers(0, len(all_items)-1, sample_size))
		sample_ = all_items[sample_i]
	else:
		sample_ = all_items

	existed = exist_dict[str(in_user_id)]

	for item in sample_:
		if item in existed:
			sample_.remove(item)

			back_up_i = torch.LongTensor(np.random.random_integers(0, len(all_items)-1, 1))
			back_up = all_items[back_up_i][0]
			while back_up in existed:
				back_up_i = torch.LongTensor(np.random.random_integers(0, len(all_items)-1, 1))
				back_up = all_items[back_up_i][0]
			sample_.append(back_up)

		else:
			continue

	return torch.LongTensor([int(x) for x in sample_])

def batch_samp_negatives(user_batch, sample_size, exist_dict,  all_items):
	batch_samp = torch.LongTensor()
	for user in user_batch:
		new_sample = sampling_negatives(user, sample_size, exist_dict, all_items)
		batch_samp = torch.cat([batch_samp, new_sample])

	return batch_samp

def pointwise_loss(net, users, items, exist_dict, all_items, sample_size):

	#negatives_raw = sampling_negatives(str(users.data[0]), sample_size, exist_dict, all_items)
	negatives_raw = batch_samp_negatives(users.data, sample_size, exist_dict, all_items)
	negatives = Variable(negatives_raw)

	positive_loss = (1.0 - net(users, items))
	negative_loss = net(users, negatives)

	return torch.cat([positive_loss, negative_loss]).mean()

def positive_loss(net, users, items):

	return (1.0 - net(users, items)).mean()

def load_batch_static(users, items, batch_size):
	i = 0
	while i+batch_size <= len(users):
		yield users[i:i+batch_size], items[i:i+batch_size]
		i+=batch_size

	if len(list(users)[i:]) > 0:
		yield users[i:], items[i:]

def plot_loss(x, y):
	print(len(x))
	print(len(y))
	plt.plot(x, y)
	plt.show()

def fit(loss_func,
		net,
		users,
		items,
		batch_size,
		epoches,
		optimizer,
		lr_scheduler,
		exist_dict,
		sample_size,
		load_func=load_batch_static,
		if_plot=True):

	epoch_losses = []

	for epoch in range(epoches):
		epoch_loss = 0.0
		n = 0

		for user_batch, item_batch in load_func(users, items, batch_size):
			user_var = Variable(torch.LongTensor(user_batch))
			item_var = Variable(torch.LongTensor(item_batch))
			optimizer.zero_grad()
			loss = loss_func(net, user_var, item_var, exist_dict, items, sample_size)
			#loss = loss_func_(net, user_var, item_var)
			epoch_loss += loss.data[0]
			loss.backward()
			optimizer.step()
			n+=1

		lr_scheduler.step()
		print('finished epoch:', epoch+1)
		epoch_loss = epoch_loss / n
		print('epoch loss:', epoch_loss)
		epoch_losses.append(epoch_loss)

	print('finished training')
	#plot_loss(list(range(epoches)), epoch_losses)


def predict(net, users, items):
	return net(users, items)

def prediction_accuracy(net, users, items):
	pass


def write_file(fname):
	with open(fname, 'w') as f:
		pass

def pickle_dump(fname, in_obj):
	with open(fname, 'wb') as f:
		pickle.dump(in_obj, f)

def pickle_load(fname):
	with open(fname, 'rb') as f:
		obj = pickle.load(f)
		
	return obj


fname = 'ml-100k/u.data'
data = load_data(fname, size=-1)
train_set, dev_set, test_set = split_static(data, .9, .01)
train_users, train_items = split_columns(train_set)


loss_func_ = pointwise_loss
#loss_func_ = positive_loss
net_ = bilinear(1000, 1700, 5)
batch_size_ = 128
epoches_ = 500
wd_ = .1
optimizer_ = optim.SGD(params=net_.parameters(), lr=.1, momentum=.9, weight_decay=1e-5)
lr_sche_ = optim.lr_scheduler.StepLR(optimizer_, step_size=150, gamma=.5)
exist_dict_ = construct_dict(train_set)
sample_size_ = 1


fit(loss_func_,
	net_,
	train_users,
	train_items,
	batch_size_,
	epoches_,
	optimizer_,
	lr_sche_,
	exist_dict_,
	sample_size_)


pickle_dump('try_weight_decay_1e-5_full_size.pickle', net_)


#net_ = pickle_load('try_weight_decay_0.pickle')


test_users, test_items = split_columns(test_set)
test_user_var = Variable(test_users)
test_item_var = Variable(test_items)
#predict_ = predict(net_, test_users, test_items)
#print(predict_)

#for item in predict_:
	#print(item.data[0])

train_user_var = Variable(train_users)
train_item_var = Variable(train_items)

print(positive_loss(net_, test_user_var, test_item_var))

exist_dict_test_ = construct_dict(test_set)
test_loss_ = pointwise_loss(net_, test_user_var, test_item_var, exist_dict_test_, test_items, 1)
print(test_loss_)

#for row_user, row_item in zip(test_users, test_items):
	#loss_row = pointwise_loss(net_, row_user, row_item, exist_dict_test_, test_items, 1)
	#print(loss_row)







































