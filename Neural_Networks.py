### Packages ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

### Welcome message ###

print(' ')
print('--- Digit Clasifier Neural Network ---')
print(' ')

### Generic Functions ###

def init_params():
	W1 = np.random.rand(16, 784) - 0.5
	B1 = np.random.rand(16, 1) - 0.5
	W2 = np.random.rand(16, 16) - 0.5
	B2 = np.random.rand(16, 1) - 0.5
	W3 = np.random.rand(10, 16) - 0.5
	B3 = np.random.rand(10, 1) - 0.5
	return W1, B1, W2, B2, W3, B3
	
def softmax(Z):
	A = np.exp(Z) / sum(np.exp(Z))
	return A

def cost_exp(Y):
	cost_exp_Y = np.zeros((Y.size, Y.max() + 1))
	cost_exp_Y[np.arange(Y.size), Y] = 1
	cost_exp_Y = cost_exp_Y.T
	return cost_exp_Y

def update_params(W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, alpha):
	W1 = W1 - alpha * dW1
	B1 = B1 - alpha * dB1
	W2 = W2 - alpha * dW2
	B2 = B2 - alpha * dB2
	W3 = W3 - alpha * dW3
	B3 = B3 - alpha * dB3
	return W1, B1, W2, B2, W3, B3

def get_predictions(A3):
	return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
	return np.sum(predictions == Y) / Y.size

### ReLU functions ###

def ReLU(Z):
	return np.maximum(Z, 0)

def ReLU_deriv(Z):
	return Z > 0

def forward_prop_ReLU(W1, B1, W2, B2, W3, B3, X):
	Z1 = W1.dot(X) + B1
	A1 = ReLU(Z1)
	Z2 = W2.dot(A1) + B2
	A2 = ReLU(Z2)
	Z3 = W3.dot(A2) + B3
	A3 = softmax(Z3)
	return Z1, A1, Z2, A2, Z3, A3

def backward_prop_ReLU(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m):
	cost_exp_Y = cost_exp(Y)

	dZ3 = A3 - cost_exp_Y
	dW3 = 1 / m * dZ3.dot(A2.T)
	dB3 = 1 / m * np.sum(dZ3)

	dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
	dW2 = 1 / m * dZ2.dot(A1.T)
	dB2 = 1 / m * np.sum(dZ2)

	dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
	dW1 = 1 / m * dZ1.dot(X.T)
	dB1 = 1 / m * np.sum(dZ1)
	return dW1, dB1, dW2, dB2, dW3, dB3

def grad_dec_ReLU(X, Y, alpha, iterations, m):
	W1, B1, W2, B2, W3, B3 = init_params()
	accuracies = []
	for i in range(iterations):
		Z1, A1, Z2, A2, Z3, A3 = forward_prop_ReLU(W1, B1, W2, B2, W3, B3, X)
		dW1, dB1, dW2, dB2, dW3, dB3 = backward_prop_ReLU(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m)
		W1, B1, W2, B2, W3, B3 = update_params(W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, alpha)
		if i % 10 == 0:
			predictions = get_predictions(A3)
			accuracy = get_accuracy(predictions, Y)
			accuracies.append(accuracy)
			if i % 100 == 0:
				print(f'(ReLU) Itteration: {i}, Accuracy: {get_accuracy(predictions, Y):.4f}')
	return W1, B1, W2, B2, W3, B3, accuracies

def make_predictions_ReLU(X, W1, B1, W2, B2, W3, B3):
	_, _, _, _, _, A3 = forward_prop_ReLU(W1, B1, W2, B2, W3, B3, X)
	predictions = get_predictions(A3)
	return predictions

def test_prediction_ReLU(index, W1, B1, W2, B2, W3, B3, test_imgs, test_labs):
	current_img = test_imgs[:, index, None]
	prediction = make_predictions_ReLU(test_imgs[:, index, None], W1, B1, W2, B2, W3, B3)
	prediction = int(prediction)
	label = test_labs[index]

	current_img = current_img.reshape((28, 28)) * 255
	plt.gray()
	plt.imshow(current_img, interpolation = 'nearest')
	plt.title(f'(ReLU) Label = {label}, Prediction = {prediction}')
	plt.axis('off')
	plt.show()

### Tanh functions ###

def tanh(Z):
	return np.tanh(Z)

def tanh_deriv(Z):
	return 1 - (np.tanh(Z))**2

def forward_prop_tanh(W1, B1, W2, B2, W3, B3, X):
	Z1 = W1.dot(X) + B1
	A1 = tanh(Z1)
	Z2 = W2.dot(A1) + B2
	A2 = tanh(Z2)
	Z3 = W3.dot(A2) + B3
	A3 = softmax(Z3)
	return Z1, A1, Z2, A2, Z3, A3

def backward_prop_tanh(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m):
	cost_exp_Y = cost_exp(Y)

	dZ3 = A3 - cost_exp_Y
	dW3 = 1 / m * dZ3.dot(A2.T)
	dB3 = 1 / m * np.sum(dZ3)

	dZ2 = W3.T.dot(dZ3) * tanh_deriv(Z2)
	dW2 = 1 / m * dZ2.dot(A1.T)
	dB2 = 1 / m * np.sum(dZ2)

	dZ1 = W2.T.dot(dZ2) * tanh_deriv(Z1)
	dW1 = 1 / m * dZ1.dot(X.T)
	dB1 = 1 / m * np.sum(dZ1)
	return dW1, dB1, dW2, dB2, dW3, dB3

def grad_dec_tanh(X, Y, alpha, iterations, m):
	W1, B1, W2, B2, W3, B3 = init_params()
	accuracies = []
	for i in range(iterations):
		Z1, A1, Z2, A2, Z3, A3 = forward_prop_tanh(W1, B1, W2, B2, W3, B3, X)
		dW1, dB1, dW2, dB2, dW3, dB3 = backward_prop_tanh(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m)
		W1, B1, W2, B2, W3, B3 = update_params(W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, alpha)
		if i % 10 == 0:
			predictions = get_predictions(A3)
			accuracy = get_accuracy(predictions, Y)
			accuracies.append(accuracy)
			if i % 100 == 0:
				print(f'(tanh) Iteration: {i}, Accuracy: {get_accuracy(predictions, Y):.4f}')
	return W1, B1, W2, B2, W3, B3, accuracies

def make_predictions_tanh(X, W1, B1, W2, B2, W3, B3):
	_, _, _, _, _, A3 = forward_prop_tanh(W1, B1, W2, B2, W3, B3, X)
	predictions = get_predictions(A3)
	return predictions

def test_prediction_tanh(index, W1, B1, W2, B2, W3, B3, test_imgs, test_labs):
	current_img = test_imgs[:, index, None]
	prediction = make_predictions_tanh(test_imgs[:, index, None], W1, B1, W2, B2, W3, B3)
	prediction = int(prediction)
	label = test_labs[index]

	current_img = current_img.reshape((28, 28)) * 255
	plt.gray()
	plt.imshow(current_img, interpolation = 'nearest')
	plt.title(f'(tanh) Label = {label}, Prediction = {prediction}')
	plt.axis('off')
	plt.show() 

### Sigmoid functions ###

def sigmoid(Z):
	return np.exp(Z) / (1 + np.exp(Z))

def sigmoid_deriv(Z):
	return np.exp(Z) / (1 + np.exp(Z))**2

def forward_prop_sigmoid(W1, B1, W2, B2, W3, B3, X):
	Z1 = W1.dot(X) + B1
	A1 = sigmoid(Z1)
	Z2 = W2.dot(A1) + B2 
	A2 = sigmoid(Z2)
	Z3 = W3.dot(A2) + B3
	A3 = softmax(Z3)
	return Z1, A1, Z2, A2, Z3, A3 

def backward_prop_sigmoid(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m):
	cost_exp_Y = cost_exp(Y)

	dZ3 = A3 - cost_exp_Y
	dW3 = 1 / m * dZ3.dot(A2.T)
	dB3 = 1 / m * np.sum(dZ3)

	dZ2 = W3.T.dot(dZ3) * sigmoid_deriv(Z2)
	dW2 = 1 / m * dZ2.dot(A1.T)
	dB2 = 1 / m * np.sum(dZ2)

	dZ1 = W2.T.dot(dZ2) * sigmoid_deriv(Z1)
	dW1 = 1 / m * dZ1.dot(X.T)
	dB1 = 1 / m * np.sum(dZ1)
	return dW1, dB1, dW2, dB2, dW3, dB3

def grad_dec_sigmoid(X, Y, alpha, iterations, m):
	W1, B1, W2, B2, W3, B3 = init_params()
	accuracies = []
	for i in range(iterations):
		Z1, A1, Z2, A2, Z3, A3 = forward_prop_sigmoid(W1, B1, W2, B2, W3, B3, X)
		dW1, dB1, dW2, dB2, dW3, dB3 = backward_prop_sigmoid(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, m)
		W1, B1, W2, B2, W3, B3 = update_params(W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, alpha)
		if i % 10 == 0:
			predictions = get_predictions(A3)
			accuracy = get_accuracy(predictions, Y)
			accuracies.append(accuracy)
			if i % 100 == 0:
				print(f'(sigmoid) Iteration: {i}, Accuracy: {get_accuracy(predictions, Y):.4f}')
	return W1, B1, W2, B2, W3, B3 , accuracies

def make_predictions_sigmoid(X, W1, B1, W2, B2, W3, B3):
	_, _, _, _, _, A3 = forward_prop_sigmoid(W1, B1, W2, B2, W3, B3, X)
	predictions = get_predictions(A3)
	return predictions

def test_prediction_sigmoid(index, W1, B1, W2, B2, W3, B3, test_imgs, test_labs):
	current_img = test_imgs[:, index, None]
	prediction = make_predictions_sigmoid(test_imgs[:, index, None], W1, B1, W2, B2, W3, B3)
	prediction = int(prediction)
	label = test_labs[index]

	current_img = current_img.reshape((28, 28)) * 255
	plt.gray()
	plt.imshow(current_img, interpolation = 'nearest')
	plt.title(f'(sigmoid) Label = {label}, Prediction = {prediction}')
	plt.axis('off')
	plt.show()

### Aditional functions ###

def plot_accuracies(accuracies_dict):
	plt.figure(figsize = (10, 8))
	for model_name, accuracies in accuracies_dict.items():
		if accuracies:
			plt.plot(accuracies, label = model_name)
	plt.xlabel('Itterations (x10)')
	plt.ylabel('Accuracy (x100%)')
	plt.title('Accuracy of the Network During the Training Process')
	plt.legend()
	plt.grid(True)
	plt.show()

def visualise_weights(W1):
	num_n = W1.shape[0]
	plt. figure(figsize = (10, 10))
	for i in range(num_n):
		weight_img = W1[i].reshape(28, 28)
		plt.subplot(4, 4, i+1)
		plt.imshow(weight_img, cmap = 'gray')
		plt.title(f'Neuron {i+1}')
		plt.axis('off')
	plt.tight_layout()
	plt.show()

def test_predictions(activation, W1, B1, W2, B2, W3, B3, test_imgs, test_labs, n):
	predictions = []
	true_labels = []

	for index in range(min(n, len(test_imgs.T))):
		if activation == 'ReLU':
			prediction = int(make_predictions_ReLU(test_imgs[:, index, None], W1, B1, W2, B2, W3, B3))
		elif activation == 'tanh':
			prediction = int(make_predictions_tanh(test_imgs[:, index, None], W1, B1, W2, B2, W3, B3))
		elif activation == 'sigmoid':
			prediction = int(make_predictions_sigmoid(test_imgs[:, index, None], W1, B1, W2, B2, W3, B3))
		else:
			raise ValueError('Invalid activation function.')
	
		predictions.append(prediction)
		true_labels.append(int(test_labs[index]))

	return predictions, true_labels

def plot_confusion_matrix(cm, labels):
	plt.figure(figsize = (10, 8))

	disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
	disp.plot(cmap = 'Blues', xticks_rotation = 'vertical', ax = plt.gca(), include_values = True)

	plt.title('Confusion Matrix Visualisation')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	plt.tight_layout()
	plt.show()

def plot_misclass_diagram(true_labs, predictions):
	cm = confusion_matrix(true_labs, predictions)
	G = nx.DiGraph()

	for i in range(10):
		G.add_node(f'{i}_true', pos=(0, i))
		G.add_node(f'{i}_pred', pos=(1, i))

	for i in range(10):
		for j in range(10):
			if i != j and cm[i, j] > 0:
				G.add_edge(f'{i}_true', f'{j}_pred', weight=cm[i, j])

	pos = nx.get_node_attributes(G, 'pos')
	edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
	min_weight = min(edge_weights)
	max_weight = max(edge_weights)
	normalized_weights = [(weight - min_weight) / (max_weight - min_weight) * 5 + 1 for weight in edge_weights]

	plt.figure(figsize = (10, 8))
	nx.draw(G, pos, with_labels = True, labels = {node: node.split('_')[0] for node in G.nodes}, node_size = 2500, node_color = 'lightblue', font_size = 15, font_weight = 'bold', alpha = 1)
	nx.draw_networkx_edges(G, pos, width = normalized_weights, alpha = 1 , edge_color = 'black')
	plt.text(-0.075, 5, 'True labels', rotation = 90, fontsize = 16, ha = 'center', va = 'center', fontweight = 'bold')
	plt.text(1.075, 5, 'Predicted labels', rotation = 90, fontsize = 16, ha = 'center', va = 'center', fontweight = 'bold')
	plt.suptitle('Misclassification Diagram', fontsize=18)
	plt.tight_layout(pad = 3.0)
	plt.show()

### Main ###

def main():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	file_name = 'mnist_test.csv'
	file_path = os.path.join(script_dir, file_name)
	
	try:
		dataset = pd.read_csv(file_path)
	except FileNotFoundError:
		print('Training data "mnist_test.csv", was not found.')

	data = np.array(dataset)
	m, n = data.shape
	np.random.shuffle(data)

	train_size = 5000
	test_size = 5000

	train_data = data[0:5000].T
	train_labs = train_data[0]
	train_imgs = train_data[1:n]
	train_imgs = train_imgs / 255.

	test_data = data[5000:m].T
	test_labs = test_data[0]
	test_imgs = test_data[1:n]
	test_imgs = test_imgs / 255.
	_, m_test = test_imgs.shape

	accuracies_dict = {}

	while True:
		print('\nPlease select a feature to run.')
		print('Features:')
		print('[1] Train model.')
		print('[2] Plot the accuracies of the trained models.')
		print('[3] Plot the confusion matrix for the last trained model.')
		print('[4] Plot the misclassification diagram for the last trained model.')
		print('[5] Visualise the weights in the first layer on the network.')
		print('[6] Generate test predictions based of the last trained model.')
		print('[7] Clear models data.')
		print('[8] Shuffle dataset.')
		print('[9] Exit.')

		choice = input('Enter choice: ')
	
		if choice == '1':
			print('Please choose an activation function')
			print('Activation functions:')
			print('[1] ReLU')
			print('[2] tanh')
			print('[3] sigmoid')
			print('[4] Return to main menu.')

			act_fun = input('Enter choice: ')

			if act_fun in ['1', '2', '3']:
				act_fun_dict = {'1': 'ReLU', '2': 'tanh', '3': 'sigmoid'}
				act_fun = act_fun_dict[act_fun]

				while True:
					try:
						lr = float(input('Enter learning rate: '))
						while True:
							itr = int(input(f'Enter number of iterations (max {train_size}): '))
							if 1 <= itr <= int(train_size):
								break
							else:
								print('Invalid input. Please try again.')

						if act_fun == 'ReLU':
							W1, B1, W2, B2, W3, B3, accuracies = grad_dec_ReLU(train_imgs, train_labs, lr, itr, m)
						elif act_fun == 'tanh':
							W1, B1, W2, B2, W3, B3, accuracies = grad_dec_tanh(train_imgs, train_labs, lr, itr, m)
						elif act_fun == 'sigmoid':
							W1, B1, W2, B2, W3, B3, accuracies = grad_dec_sigmoid(train_imgs, train_labs, lr, itr, m)
						break
					except ValueError:
						print('Invalid input. Please try again.')
			
			elif act_fun == '4':
				continue	

			else:
				print('Invalid input. Please try again.')
				continue

			label = f'{act_fun}: alpha = {lr}'
			accuracies_dict[label] = accuracies
			fin_acc = accuracies[-1]
			print(' ')
			print(f'The final accuracy achived was: {fin_acc}')

		elif choice == '2':
			if any(accuracies for accuracies in accuracies_dict.values()):
				print('Ploting custom accuracies.')
				plot_accuracies(accuracies_dict)
			else:
				print('No models have been trained yet. Train a custom model first.')
		
		elif choice == '3':
			if accuracies_dict:
				while True:
					try:
						n = int(input(f'Enter a number of test examples to use (max {test_size}): '))
						if 1 <= n <= int(test_size):
							break
						else:
							print(f'Please enter a value between 1 and {test_size}.')
					except ValueError:
						print('Invalid input. Please try again.')
		
				predictions, true_labels = test_predictions(act_fun, W1, B1, W2, B2, W3, B3, test_imgs, test_labs, n)

				cm = confusion_matrix(true_labels, predictions)
				labels = [str(i) for i in range(10)]

				print('Plotting confusion matrix.')
				plot_confusion_matrix(cm, labels)

			else:
				print('No models have been trained yet. Train a model first.')

		elif choice == '4':
			if accuracies_dict:
				while True:
					try:
						n = int(input(f'Enter a number of test examples to use (max {test_size}): '))
						if 1 <= n <= int(test_size):
							break
						else:
							print(f'Please enter a value between 1 and {test_size}.')
					except ValueError:
						print('Invalid input. Please try again.')
				
				predictions, true_labels = test_predictions(act_fun, W1, B1, W2, B2, W3, B3, test_imgs, test_labs, n)

				cm = confusion_matrix(true_labels, predictions)
				
				print('Plotting misclasification diagram.')
				plot_misclass_diagram(true_labels, predictions)

			else:
				print('No models have been trained yet. Train a model first.')

		elif choice == '5':
			if accuracies_dict:
				print('Plotting weight visuals')
				visualise_weights(W1)
			else:
				print('No models have been trained yet. Train a model first.')

		elif choice == '6':
			if accuracies_dict:
				while True:
					try:
						noe = int(input(f'Enter the number of test examples to visualise (max {test_size}): '))
						if 1 <= noe <= int(test_size):
							break
						else:
							print(f'Invalid input. Please enter a value between 1 and {test_size}.')
					except ValueError:
						print('Invalid input. Please enter a numeric value.')

				if act_fun == 'ReLU':
					for i in range(noe):
						test_prediction_ReLU(i, W1, B1, W2, B2, W3, B3, test_imgs, test_labs)
				elif act_fun == 'tanh':
					for i in range(noe):
						test_prediction_tanh(i, W1, B1, W2, B2, W3, B3, test_imgs, test_labs)
				elif act_fun == 'sigmoid':
					for i in range(noe):
						test_prediction_sigmoid(i, W1, B1, W2, B2, W3, B3, test_imgs, test_labs)

			else:
				print('No models have been trained yet. Train a model first.')

		elif choice == '7':
			confirm = input('Are you sure you want to clear data? (y/n) : ')
			if confirm == 'y':
				accuracies_dict = {}
				print('Data cleared.')
			elif confirm == 'n':
				print('Data has not been cleared.')
			else:
				print('Invalid input. Try again.')

		elif choice == '8':
			data = np.array(dataset)
			m, n = data.shape
			np.random.shuffle(data)

			train_size = 5000
			test_size = 5000
	
			train_data = data[0:5000].T
			train_labs = train_data[0]
			train_imgs = train_data[1:n]
			train_imgs = train_imgs / 255.

			test_data = data[5000:m].T
			test_labs = test_data[0]
			test_imgs = test_data[1:n]
			test_imgs = test_imgs / 255.
			_, m_test = test_imgs.shape

			print('Dataset shuffled.')

		elif choice == '9':
			print('Exiting program. Goodbye!')
			break
		else:
			print('Invalid input. Please try again.')


if __name__ == '__main__':
	main()



























