image = dict(
	height = 60,
	width = 80,
	colorspace = 'grayscale',
	channels = 1,
	format = 'JPEG',
	noise_scale = 1,
)
dataset = dict(
	size = 1000,
	train_ratio = 0.8,
	test_ratio = 0.2,
	num_categories = 2,
	label0_folder = 'incorrect',
	label1_folder = 'correct',
)
directory = dict(
	training = 'training',
	testing = 'testing',
	output = 'output',
)
shards = dict(
	training = 1,
	testing = 1,
)
threads = dict(
	preprocess = 1,
)

tfRecords = dict(
	prefix = 'try',
	training_file = 'train-00000-of-00001',
	testing_file = 'validation-00000-of-00001',
)

hyperparameters = dict(
	learning_rate = 0.01,
	#loss_function = '',
	batch_size = 100,
	num_epochs = 10,
	num_epochs_eval = 1,
)

# specify the neural network and the loss function for training
# available models can be found in models/
# available loss functions can be found in loss_functions/
model = dict(
	model_import = 'flat_h1_h2',
	loss_function_import = 'cross_entropy',
)