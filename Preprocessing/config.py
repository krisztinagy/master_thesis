image = dict(
	height = 60,
	width = 80,
	colorspace = 'grayscale',
	channels = 1,
	format = 'JPEG',
	noise_scale = 1,
)
dataset = dict(
	training_set_size = 200,
	testing_set_size = 100,
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
	loss_function = '',
	batch_size = 100,
	num_epochs = 10,
	num_epochs_eval = 1,
)

model = dict(
	model_import = 'cnn_c1_c2_h1',
	loss_function_import = 'cross_entropy',
)