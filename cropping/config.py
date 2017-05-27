image = dict(
	height = 64,
	width = 64,
	colorspace = 'grayscale',
	channels = 1,
	format = 'JPEG',
	noise_scale = 1,
)
dataset = dict(
	size = 100000,
	train_ratio = 0.8,
	test_ratio = 0.2,
	num_categories = 2,
	label0_folder = 'label0',
	label1_folder = 'label1',
)
directory = dict(
	training = '/home/nagy729krisztina/cropping/locarno_cropped/training',
	testing = '/home/nagy729krisztina/cropping/locarno_cropped/testing',
	output = '/home/nagy729krisztina/cropping/locarno_cropped/output',
    tfrecords = '/home/nagy729krisztina/cropping/locarno_cropped/output/' + '64/' + str(dataset['size']),
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
	batch_size = 10,
	num_epochs = 1,
	num_epochs_eval = 1,
)

# specify the neural network and the loss function for training
# available models can be found in models/
# available loss functions can be found in loss_functions/
model = dict(
	model_import = 'vgg19_trainable',
	loss_function_import = 'cross_entropy',
)