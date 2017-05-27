image = dict(
	height = 64,
	width = 64,
	colorspace = 'grayscale',
	channels = 1,
	format = 'JPEG',
	noise_scale = 1,
)

image_str = dict(
    str = str(image['height']) + '_' + str(image['width']) + '_' + str(image['channels']) + '/'
)

dataset = dict(
	size = 100,
	train_ratio = 0.8,
	test_ratio = 0.2,
	num_categories = 2,
	label0_folder = 'label0',
	label1_folder = 'label1',
)

# specify the neural network and the loss function for training
# available models can be found in models/
# available loss functions can be found in loss_functions/
model = dict(
	model_import = 'vgg_64_gray',
	loss_function_import = 'cross_entropy',
)

directory = dict(
	training = '/home/nagy729krisztina/cropping/locarno_cropped/training_images',
	testing = '/home/nagy729krisztina/cropping/locarno_cropped/testing_images',
	output = '/home/nagy729krisztina/M4_crops/results/' + model['model_import'],
    tfrecords = '/home/nagy729krisztina/cropping/locarno_cropped/tfrecords/' + image_str['str'] + str(dataset['size']),
    tensorboard = 'tensorboard_logs',
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
	num_epochs = 10,
	num_epochs_eval = 1,
)

