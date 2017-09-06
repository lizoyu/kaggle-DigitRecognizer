from keras.models import load_model
from keras.backend import tf as ktf
from lib.data_utils import create_submission

#models = ['tuneResNet_03-0.02.h5', 'tuneResNet_04-0.02.h5', 'tuneResNet_05-0.01.h5', 'tuneResNet_06-0.01.h5']
models = ['tuneResNet_early_08-0.0068.h5']
for m in models:
	tunemodel = load_model('../models/'+m, custom_objects={'ktf': ktf})
	print('Load model successfully: ', m)
	create_submission(tunemodel, '../data/test.csv', '../submission/submission_'+m+'.csv', 16, fit=True)