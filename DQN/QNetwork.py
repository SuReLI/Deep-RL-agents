
import tensorflow as tf

from NetworkArchitecture import NetworkArchitecture


class QNetwork:

	def __init__(self, state_size, action_size):

		with tf.variable_scope("QNetwork"):
	        self.state_size = state_size
    	    self.action_size = action_size

        	# Define the model
            self.model = NetworkArchitecture(self.state_size)

            # Convolution network - or not
            self.inputs = self.model.build_layers()

            # LSTM Network - or not
            if parameters.LSTM:
                # Input placeholder
                self.state_in = self.model.build_lstm()

                self.lstm_state_init = self.model.lstm_state_init
                self.state_out, model_output = self.model.return_output(True)

            else:
                model_output = self.model.return_output(False)
