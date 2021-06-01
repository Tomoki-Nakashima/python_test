import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add( Dense( units = 16, input_shape = (2, ) ) )
model.add( Dense( units = 5 ) )
model.add( Dense( units = 1 ) )
model.compile( loss = 'mean_squard_error', optimizer = 'sgd' )

import numpy

x_train = numpy.array( [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ] )
y_train = numpy.array( [ 0, 1, 1, 1 ] )

history = model.fit( x_train, y_train, epochs = 15 )

