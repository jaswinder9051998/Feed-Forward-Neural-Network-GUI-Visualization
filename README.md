# Feed_Forward_GUI_Visualization
This is a gui based feed-forward network contour plot visualization project
## COMPONENTS
## gui (inputs)
![Alt Text](https://media.giphy.com/media/Q8aRnP0omi4AYMgUCa/giphy.gif)
  * number of data points
  * number of epochs
  * hidden layers size in space seperated format eg. 2 3 2 -> first hidden layer-2, second-3, third-2
  * learning rate
## Feed forward neural network implemented from scratch
![Alt Text](https://media.giphy.com/media/d5enXmCkE909163Qgc/giphy.gif)
  * learning rate graph
  * contour plot generator
## pygame enviroment 
To show the user defined neural network and to register the neuron selected by user

# OUTLINE
After taking the user input, a dataset is created using sklearn's "make_blob", and the neural net is trained over it using the user specified parameters.
The pygame enviroment,is used to represent the neural net. Selection of a neuron is registered when the user clicks on it, and the corresponding contour plot is shown
