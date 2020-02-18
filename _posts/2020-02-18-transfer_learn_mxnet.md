# Title
> summary


# Transfer Learning Using MXNet

{1 sentence about what is transfer learning}. This post provides you an easy to follow tutorial on how to “train a base neural net” on a dataset and use that pre-trained network to “transfer learn” on a different dataset using MXNet/Gluon framework. The high level steps in this tutorial are very similar for any kind of transfer learning problem - tabular, time series, language or computer vision. The major differences when working with different problems are “network architecture” and “transformations and pre-processesing steps”. My goal is to provide a skeletal using text data (movie and hotel reviews) as an example, that you can adapt for different tasks. I have primarily used MXNet, Catboost and Sklearn libraries for this post. {Here} is the link to the jyputer notebook in case you directly want to jump to the code and skip reading the explainations. 

I haven’t covered any theory about what is transfer learning, why and where is it useful. To learn more on the theory part, I recommend this post by *Sebastian Ruder* - Transfer Learning - Machine Learning's Next Frontier (https://ruder.io/transfer-learning/). Ruder has done his PhD in this topic. So his work is really detailed. Here is the link to his thesis (https://ruder.io/thesis/) for deep divers. 
