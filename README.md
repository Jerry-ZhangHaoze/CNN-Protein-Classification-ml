# CNN-Protein-Classification-ml
Protein sequence classification with A domain specificity using Convolutional Neural Network (CNN) for genome mining
## Background 
Protein classification plays a critical role in genome mining and bioinformatics.
With time, many models are utilized to achieve a better or optimal results under the given conditions for the genome mining.This project applies convolutional Neural Networks (CNN), which is a mature model exceling at process the graphical information and the grid information, to classify protein sequences with a focus on A-domain protein fragments.

The goal is to explore how deep learning models can capture sequence patterns to make more accurate prediction and improve classification performance for biological data with the different inner factors optimization at different level of sizes.

## Data Source
This project uses two datasets:

1.**Trainingdata (Small-scale)**
- Provided dataset
- contains shorter/ more 'simple' and clear list
- Used for initial model construction, training, and validation

2.**Parasectdata (Large-scale)**
- Adapted from BTheDragonMaster data:
  https://github.com/BTheDragonMaster/parasect/tree/master/src/parasect/data
- Combined from the sources based on the characters:
  ID, Sequence, 8A signature, Stachelhaus code, specificity
- Adapted version included and utilized in the actual model training due to data loss during the process of combination and overcomplicated data which fluctuate the whole result significantly specifically for the multi-corresponding situation along with the down sizing to ease the process.

## Methods
### Sequence Encoding
Protein sequences are converted into numerical representations using one-hot encoding, 
where each amino acid is mapped into a 21-dimensional vector.

### Model Architecture
A Convolutional Neural Network (CNN) is used:
- Conv1D layer for detecting local sequence patterns
- Flatten layer for feature transformation
- Dense layer with softmax activation for classification

### Training
- Loss function: categorical cross-entropy
- Optimizers: comparison among SGD, RMSprop, and Adams Family
- Test/Train split ratio:Ratio of the data used to compare and learn and to test the results
- **Batch sizes & Epoch sizes correlation**
  To effective derive the experimental runs and results, a potential positive correlationship between the batch sizes and epoch sizes is observed and a combination of the factors is assigned to each-size data separately:
Training data: 128 batch sizes and 150 epoch sizes & Parasect data: 256 batch sizes and 250 epoch sizes
- **Batch sizes & Learning rate correlation**
    Under the same logic, the learning rate is preassigned with the findings that the results are optimized with higher learning rate for smaller dataset while the larger one needs slower rate, assign:
Training data: 0.01 and Parasect data with 0.001

## Experiments
Two experiments are conducted along with the changing on the optimizer and test/train split ratio

1.**Dataset 1 (Column 3 as input)**
- Uses shorter sequence length (10)
- Evaluates classification performance on fragment-level data

2.**Dataset 2 (Column 2 as input)**
- Uses longer sequence length (34)
- Evaluates model performance on extended protein sequences

## Results
- Generally, the CNN model provides stable classification and prediction performance as promised for both datasets while finding the correlationship of the batch sizes and epoch sizes along with the learning rates regarding to the different sizes.
- The characteristics of different optimizers are also learned especially for the adams and RMSProp with their ability and speciality on capturing the changes(momentum) or maintaining the stability
- Longer sequence input also improves the classification accuracy as they avoid certain level of promiscuity(multi-correspondance) and the gaps while the higher training ratio is dependent to provide a higher results at this stage.
( **Average 60-70% of test accuracy with reasonable test loss around a test ratio of 0.2** )
- Overall, the model successfully captures local structural patterns in protein sequences.

## Future work
- Incorporating larger and more diverse protein datasets, taking the multi-correspondance case into consideration, and increasing the test ratio while maintaing the high test accuracy with low loss
- Exploring other possible architectures model
- Apply to the advanced information and discovery to put into pipelines.
