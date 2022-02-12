### MNIST-SHADOW MODEL
```
ARCHITECTURE - LENET
Layers:
    - 2 CONV
    - 2 AVERAGE POOLING
    - 2 FULL CONVERTED 
    - 1 OUTPUT

Activation Function:
    - tanh (others)
    - softmax (output)



```


# ** MI: membership inference **

Library used:
```
1. keras
2. tensorflow
3. scikit-learn
4. pandas
5. matplotplib

$ conda install library_name

### How to run
$ python shadow-model.py -d dataset-name

```

## Run confidence based attack
```
$ python conf_based.py -d mnist -s none -a NN -m shadow_models/mnist/mnist.h5
```


## Task:

- [x] Training shadow model
- [x] Develop attacker model - DOING
- [x] Test attacker model
- [] Use genetic algorithm model to test attacker model

