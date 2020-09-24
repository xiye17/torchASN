# torchASN
A pytorch implementation of ["Abstract Syntax Networks for Code Generation and Semantic Parsing"](https://arxiv.org/pdf/1704.07535.pdf).

 
 ## Prerequisites
 
 * pytorch >= 1.4.0
 
 ## Run on Demo Dataset
 
 Instructions of running  `torchASN` on [DeepRegex](https://arxiv.org/pdf/1608.03000.pdf), an NL-to-Regex dataset, are included. This implementation achieves 61.6% DFA-accuracy on *DeepRegex*. For comparison, the performance of a seq-to-seq model with attention is 58.2%.
 
To run the code,

```
# Preprocess dataset
python dataset/turk/make_dataset.py

# Train a model. A pretrained model is included at checkpoints/turk/pretrained.bin.
./scripts/turk/train.sh

# Test a model. <model_file> is the pointer to the model, e.g., the pretrained one mentioned above.
./scripts/turk/test.sh <model_file>
```

## Addapting to New Programming Language

Adapting to a new programming langauge requires following steps:

1. Write down the domain specific language in the Abstract Syntax Description Language (ASDL) form. (refer to `data/turk/turk_asdl.txt`)

2. Implement the `TransitionSystem` for the new DSL to (1) convert the programs between logical forms and ASTs (2) check if a partial program (incomplete AST) is correct during beam search decoding. Refer to `grammar/turk/turk_trainsition_system.py`.

3. Modify the evaluation code accordingly.

## Credit
Part of the codes and system design are modified from [TranX](https://github.com/pcyin/tranX).
