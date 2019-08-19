**Summary of the algorithm**

- Start with an interlinear translation of Modern English mapped to Chaucer.
- Input sentences are cleaned of any punctuation, then turned into a list of word tokens.
- The "vocabulary" for the model is generated using a set of training sentence pairs. 
- An encoder LSTM turns input sequences to 2 state vectors which are discarded, only keeping the LSTM state
- A decoder LSTM is trained to turn the target sequences into the same sequence but offset by one timestep in the future.
- In inference mode, when we want to decode unknown input sequences:
    - Encode the input sequence into state vectors
    - target sequence is generated 
    - Feed the state vectors and 1-word target sequence to the decoder to predict the next word
    - Sample the next word using these predictions
    - Append the sampled word to the target sequence
    - Repeat until we generate the end-of-sequence word or we hit the maximum word length for a sentence

**References**

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Learning Phrase Representations usingRNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- https://chaucer.fas.harvard.edu/pages/text-and-translations
