You are an EEG emotion recognion professinal resarcher.

Help user to create a new model for EEG emotion recognion.


Read ./PCL.md and get the related infos.

**You don't need to follow current structure of SABER model. You can completely change it. Instead refer to the `Basic model` in ./PCL.md as a skeleton of concept.**

Now you don't design the feature extractor. You can assume it is already done.

### What you should Design:

1.The classifier. Should it keep a simple linear layer or something like PRPL.

2.How to design a mechanism to generate the reliable pesudo label for target feature.

3.Since we have the pesudo label for target feature. How to construct the source prototype and target prototype according to pesudo label? Use some more innovative way. You may not follow the strcuture of models in ./PCL.md, instead use your own knowledge(may comes from other fields) to design a better one.

4.After get the source prototype and target prototype, How to use them to make to adapt targt domain better.

