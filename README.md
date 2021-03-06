# Unique subfields of neuroscience exhibit more diverse language

_Titipat Achakulvisut, Daniel E. Acuna, Danielle S. Bassett, Konrad P. Kording_

> The fields within neuroscience differ in their heterogeneity and their variability of language usage.
Some fields exhibit more diverse language while others write similarly to their neighboring fields.
However, it is unclear how language diversity and similarity are intertwined.
In this work, we propose that language differentiation of a sub-field promotes its language variability,
a hallmark of exploration. We test this hypothesis using a dataset of recent abstracts from the Society for Neuroscience (SfN) conference.
We find that a field that is more linguistically distant from its nearest field indeed has more within-field variability,
even after controlling for field size and changes over time, suggesting more breadth of ideas.


## Descriptions

This is a repository for submitting manuscript _Unique subfields of neuroscience exhibit more diverse language_.
We analyze language usage of neuroscience abstracts in the largest neuroscience conference, Society for Neuroscience conference.
We found that a field that is more linguistically distant from its nearest fields
tends to use more diverse language. We interpret that these fields have more breadth of ideas.


The repository contains following folders

- `data` folder contains post-processed SfN dataset
- `code` folder contains code for calculating language variability (see more details in the folder)
- `manuscript` folder contains submission writing for PLOS ONE


You can read the full manuscript [here](https://github.com/titipata/language-variability-neuro/blob/master/manuscript/unique_subfields_achakulvisut.pdf)


## Data Availability

The data is license under [SfN agreement](https://www.sfn.org/) which can be used for
research purpose only. See `data` folder to see how to download the dataset.
The data contains roughly 60k shuffle SfN abstracts from year 2013 to 2017 with
corresponding subtopic/topic/theme annotated by SfN.



## Dependencies

- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/s)
- [seaborn](https://seaborn.pydata.org/index.html)
- [linear-regressions-mixture](https://github.com/victorkristof/linear-regressions-mixture)
