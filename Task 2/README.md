# Task II: Classical Graph Neural Network (GNN)
For Task II, you will use ParticleNet’s data for Quark/Gluon jet classification available [here](https://zenodo.org/records/3164691#.YigdGt9MHrB) with its corresponding description. 
- Choose 2 Graph-based architectures of your choice to classify jets as being quarks or gluons. Provide a description on what considerations you have taken to project this point-cloud dataset to a set of interconnected nodes and edges. 
- Discuss the resulting performance of the 2 chosen architectures. 

## Dataset Description
Two datasets of quark and gluon jets generated with Pythia 8, one with all kinematically realizable quark jets and one that excludes charm and bottom quark jets (at the level of the hard process). The one without c and b jets was originally used in [Energy Flow Networks: Deep Sets for Particle Jets](https://arxiv.org/abs/1810.05165). Generation parameters are listed below:
- Pythia 8.226 (without bc jets), Pythia 8.235 (with bc jets), $\sqrt{s} = 14 TeV$
- Quarks from WeakBosonAndParton:qg2gmZq, gluons from WeakBosonAndParton:qqbar2gmZg with the Z decaying to neutrinos
- FastJet 3.3.0, anti-ki jets with R=0.4
- $p_T^{\text{jet}} \in [500, 550]\ \text{GeV}, \quad |y^{\text{jet}}| < 1.7$


There are 20 files in each dataset, each in compressed NumPy format. Files including charm and bottom jets have 'withbc' in their filename. There are two arrays in each file
- X: (100000,M,4), exactly 50k quark and 50k gluon jets, randomly sorted, where M is the max multiplicity of the jets in that file (other jets have been padded with zero-particles), and the features of each particle are its pt, rapidity, azimuthal angle, and pdgid.
- y: (100000,), an array of labels for the jets where gluon is 0 and quark is 1.

The datasets can be downloaded and read into python automatically using the [EnergyFlow Python package](https://energyflow.network/docs/datasets/#quark-and-gluon-jets). We shall be using this package for the task rather than downloading the datasets from source.