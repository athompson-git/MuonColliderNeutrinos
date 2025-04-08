# Neutrinos from a Muon Storage Ring / Muon Collider

## Neutrino flux codes and scattering models for [To appear]
[![DOI](https://zenodo.org/badge/920847917.svg)](https://zenodo.org/badge/latestdoi/920847917)

This repository contains python classes and functions to perform weighted monte carlo (MC) simulations and modeling of the neutrino flux from a circular muon storage ring or muon collider. In addition, it contains integration methods to calculate the rates of electron scattering in a detector material using the ```alplib``` library to facilitate some helper classes. 


Author: Adrian Thompson
Contact: ```a.thompson@northwestern.edu```

### Requirements:
* alplib: [https://github.com/athompson-git/alplib](https://github.com/athompson-git/alplib)
* LFS for large flux files
* numpy, scipy
* tqdm

### File Descriptions
```nu_crosssection.py``` contains
* ```NeutrinoNucleonCCQE``` : An implementation of the CCQE cross section for arbitrary lepton final state using the form reported in Formaggio, Zeller

```muon_collider_mc.py``` contains the monte carlo classes to simulate or model the flux and electron scattering of the neutrino beams sourced by the circular storage ring. In particular, it has

* ```NeutrinoFluxMuonRing``` class, with ```.simulate_fluxes_mc()``` and ```.simulate_flux()``` methods. The former performs geometric random sampling of the muon decay points along the ring and populates lists of 4-vectors with the neutrino momenta in the lab frame, while ```.simulate_flux()``` just populates the flux weights with a simple acceptance factor (see paper for details).

* ```EvESFromNeutrinoFlux``` computes the elastic neutrino electron scattering (E$\nu$ES) rates in a detector with a specified size and material, using the ```Material``` class defined in ```alplib``` (one of the dependencies).

```muon_decay.py``` has the functions for the differential decay distributions for the daughter neutrinos of muon decay. Namely, ```d2NdydOmega_numu(y, theta, Emu, l_det, N_mu)``` and ```d2NdydOmega_nue(y, theta, Emu, l_det, N_mu)``` are the laboratory frame differential decay rates for the muon neutrino and electron anti-neutrino, respectively. 