# lambda_calc
Computationally cheap method to calculate electron-phonon coupling from frozen phonon calculations in large systems using phonopy and Quantum ESPRESSO

1. Calculate the phonons of the system using [phonopy](https://phonopy.github.io/phonopy/)
2. Run frozen_phonon_maker.sh to generate frozen phonon supercells
3. Calculate the frozen phonon and unperturbed supercells using QE pw.x with the same input parameters apart from the atomic coordinates (same k-mesh etc.)
4. Calculate the density of states of the unperturbed supercell
5. Use the data-file-schema.xml output files from the QE calculations as inputs to lamda_calc_band_v0-2.py
6. Run proc_lamda.sh with a window corresponding to the maximum phonon energy determined by phonopy

Please cite our paper if using this code https://arxiv.org/abs/2405.02519
