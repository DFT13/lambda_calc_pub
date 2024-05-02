#!/usr/bin/env python
# coding: utf-8
import xml.etree.ElementTree as ET
import xmlschema
import numpy as np
import matplotlib.pyplot as plt
import sys, argparse,os

#Hartree in eV
HartreeineV = 27.211396132 #eV
HartreeinKelvin = 315775.326864 #K
def eVtoHartree(E):
    return E/HartreeineV
def HartreetoeV(E):
  return E*HartreeineV

def calc_gk(dE,dx,M0,wph):
  g = (dE/dx)*np.sqrt(1/(2*M0*wph))
  return g

def calc_lamda(g,N,wph):
  lamda = (2*g*g)/(wph*N)
  return lamda

def calc_pi(gkqmn,fnk,fmkq,Enk,Emkq,wph,nu):
  pi = gkqmn**2 * (fnk - fmkq) * (nu/((Enk-Emkq)**2 + nu**2))
  return pi

def lorentzian(gamma,x,x0):
  return (1/np.pi)*((.5*gamma)/((x-x0)**2+(0.5*gamma)**2))

def gaussian(sigma,x,x0):
  return (1.0/(sigma*np.sqrt(2.0*np.pi)))*(np.exp(-0.5*(x-x0)**2/(sigma**2)))
#Allen-Dynes
def Tc(wlog,lamda,muc):
  Tc = (wlog/1.2) * np.exp((-1.04 * (1.0 + lamda))/(lamda - (muc * (1 + (0.62 * lamda)))))
  return Tc

parser=argparse.ArgumentParser()

parser.add_argument("--eqfile", help="xml file with the equilibrium cell data")
parser.add_argument("--phonfile", help="xml file with the phonon cell data")
parser.add_argument("--NEf", help="Density of states at Fermi energy States/eV")
parser.add_argument("--window", help="Window near Fermi energy in meV")

args=parser.parse_args()
#Quantum ESPRESSO qes schema needed to read the .xml files, may need to find the schema compatible with your QE version
name_schema = os.path.dirname(os.path.realpath(__file__)) + "/qes_current_develop.xsd"
schema = xmlschema.XMLSchema(name_schema)

datasc = ET.parse(args.eqfile).getroot()
data_dict_sc = schema.to_dict(datasc)

dataphon = ET.parse(args.phonfile).getroot()
data_dict_phon = schema.to_dict(dataphon)

#Number of bands defined
nbndsc = (data_dict_sc["output"]["band_structure"]["nbnd"])
nbndp = (data_dict_phon["output"]["band_structure"]["nbnd"])
print("nbnd = ",nbndsc)
print("nbndp = ",nbndp)
#Extracting Monkhorst-Pack grid numbers
nk1sc = data_dict_sc["output"]["band_structure"]["starting_k_points"]["monkhorst_pack"]['@nk1'] #nk1sc was nk1
nk2sc = data_dict_sc["output"]["band_structure"]["starting_k_points"]["monkhorst_pack"]['@nk2'] 
nk3sc = data_dict_sc["output"]["band_structure"]["starting_k_points"]["monkhorst_pack"]['@nk3'] 
nk1phon = data_dict_phon["output"]["band_structure"]["starting_k_points"]["monkhorst_pack"]['@nk1'] #nk1phon was nk1sc
nk2phon = data_dict_phon["output"]["band_structure"]["starting_k_points"]["monkhorst_pack"]['@nk2']
nk3phon = data_dict_phon["output"]["band_structure"]["starting_k_points"]["monkhorst_pack"]['@nk3']
print("nk1 =",nk1sc, ", nk2 =",nk2sc, ", nk3 =",nk3sc)
print("nk1 =",nk1phon, ", nk2 =",nk2phon, ", nk3 =",nk3phon)
#Extracting total number of k-points
nkssc = data_dict_sc["output"]["band_structure"]["nks"]
nksp = data_dict_phon["output"]["band_structure"]["nks"]
print("nkssc = ",nkssc)
print("nksp = ",nksp)
#Fermi energy
Efermisc = data_dict_sc["output"]["band_structure"]["fermi_energy"]
Efermiphon = data_dict_phon["output"]["band_structure"]["fermi_energy"]
print("fermi_energy = ",Efermisc)
print("fermi_energy phonon = ",Efermiphon)
#Getting the reciprocal lattice vectors
b_matsc = np.array([data_dict_sc["output"]["basis_set"]["reciprocal_lattice"]["b1"],                  data_dict_sc["output"]["basis_set"]["reciprocal_lattice"]["b2"],                  data_dict_sc["output"]["basis_set"]["reciprocal_lattice"]["b3"]])
print("Reciprocal Lattice Vectors primitive: \n", b_matsc)

b_matphon = np.array([data_dict_phon["output"]["basis_set"]["reciprocal_lattice"]["b1"],                    data_dict_phon["output"]["basis_set"]["reciprocal_lattice"]["b2"],                    data_dict_phon["output"]["basis_set"]["reciprocal_lattice"]["b3"]])
print("Reciprocal Lattice Vectors supercell: \n", b_matphon)

cellvec = np.array([data_dict_sc["output"]["atomic_structure"]["cell"]["a1"],                      data_dict_sc["output"]["atomic_structure"]["cell"]["a2"],                     data_dict_sc["output"]["atomic_structure"]["cell"]["a3"]])
print("cell vectors: \n",cellvec)

alatsc= data_dict_sc["output"]["atomic_structure"]['@alat']
print("alatsc = ", alatsc)
alatphon= data_dict_phon["output"]["atomic_structure"]['@alat']
print("alatphon = ", alatphon)

#create kmesh_sc of k_points, eigenvalues and occupations
k_point_arr_sc = np.empty((nkssc,3))
eigenvalue_arr_sc = np.empty((nkssc,nbndsc))
occ_arr_sc = np.empty((nkssc,nbndsc))
i=0
for kpoint in data_dict_sc["output"]["band_structure"]["ks_energies"]:
    k_point_arr_sc[i] = np.array(kpoint['k_point']["$"])
    eigenvalue_arr_sc[i] = np.array(kpoint["eigenvalues"]["$"])
    occ_arr_sc[i] = np.array(kpoint["occupations"]["$"])
    i=i+1

for i in range(len(occ_arr_sc)):
    for j in range(len(occ_arr_sc[i])):
        if occ_arr_sc[i][j] > 1.0:
            occ_arr_sc[i][j] = 1.0
        if occ_arr_sc[i][j] < 0.0001:
            occ_arr_sc[i][j] = 0.0

#Create array of rotation matrices
nrotsc = data_dict_sc["output"]["symmetries"]["nrot"]
rot_matsc = np.zeros((nrotsc,3,3), dtype = np.float32)
i=0
for symmetry in data_dict_sc["output"]["symmetries"]["symmetry"]:
    rot_matsc[i][0][:] = symmetry["rotation"]["$"][0:3]
    rot_matsc[i][1][:] = symmetry["rotation"]["$"][3:6]
    rot_matsc[i][2][:] = symmetry["rotation"]["$"][6:9]
    i=i+1
    
#Create 3D array that indexes each k-point for q-value calculations
kmesh_sc = np.zeros((nk1sc,nk2sc,nk3sc), dtype=np.int32)
kmesh_sc[:][:][:] = -1

for i in range(len(k_point_arr_sc)):
    j=j+1
    #Turns k_point_arr[i] in reciprocal cartesian basis to kinstance in reciprocal lattice relative basis
    kinstance = (np.linalg.inv(b_matsc.T).dot(k_point_arr_sc[i]) )
    na = round(kinstance[0]*float(nk1sc))
    nb = round(kinstance[1]*float(nk2sc))
    nc = round(kinstance[2]*float(nk3sc))
    while na >= nk1sc:
      na = na-nk1sc
    while nb >= nk2sc:
      nb = nb-nk2sc
    while nc >= nk3sc:
      nc = nc-nk3sc
    while na <= -nk1sc:
      na = na+nk1sc
    while nb <= -nk2sc:
      nb = nb+nk2sc
    while nc <= -nk3sc:
      nc = nc+nk3sc

    if kmesh_sc[na][nb][nc] == -1:
        kmesh_sc[na][nb][nc] = i
#Rotation occurring on k point described in relative basis set vectors rather than cartesian
for i in range(len(k_point_arr_sc)):
    for mat in rot_matsc:
        j=j+1
        #Rotation matrix must act on relative recirpocal basis set vectors rather than cartesian recirpocal. Matrix acting to the right must also be mat.T
        kinstance = mat.T.dot(np.linalg.inv(b_matsc.T).dot(k_point_arr_sc[i]))
        na = round(kinstance[0]*float(nk1sc))
        nb = round(kinstance[1]*float(nk2sc))
        nc = round(kinstance[2]*float(nk3sc))          
        while na >= nk1sc:
          na = na-nk1sc
        while nb >= nk2sc:
          nb = nb-nk2sc
        while nc >= nk3sc:
          nc = nc-nk3sc
        while na <= -nk1sc:
          na = na+nk1sc
        while nb <= -nk2sc:
          nb = nb+nk2sc
        while nc <= -nk3sc:
          nc = nc+nk3sc
        if kmesh_sc[na][nb][nc] == -1:
            kmesh_sc[na][nb][nc] = i

#create kmesh_sc of k_points, eigenvalues and occupations
k_point_arr_phon = np.empty((nksp,3))
eigenvalue_arr_phon = np.empty((nksp,nbndp))
occ_arr_phon = np.empty((nksp,nbndp))
i=0
for kpoint in data_dict_phon["output"]["band_structure"]["ks_energies"]:
    k_point_arr_phon[i] = np.array(kpoint['k_point']["$"])
    eigenvalue_arr_phon[i] = np.array(kpoint["eigenvalues"]["$"])
    occ_arr_phon[i] = np.array(kpoint["occupations"]["$"])
    i=i+1   

for i in range(len(occ_arr_phon)):
    for j in range(len(occ_arr_phon[i])):
        if occ_arr_phon[i][j] > 1.0:
            occ_arr_phon[i][j] = 1.0
        if occ_arr_phon[i][j] < 0.0001:
            occ_arr_phon[i][j] = 0.0
            
#Create array of rotation matrices
nrot_phon = data_dict_phon["output"]["symmetries"]["nrot"]
rot_matphon = np.zeros((nrot_phon,3,3), dtype = np.float32)
i=0
for symmetry in data_dict_phon["output"]["symmetries"]["symmetry"]:
    rot_matphon[i][0][:] = symmetry["rotation"]["$"][0:3]
    rot_matphon[i][1][:] = symmetry["rotation"]["$"][3:6]
    rot_matphon[i][2][:] = symmetry["rotation"]["$"][6:9]
    i=i+1
    
#Create 3D array that indexes each k-point for q-value calculations
kmesh_phon = np.zeros((nk1sc,nk2sc,nk3sc), dtype=np.int32)
kmesh_phon[:][:][:] = -1

for i in range(len(k_point_arr_phon)):
    j=j+1
    #Turns k_point_arr[i] in reciprocal cartesian basis to kinstance in reciprocal lattice relative basis
    kinstance = (np.linalg.inv(b_matsc.T).dot(k_point_arr_phon[i]) )
    na = round(kinstance[0]*float(nk1sc))
    nb = round(kinstance[1]*float(nk2sc))
    nc = round(kinstance[2]*float(nk3sc))
    while na >= nk1sc:
      na = na-nk1sc
    while nb >= nk2sc:
      nb = nb-nk2sc
    while nc >= nk3sc:
      nc = nc-nk3sc
    while na <= -nk1sc:
      na = na+nk1sc
    while nb <= -nk2sc:
      nb = nb+nk2sc
    while nc <= -nk3sc:
      nc = nc+nk3sc

    if kmesh_phon[na][nb][nc] == -1:
        kmesh_phon[na][nb][nc] = i
#Rotation occurring on k point described in relative basis set vectors rather than cartesian
for i in range(len(k_point_arr_phon)):
    for mat in rot_matphon:
        j=j+1
        #Rotation matrix must act on relative recirpocal basis set vectors rather than cartesian reciprocal. Matrix acting to the right must also be mat.T
        kinstance = mat.T.dot(np.linalg.inv(b_matsc.T).dot(k_point_arr_phon[i]))
        na = round(kinstance[0]*float(nk1sc))
        nb = round(kinstance[1]*float(nk2sc))
        nc = round(kinstance[2]*float(nk3sc))  
        while na >= nk1sc:
          na = na-nk1sc
        while nb >= nk2sc:
          nb = nb-nk2sc
        while nc >= nk3sc:
          nc = nc-nk3sc
        while na <= -nk1sc:
          na = na+nk1sc
        while nb <= -nk2sc:
          nb = nb+nk2sc
        while nc <= -nk3sc:
          nc = nc+nk3sc
        if kmesh_phon[na][nb][nc] == -1:
            kmesh_phon[na][nb][nc] = i

nucleon_mass_unit = 1822.89

nat_type = data_dict_sc["output"]["atomic_species"]['@ntyp']
nat= data_dict_sc["output"]["atomic_structure"]['@nat']
print("natom = ",nat)
atom_positions = np.empty((nat,3))
atom_masses = np.empty(nat)
i=0
for atom in data_dict_sc["output"]["atomic_structure"]["atomic_positions"]["atom"]:
  atom_positions[i] = np.array(atom["$"])
  for species in data_dict_sc["output"]["atomic_species"]["species"]:
    if species['@name'] == atom['@name']:
      atom_masses[i] = species["mass"]
  i=i+1
  
nat_typep = data_dict_phon["output"]["atomic_species"]['@ntyp']
natp= data_dict_phon["output"]["atomic_structure"]['@nat']
atom_positionsp = np.empty((natp,3))
atom_massesp = np.empty(natp)
i=0
for atom in data_dict_phon["output"]["atomic_structure"]["atomic_positions"]["atom"]:
  atom_positionsp[i] = np.array(atom["$"])
  for species in data_dict_phon["output"]["atomic_species"]["species"]:
    if species['@name'] == atom['@name']:
      atom_massesp[i] = species["mass"]
  i=i+1

total_mass = 0.0                                                                
for mass in atom_masses:                                                                         
  total_mass = total_mass + mass
  
sumx = np.zeros(3)
for i in range(nat):
  for k in range(nat):
    pos = np.linalg.inv(cellvec.T).dot(atom_positions[i])
    posp = np.linalg.inv(cellvec.T).dot(atom_positionsp[k])
    diffpos = pos- posp
    if diffpos.dot(diffpos) > (0.2**2):
      for j in range(3):
        while posp[j] < 0.0:
          posp[j] = posp[j] + 1.0
        while posp[j] > 1.0:
          posp[j] = posp[j] - 1.0      
        while pos[j] < 0.0:
          pos[j] = pos[j] + 1.0
        while pos[j] > 1.0:
          pos[j] = pos[j] - 1.0
        if (pos[j] - posp[j]) > 0.2:
          posp[j] = posp[j] + 1.0
        if (pos[j] - posp[j]) < -0.2:
          posp[j] = posp[j] -1.0

    diffpos = pos - posp

    disp = cellvec.dot(diffpos)
    if disp.dot(disp) < 0.2**2:
      midisp = (atom_masses[i]*(disp**2))/(total_mass)
      print(i,"atom mass = ",atom_masses[i],"\ndisp = ",disp,"\n")
      sumx = sumx + midisp
      break

x = np.sqrt(sumx[0]+sumx[1]+sumx[2])
print("x =",x)                      
M = total_mass* nucleon_mass_unit
print("M =",M)

#Calculate phonon frequency wphonon = [E"|x=0/Meff]^1/2
xphon = np.zeros(3)
xphon = [-x,0.0,x]
ediff = data_dict_phon["output"]["total_energy"]["etot"]-data_dict_sc["output"]["total_energy"]["etot"]
yphon = np.zeros(3)
yphon = [ediff,0.0,ediff]

#z[0] = E"|x=0
z = np.polyfit(xphon,yphon, deg=2)
wphonon = np.sqrt(2*z[0]/M)
print("wphonon =",HartreetoeV(wphonon)*1000," meV")
print("E diff = ", HartreetoeV(ediff)*1000," meV")
g = calc_gk(ediff,x,M,wphonon)
print("g = ", HartreetoeV(g)*1000," meV")

#Main development code

#Set density of states
NEf = float(args.NEf)*HartreeineV

nofbands = np.zeros(10)
glist = []
glistmeV = []
lamdalist = []

inwindow = float(args.window)/1000.0

nestwindow = eVtoHartree(inwindow) 
window = eVtoHartree(inwindow)
print("nest window =", HartreetoeV(nestwindow)*1000," meV")
print("wphonon =",HartreetoeV(wphonon)*1000," meV")

outfilestring1 = "outsmear.csv"
foutsm = open(outfilestring1,"w+")
foutsm.write("%8s\n" % ("Width Ry"))

outfilestring2 = "outlamda1.csv"
foutl1 = open(outfilestring2,"w+")
foutl1.write("%8.2f\n" % (HartreetoeV(wphonon)*1000))

q = [0,0,0]
maxdeltaE = 0.0
maxlamda_k = 0.0
print(q)
for cntgamma in range(1,10):
  glist_gamma = []
  glistmeV_gamma = []
  lamdatot = 0.0
  gamma = float(cntgamma) * 0.001/2.0

  for i in range(nk1sc):
    for j in range(nk2sc):
      for k in range(nk3sc):
        ksc = np.array([i,j,k],dtype=np.int32)
        kindex_sc = kmesh_sc[tuple(ksc)]
        kindex_phon= kmesh_phon[tuple(ksc)]
        
        ikq = i + q[0]
        jkq = j + q[1]
        kkq = k + q[2]
        if ikq >= nk1sc:
          ikq = ikq-nk1sc
        if jkq >= nk2sc:
          jkq = jkq-nk2sc
        if kkq >= nk3sc:
          kkq = kkq-nk3sc
        kindexkq = kmesh_sc[ikq][jkq][kkq]
        if kindexkq != kindex_sc:
          print("Warning,  kindexkq != kindex_sc")
        
        kqsc = np.array([ikq,jkq,kkq],dtype=np.int32)
        while kqsc[0] >= nk1sc:
          kqsc[0] = kqsc[0]-nk1sc
        while kqsc[1] >= nk2sc:
          kqsc[1] = kqsc[1]-nk2sc
        while kqsc[2] >= nk3sc:
          kqsc[2] = kqsc[2]-nk3sc
        kqindex_sc= kmesh_sc[tuple(kqsc)]
        kqindex_phon= kmesh_phon[tuple(kqsc)]
               
        bandlistk = []
        bandlistkq = []
        bandlistsc = []
        bandlistsckq = []
        for bndk in range(nbndsc):
          if ((eigenvalue_arr_sc[kindex_sc][bndk]-Efermisc)**2 < (nestwindow)**2):
            bandlistk.append(bndk)
        for bndkq in range(nbndsc):
          if ((eigenvalue_arr_sc[kindex_sc][bndkq]-Efermisc)**2 < (nestwindow)**2):
            bandlistkq.append(bndkq)
        for bndsc in range(nbndsc):
          if ((eigenvalue_arr_sc[kindex_sc][bndsc]-Efermisc)**2 < (window)**2):
            bandlistsc.append(bndsc)
        for bndsc in range(nbndsc):
          if ((eigenvalue_arr_sc[kqindex_sc][bndsc]-Efermisc)**2 < (window)**2):
            bandlistsckq.append(bndsc)
        nofbands[len(bandlistsc)] = nofbands[len(bandlistsc)] + 1
               
        deltaEmax1 = 0.0
        for bndsc1 in bandlistsc:
          deltaE1 = 0.0
          for bndsc2 in bandlistsc:
            deltaE1 = np.abs(np.abs((eigenvalue_arr_sc[kindex_sc][bndsc1]-Efermisc)-(eigenvalue_arr_sc[kindex_sc][bndsc2]-Efermisc))-\
                             np.abs((eigenvalue_arr_phon[kindex_phon][bndsc1]-Efermiphon)-(eigenvalue_arr_phon[kindex_phon][bndsc2]-Efermiphon)))            
            if deltaE1 >deltaEmax1:
              deltaEmax1 = deltaE1
        gk = calc_gk(np.abs(deltaEmax1),x,M,wphonon)
        
        for bndk in bandlistk:
          for bndkq in bandlistk:
            lamda_k = calc_lamda(gk, NEf, wphonon) * gaussian(gamma,eigenvalue_arr_sc[kindex_sc][bndk],Efermisc) * gaussian(gamma,eigenvalue_arr_sc[kindexkq][bndkq],Efermisc)     
            lamdalist.append(lamda_k)
            lamdatot = lamda_k + lamdatot
            glist_gamma.append(gk)
            glistmeV_gamma.append(HartreetoeV(gk)*1000)

  lamdafin = lamdatot/(nk1sc*nk2sc*nk3sc)
  print("%6.3f,%7.5f" % (gamma*2.0, lamdafin))
  foutsm.write("%8.3f\n" % (gamma*2.0))
  foutl1.write("%8.5f\n" % (lamdafin))
  glist.append(glist_gamma)
  glistmeV.append(glistmeV_gamma)




