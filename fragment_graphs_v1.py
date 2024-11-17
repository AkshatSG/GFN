import sys, os
import random
import time
import csv
import itertools
import copy
import joblib

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import dgl
import dgl.function as fn
import networkx as nx
import pickle
import pandas as pd 
import gym
import warnings

from rdkit import Chem  
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog

#---------------------------------------------------------------UTILITY FUNCTIONS--------------------------------------------------------
def get_att_points(mol):
	att_points = []
	for a in mol.GetAtoms(): 
		if a.GetSymbol() == '*':
			att_points.append(a.GetIdx())

	return att_points
	
ATOM_VOCAB = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl','Br', '*']  #Atom vocabulary for molecules

FRAG_VOCAB = open('motifs_350.txt','r').readlines() # n=350 with attachment points highlighted by dummy atoms (*)

FRAG_VOCAB = [s.strip('\n').split(',') for s in FRAG_VOCAB] 
FRAG_VOCAB_MOL = [Chem.MolFromSmiles(s[0]) for s in FRAG_VOCAB]
FRAG_VOCAB_ATT = [get_att_points(m) for m in FRAG_VOCAB_MOL]

#One-of-K encoding for a given molecule
def one_of_k_encoding_unk(x, allowable_set):
	"""Maps inputs not in the allowable set to the last element."""
	if x not in allowable_set:
		x = allowable_set[-1]  #All other atoms except those in vocab are considered DUMMY
	return list(map(lambda s: float(x == s), allowable_set))

#Edge features
def edge_feature(bond):
	bt = bond.GetBondType()
	return np.asarray([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()])

def atom_feature(atom, use_atom_meta):
	if use_atom_meta == False:
		return np.asarray(one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB))
	else:
		return np.asarray(one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB) + one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) + [atom.GetIsAromatic()])

# From GCPN
def convert_radical_electrons_to_hydrogens(mol):
	"""
	Converts radical electrons in a molecule into bonds to hydrogens. Only
	use this if molecule is valid. Results in a new mol object
	:param mol: rdkit mol object
	:return: rdkit mol object
	"""
	m = copy.deepcopy(mol)
	if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
		return m
	else:  # a radical
		for a in m.GetAtoms():
			num_radical_e = a.GetNumRadicalElectrons()
			if num_radical_e > 0:
				a.SetNumRadicalElectrons(0)
				a.SetNumExplicitHs(num_radical_e)
	return m

def adj2sparse(adj):
	"""
		adj: [3, 47, 47] float numpy array
		return: a tuple of 2 lists
	"""
	adj = [x*(i+1) for i, x in enumerate(adj)]
	adj = [sparse.dok_matrix(x) for x in adj]
	
	if not all([adj_i is None for adj_i in adj]):
		adj = sparse.dok_matrix(np.sum(adj))
		adj.setdiag(0)   

		all_edges = list(adj.items())
		e_types = np.array([edge[1]-1 for edge in all_edges], dtype=int)
		e = np.transpose(np.array([list(edge[0]) for edge in all_edges]))

		n_edges = len(all_edges)

		e_x = np.zeros((n_edges, 4))
		e_x[np.arange(n_edges),e_types] = 1
		e_x = torch.Tensor(e_x)
		return e, e_x
	else:
		return None

def map_idx(idx, idx_list, mol):
	abs_id = idx_list[idx]
	neigh_idx = mol.GetAtomWithIdx(abs_id).GetNeighbors()[0].GetIdx()
	return neigh_idx
#---------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------MOLECULE ENVIRONMENT------------------------------------------------
class MoleculeEnvironment:
	def __init__(self, starting_smi, predictor, data_type='zinc', reward_step_total=1, is_normalize=0, reward_type='crystal', reward_target=0.5, has_scaffold=False, has_feature=False, is_conditional=False, conditional='low', max_action=128, min_action=20, force_final=False):

		self.is_normalize = bool(is_normalize)
		self.has_feature = has_feature

		# init smi
		self.smi = starting_smi  #Scaffold or complete molecule for optimization  #XXX: Must have attachment points

		self.mol = Chem.MolFromSmiles(self.smi)
		self.smile_list = []
		self.smile_old_list = []

		possible_atoms = ATOM_VOCAB
		possible_motifs = FRAG_VOCAB
		possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
		self.atom_type_num = len(possible_atoms)
		self.motif_type_num = len(possible_motifs)
		self.possible_atom_types = np.array(possible_atoms)
		self.possible_motif_types = np.array(possible_motifs)
		self.possible_bond_types = np.array(possible_bonds, dtype=object)

		#self.d_n = len(self.possible_atom_types)+18 
		self.d_n = len(self.possible_atom_types)

		self.max_action = max_action  #128
		self.min_action = min_action  #20

		self.max_atom = 150  #Maximum number of atoms in the molecule can be 150 --> Can be tuned
		self.action_space = gym.spaces.MultiDiscrete([20, len(FRAG_VOCAB), 20])  #AP in source, motif, AP in motif

		self.counter = 0
		self.level = 0 # for curriculum learning, level starts with 0, and increase afterwards

		#self.predictor = DockingVina(docking_config)

		self.attach_point = Chem.MolFromSmiles('*')
		self.Na = Chem.MolFromSmiles('[Na+]')
		self.K = Chem.MolFromSmiles('[K+]')
		self.H = Chem.MolFromSmiles('[H]')
		
		self.proxy = MoleculeProxy(predictor)  #Reward function - activity prediction model
		#self.agent = GFlowNetAgent(self)  #GFN agent

	def seed(self,seed):
		np.random.seed(seed=seed)
		random.seed(seed)
		
	def reset_batch(self):
		self.smile_list = []

	def normalize_adj(self,adj):
		degrees = np.sum(adj,axis=2)

		D = np.zeros((adj.shape[0],adj.shape[1],adj.shape[2]))
		for i in range(D.shape[0]):
			D[i,:,:] = np.diag(np.power(degrees[i,:],-0.5))
		adj_normal = D@adj@D
		adj_normal[np.isnan(adj_normal)]=0
		return adj_normal

	def reward_single(self, smile_list):
		return self.proxy.calculate_reward(self.mol)

	def step(self, ac):
		"""
		Perform a given action
		:param action:
		:param action_type:
		:return: reward of 1 if resulting molecule graph does not exceed valency,
		-1 if otherwise
		"""
		#ac = ac[0]
		 
		### init
		info = {}  # info we care about
		self.mol_old = copy.deepcopy(self.mol) # keep old mol
		
		stop = False	
		new = False
		
		if (self.counter >= self.max_action) or get_att_points(self.mol) == []:
			new = True
		else:
			state_atom = self._add_motif(ac) # problems here

		reward_step = 0.05
		if self.mol.GetNumAtoms() > self.mol_old.GetNumAtoms():
			reward_step += 0.005
		self.counter += 1

		if new:			
			reward = 0
			# Only store for obs if attachment point exists in o2
			if get_att_points(self.mol) != []:
				mol_no_att = self.get_final_mol() 
				Chem.SanitizeMol(mol_no_att, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
				smi_no_att = Chem.MolToSmiles(mol_no_att)
				info['smile'] = smi_no_att
				print("smi:", smi_no_att)
				self.smile_list.append(smi_no_att)

				# Info for old mol
				mol_old_no_att = self.get_final_mol_ob(self.mol_old)
				Chem.SanitizeMol(mol_old_no_att, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
				smi_old_no_att = Chem.MolToSmiles(mol_no_att)
				info['old_smi'] = smi_old_no_att
				self.smile_old_list.append(smi_old_no_att)

				stop = True
			else:
				stop = False
			self.counter = 0	  

		### use stepwise reward
		else:
			reward = reward_step

		info['stop'] = stop

		# get observation
		ob = self.get_observation()
		return ob,reward,new,info,state_atom

	def reset(self,smile=None):
		'''
		to avoid error, assume an atom already exists
		:return: ob
		'''
		if smile is not None:
			self.mol = Chem.RWMol(Chem.MolFromSmiles(smile))
			Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
		else:
			# init smi
			self.smi = "C(=N[*:2])c1ccc([*:1])cc1"  #Default molecule is a benzene with an attachment point
			self.mol = Chem.MolFromSmiles(self.smi)
			
		# self.smile_list = [] # only for single motif
		self.counter = 0
		ob = self.get_observation()
		return ob

	def sample_motif(self):
		go_on = True
		while go_on:
			#print(Chem.MolToSmiles(self.mol))
			cur_mol_atts = get_att_points(self.mol)
			#print(len(cur_mol_atts), Chem.MolToSmiles(self.mol))
			try:
				ac1 = np.random.randint(len(cur_mol_atts))
			except:
				self.reset()
				cur_mol_atts = get_att_points(self.mol)
				ac1 = np.random.randint(len(cur_mol_atts))
				
			ac2 = np.random.randint(self.motif_type_num)
			motif = FRAG_VOCAB_MOL[ac2]
			ac3 = np.random.randint(len(FRAG_VOCAB_ATT[ac2]))
			a = self.action_space.sample()
			
			a[0] = ac1
			a[1] = ac2
			a[2] = ac3

			go_on = False
		return a

	def _add_motif(self, ac):
		#print(Chem.MolToSmiles(self.mol), Chem.MolToSmiles(self.attach_point), Chem.MolToSmiles(self.Na), ac)
		#OC1=CC=C[*:1]C=C1 * [Na+] [  0 287   0]
		cur_mol = Chem.ReplaceSubstructs(self.mol, self.attach_point, self.Na)[ac[0]]
		#print(Chem.MolToSmiles(cur_mol))  #OC1=CC=C[Na+]C=C1
		motif = FRAG_VOCAB_MOL[ac[1]]
		#print(Chem.MolToSmiles(motif))  #OC(CN[*:2])(c1ccc(F)cc1)[*:1]
		att_point = FRAG_VOCAB_ATT[ac[1]]
		#print(att_point)  #[4, 12]
		motif_atom = map_idx(ac[2], att_point, motif)
		#print(motif_atom)  #3
		motif = Chem.ReplaceSubstructs(motif, self.attach_point, self.K)[ac[2]]
		#print(Chem.MolToSmiles(motif))  #OC(CN[K+])(c1ccc(F)cc1)[*:1]
		motif = Chem.DeleteSubstructs(motif, self.K)
		#print(Chem.MolToSmiles(motif))  #NCC(O)(c1ccc(F)cc1)[*:1]
		next_mol = Chem.ReplaceSubstructs(cur_mol, self.Na, motif, replacementConnectionPoint=motif_atom)[0]
		#print(Chem.MolToSmiles(next_mol))  #OC1=CC=CN(CC(O)(c2ccc(F)cc2)[*:1])C=C1
		self.mol = next_mol
		return motif_atom

	def get_final_smiles_mol(self):
		"""
		Returns a SMILES of the final molecule. Converts any radical
		electrons into hydrogens. Works only if molecule is valid
		:return: SMILES
		"""
		m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
		m = convert_radical_electrons_to_hydrogens(m)
		return m, Chem.MolToSmiles(m, isomericSmiles=True)

	def get_final_mol(self):
		"""
		Returns a rdkit mol object of the final molecule. Converts any radical
		electrons into hydrogens. Works only if molecule is valid
		:return: SMILES
		"""
		m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
		return m
	
	def get_final_mol_ob(self, mol):
		m = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
		return m

	def get_observation(self, expert_smi=None):
		"""
		ob['adj']:d_e*n*n --- 'E'
		ob['node']:1*n*d_n --- 'F'
		n = atom_num + atom_type_num
		"""
		ob = {}

		if expert_smi:
			mol = Chem.MolFromSmiles(expert_smi)
		else:
			ob['att'] = get_att_points(self.mol)
			mol = copy.deepcopy(self.mol)
		
		try:
			Chem.SanitizeMol(mol)
		except:
			pass

		smi = Chem.MolToSmiles(mol)

		n = mol.GetNumAtoms()
		F = np.zeros((1, self.max_atom, self.d_n))  #self.max_atom = 150

		for a in mol.GetAtoms():
			atom_idx = a.GetIdx()
			
			atom_symbol = a.GetSymbol()
			if self.has_feature:
				float_array = atom_feature(a, use_atom_meta=True)
			else:
				float_array = (atom_symbol == self.possible_atom_types).astype(float)

			F[0, atom_idx, :] = float_array

		d_e = len(self.possible_bond_types)
		E = np.zeros((d_e, self.max_atom, self.max_atom))

		for b in mol.GetBonds(): 
			begin_idx = b.GetBeginAtomIdx()
			end_idx = b.GetEndAtomIdx()
			bond_type = b.GetBondType()
			float_array = (bond_type == self.possible_bond_types).astype(float)
			try:
				assert float_array.sum() != 0
			except:
				print('error',bond_type)
			E[:, begin_idx, end_idx] = float_array
		
		if self.is_normalize:
			E = self.normalize_adj(E)
		
		ob_adj = adj2sparse(E.squeeze())
		ob_node = torch.Tensor(F)
		g = dgl.DGLGraph()
		#g = dgl.graph()

		ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
		g.add_nodes(ob_len)
		if ob_adj is not None and len(ob_adj[0])>0 :
			g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
		g.ndata['x'] = ob_node[:, :int(ob_len),:].squeeze(0)
		
		ob['g'] = g
		ob['smi'] = smi
		ob['node_features'] = ob_node
		ob['adj'] = E
		
		return ob

	def get_observation_mol(self,mol):
		"""
		ob['adj']:d_e*n*n --- 'E'
		ob['node']:1*n*d_n --- 'F'
		n = atom_num + atom_type_num
		"""
		ob = {}

		ob['att'] = get_att_points(mol)
		
		try:
			Chem.SanitizeMol(mol)
		except:
			pass

		smi = Chem.MolToSmiles(mol)

		n = mol.GetNumAtoms()
		F = np.zeros((1, self.max_atom, self.d_n))

		for a in mol.GetAtoms():
			atom_idx = a.GetIdx()
			
			atom_symbol = a.GetSymbol()
			if self.has_feature:
				float_array = atom_feature(a, use_atom_meta=True)
			else:
				float_array = (atom_symbol == self.possible_atom_types).astype(float)

			F[0, atom_idx, :] = float_array

		d_e = len(self.possible_bond_types)
		E = np.zeros((d_e, self.max_atom, self.max_atom))

		for b in mol.GetBonds(): 

			begin_idx = b.GetBeginAtomIdx()
			end_idx = b.GetEndAtomIdx()
			bond_type = b.GetBondType()
			float_array = (bond_type == self.possible_bond_types).astype(float)

			try:
				assert float_array.sum() != 0
			except:
				print('error',bond_type)
			E[:, begin_idx, end_idx] = float_array
		
		if self.is_normalize:
			E = self.normalize_adj(E)
		
		ob_adj = adj2sparse(E.squeeze())
		ob_node = torch.Tensor(F)
		g = dgl.DGLGraph()
		#g = dgl.graph()

		ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
		g.add_nodes(ob_len)
		if ob_adj is not None and len(ob_adj[0])>0 :
			g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
		g.ndata['x'] = ob_node[:, :int(ob_len),:].squeeze(0)
		
		ob['g'] = g
		ob['smi'] = smi
		return ob
	
	#Based on the starting molecule and its attachment points, the number of actions will increase	
	def get_possible_actions(self):
		actions = []
		num_frags = len(FRAG_VOCAB)
		att_point_source = len(get_att_points(self.mol))

		for att1 in range(att_point_source):
			for f in range(num_frags):
				att_point_frag = len(get_att_points(FRAG_VOCAB_MOL[f]))
				for att2 in range(att_point_frag):
					actions.append([att1, f, att2])
		return actions
	
#---------------------------------------------------------------------PROXY---------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem
from rdkit.Contrib.SA_Score import sascorer

'''
class MoleculeProxy:
	def __init__(self, target_weight=500, target_logp=2.5, max_hbd=5, max_hba=10):
		self.target_weight = target_weight
		self.target_logp = target_logp
		self.max_hbd = max_hbd
		self.max_hba = max_hba

	def calculate_reward(self, mol):
		if mol is None or mol.GetNumAtoms() == 0:
			return -10

		mol_copy = Chem.Mol(mol)
		try:
			Chem.SanitizeMol(mol_copy)
			AllChem.Compute2DCoords(mol_copy)
		except:
			return -10

		mol_weight = ExactMolWt(mol)
		logp = Crippen.MolLogP(mol)
		hbd = rdMolDescriptors.CalcNumHBD(mol)
		hba = rdMolDescriptors.CalcNumHBA(mol)
		sa_score = self.calculate_sa_score(mol)

		weight_reward = 1 - abs(mol_weight - self.target_weight) / self.target_weight
		logp_reward = 1 - abs(logp - self.target_logp) / max(abs(self.target_logp), 1)
		hbd_reward = 1 if hbd <= self.max_hbd else 0
		hba_reward = 1 if hba <= self.max_hba else 0
		sa_reward = 1 - sa_score / 10

		total_reward = (weight_reward + logp_reward + hbd_reward + hba_reward + sa_reward) / 5
		return max(0, total_reward)

	def calculate_sa_score(self, mol):
		# return AllChem.CalcSyntheticAccessibilityScore(mol)
		return sascorer.calculateScore(mol)
'''

class MoleculeProxy:
	def __init__(self, model):
		self.model = model

	def calculate_reward(self, mol):
		if mol is None or mol.GetNumAtoms() == 0:
			return -10
		mol_copy = Chem.Mol(mol)
		try:
			#Chem.SanitizeMol(mol_copy)  #Sanitization can be very stringent in RDKit (leading to kekulization issues)
			AllChem.Compute2DCoords(mol_copy)
		except:
			return -1
			
		fp = AllChem.GetMorganFingerprintAsBitVect(mol_copy, 2, nBits=1024)
		fp_array = np.array(fp)
		fp_array = fp_array.reshape(1, -1)
		prediction = self.model.predict(fp_array)[0]
		#print(prediction)
		#return max(0, prediction)  #To avoid negative permeability values
		return prediction  #XXX: Negative rewards can lead to insignificant loss values

#------------------------------------------------------------------POLICY NETWORK-----------------------------------------------------	   
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class PolicyNetwork(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(PolicyNetwork, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)
'''

#Molecular graph encoder module - input to the GFN policy network
class MolecularGraphEncoder(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(MolecularGraphEncoder, self).__init__()
		self.conv1 = nn.Linear(input_dim, hidden_dim)
		self.conv2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc = nn.Linear(hidden_dim, output_dim)
	
	def forward(self, x, adjacency_matrix):
		# x: Node features
		# adjacency_matrix: Adjacency matrix of the graph
		#print(x.size(), adjacency_matrix.size())  #torch.Size([1, 150, 10]) torch.Size([4, 150, 150])
		h = torch.relu(self.conv1(torch.matmul(adjacency_matrix, x)))
		h = torch.relu(self.conv2(torch.matmul(adjacency_matrix, h)))
		h = self.fc(h.mean(dim=0))  # Aggregate node features
		return h
		
class GFlowNetPolicy(nn.Module):
	def __init__(self, latent_dim, num_actions):
		super(GFlowNetPolicy, self).__init__()
		self.fc1 = nn.Linear(latent_dim, latent_dim)
		self.fc2 = nn.Linear(latent_dim, num_actions)
	
	def forward(self, latent_representation):
		x = torch.relu(self.fc1(latent_representation))
		action_logits = self.fc2(x)
		action_probs = torch.softmax(action_logits, dim=-1)
		return action_probs  #Probability of a fragment getting added to a molecular structure
#--------------------------------------------------------------------------------------------------------------------------------------	

#-------------------------------------------------------------------GFN AGENT----------------------------------------------------------	
'''
#TODO: Must be replaced with a graph neural network instead of MLP layers
#Takes the state (molecular graph) as input and predicts the probability of the next fragment
class GFlowNetAgent:
	def __init__(self, env, hidden_dim=64):
		self.env = env
		self.input_dim = self.get_state_dim()
		self.output_dim = self.get_action_dim()

		self.forward_policy = PolicyNetwork(self.input_dim, hidden_dim, self.output_dim)
		self.backward_policy = PolicyNetwork(self.input_dim, hidden_dim, self.output_dim)

	def get_state_dim(self):
		return 4

	def get_action_dim(self):
		return len(self.env.action_space)

	def state_to_tensor(self, state):
		return torch.tensor([state['num_atoms'], state['num_bonds'], len(state['atom_types']), len(state['bond_types'])], dtype=torch.float32)

	def forward_action(self, state):
		state_tensor = self.state_to_tensor(state)
		possible_actions = self.env.get_possible_actions()
		with torch.no_grad():
			action_probs = F.softmax(self.forward_policy(state_tensor), dim=0)
		action_index = torch.multinomial(action_probs, 1).item()
		return possible_actions[action_index]

	def backward_action(self, state):
		state_tensor = self.state_to_tensor(state)
		possible_actions = self.env.get_possible_actions()
		with torch.no_grad():
			action_probs = F.softmax(self.backward_policy(state_tensor), dim=0)
		action_index = torch.multinomial(action_probs, 1).item()
		return possible_actions[action_index]
'''

class GFlowNetAgent(nn.Module):
	def __init__(self, input_dim, hidden_dim, latent_dim, num_actions):
		super(GFlowNetAgent, self).__init__()
		self.encoder = MolecularGraphEncoder(input_dim, hidden_dim, latent_dim)
		self.forward_policy = GFlowNetPolicy(latent_dim, num_actions)
		self.backward_policy = GFlowNetPolicy(latent_dim, num_actions)
	
	def forward(self, x, adjacency_matrix):
		latent_representation = self.encoder(x, adjacency_matrix)
		forward_probs = self.forward_policy(latent_representation)
		backward_probs = self.backward_policy(latent_representation)
		return forward_probs, backward_probs
#--------------------------------------------------------------------------------------------------------------------------------------	

#-------------------------------------------------------LOSS FUNCTIONS-----------------------------------------------------------------
import torch
import torch.nn.functional as F

def trajectory_balance_loss(forward_probs, backward_probs, trajectory, reward):
	# forward_probs: Forward policy probabilities for each step in the trajectory
	# backward_probs: Backward policy probabilities for each step in the trajectory
	# trajectory: A list of state-action pairs representing the trajectory
	# reward: Reward for the final state

	forward_log_prob = 0.0
	backward_log_prob = 0.0

	# Calculate the log-probabilities for forward and backward policies along the trajectory
	for state, action in trajectory:
		forward_log_prob += torch.log(forward_probs[state][action]).detach().clone().requires_grad_(True)
		backward_log_prob += torch.log(backward_probs[state][action]).detach().clone().requires_grad_(True)

	# Compute the trajectory balance loss
	tb_loss = (forward_log_prob - backward_log_prob - torch.log(torch.tensor(reward, requires_grad=True)))**2
	#print(forward_log_prob, backward_log_prob, reward, tb_loss)
	return tb_loss

'''
def calculate_trajectory_balance_loss(trajectory, agent):
	  loss = torch.tensor(0.0, requires_grad=True)
	  Z = torch.tensor(1.0, requires_grad=True)  # Partition function as a tensor
	  epsilon = 1e-10  # Small constant to prevent log(0)

	  for i in range(len(trajectory) - 1):
		  state, action, next_state, reward, possible_actions = trajectory[i]
		  next_action = trajectory[i+1][1] if i < len(trajectory) - 2 else None

		  state_tensor = agent.state_to_tensor(state)
		  next_state_tensor = agent.state_to_tensor(next_state)

		  forward_logits = agent.forward_policy(state_tensor)
		  backward_logits = agent.backward_policy(next_state_tensor)

		  action_index = possible_actions.index(action)

		  forward_prob = F.softmax(forward_logits, dim=0)[action_index]
		  backward_prob = F.softmax(backward_logits, dim=0)[action_index] if next_action else torch.tensor(1.0)

		  if i == 0:
			  loss = loss + torch.log(Z) + torch.log(forward_prob)
		  elif i == len(trajectory) - 2:
			  # Add epsilon to prevent log(0)
			  loss = loss + torch.log(torch.tensor(max(reward, epsilon), dtype=torch.float32)) - torch.log(backward_prob)
		  else:
			  loss = loss + torch.log(forward_prob) - torch.log(backward_prob)

	  return -loss  # Negative because we want to maximize this quantity
'''
#----------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------TRAINING LOOP-------------------------------------------------------
import torch.nn as nn
import torch.optim as optim

def train_gflownet(env, input_dim, hidden_dim, latent_dim, n_actions, all_actions, num_episodes=1000, learning_rate=1e-4, clip_value=1.0):
	agent = GFlowNetAgent(input_dim, hidden_dim, latent_dim, num_actions)
	optimizer = optim.Adam(list(agent.forward_policy.parameters()) + list(agent.backward_policy.parameters()), lr=learning_rate)

	for episode in range(num_episodes):
		try:
			#action = env.sample_motif()
			act_idx = np.random.randint(0, len(all_actions))
			action = all_actions[act_idx]
			trajectory = env.step(action)
		except:  #No. of atoms in molecule exceeds 150
			env.reset()
			#action = env.sample_motif()
			act_idx = np.random.randint(0, len(all_actions))
			action = all_actions[act_idx]
			trajectory = env.step(action)
		
		for i, ac in enumerate(all_actions):
			if(tuple(ac)==tuple(action)):
				final_index = i
		
		traj_final = [(trajectory[-1], final_index)]  #State atom, action pair
		#print(traj_final)
		if not trajectory:
			continue
			
		#print(trajectory)  #({'att': [2], 'g': Graph(num_nodes=20, num_edges=22, ndata_schemes={'x': Scheme(shape=(10,), dtype=torch.float32)} edata_schemes={'x': Scheme(shape=(4,), dtype=torch.float32)}), 'smi': 'OC1=[*:1]C=CC=C2(C=C1)CCCCC2OC1CCCC1'}, 0.055, False, {'stop': False})
		
		#TODO: Node feature matrix and adjacency matrix from DGL graph
		#reward = trajectory[1]  #TODO: Should be replaced with the actual reward proxy model
		#reward = env.reward_single((trajectory[0]['smi']))
		reward = env.proxy.calculate_reward(env.get_final_mol())
		reward = float(reward*-1)
		#print(trajectory[0]['smi'], Chem.MolToSmiles(env.get_final_mol()), reward)
		
		#print(trajectory)
		node_features = torch.tensor(trajectory[0]['node_features']).float()
		adj_matrix = torch.tensor(trajectory[0]['adj']).float()
		forward_probs, backward_probs = agent(node_features, adj_matrix)
		#print(forward_probs.size(), backward_probs.size())  #torch.Size([150, 140000]) torch.Size([150, 140000])
		loss = trajectory_balance_loss(forward_probs, backward_probs, traj_final, reward)
		#loss = calculate_trajectory_balance_loss(trajectory, agent)

		if torch.isnan(loss) or torch.isinf(loss):
			print(f"Warning: Invalid loss value at episode {episode}. Skipping.")
			continue

		optimizer.zero_grad()
		loss.backward()

		# Add gradient clipping
		#nn.utils.clip_grad_norm_(agent.forward_policy.parameters(), clip_value)
		#nn.utils.clip_grad_norm_(agent.backward_policy.parameters(), clip_value)

		optimizer.step()
		env.reset()

		if episode % 100 == 0:
			print(f"Episode {episode}, Loss: {loss.item()}")
			evaluate_model(env, all_actions)

		#if episode % 1000 == 0:
			#evaluate_model(env)
#---------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------MODEL EVALUATION----------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import Descriptors

def evaluate_model(env, all_actions, num_samples=100):
	valid_molecules = 0
	total_reward = 0
	unique_smiles = set()

	for _ in range(num_samples):
		try:
			#action = env.sample_motif()
			act_idx = np.random.randint(0, len(all_actions))
			action = all_actions[act_idx]
			trajectory = env.step(action)
			final_state = Chem.MolToSmiles(env.get_final_mol())
			mol = Chem.MolFromSmiles(final_state)  #To check that the final SMILES is actually valid

			if mol:
				valid_molecules += 1
				reward = env.proxy.calculate_reward(mol)  #TODO: Should be replaced with the actual reward proxy model
				reward = float(reward*-1)
				total_reward += reward
				smiles = Chem.MolToSmiles(mol)
				unique_smiles.add(smiles)
				#env.reset()
		except:
			env.reset()  #To prevent no. of atoms from exceeding 150

	validity_rate = valid_molecules / num_samples
	avg_reward = total_reward / valid_molecules if valid_molecules > 0 else 0
	uniqueness_rate = len(unique_smiles) / valid_molecules if valid_molecules > 0 else 0

	print(f"Validity rate: {validity_rate:.2f}")
	print(f"Average reward: {avg_reward:.2f}")
	print(f"Uniqueness rate: {uniqueness_rate:.2f}")

	# Print some example molecules
	print("Example generated molecules:")
	for smiles in list(unique_smiles):
		print(smiles, env.proxy.calculate_reward(Chem.MolFromSmiles(smiles)))
#---------------------------------------------------------------------------------------------------------------------------------------

#print(FRAG_VOCAB)
predictor = joblib.load("random_forest_model.joblib", "r")
env = MoleculeEnvironment("C(=N[*:2])c1ccc([*:1])cc1", predictor, reward_target=7.0)
#print(env.action_space)  #MultiDiscrete([ 20 350  20])
#action = env.sample_motif()
#results = env.step(action)
#env._add_motif(action)
#print(results)

proxy = MoleculeProxy(predictor)
reward = proxy.calculate_reward(Chem.MolFromSmiles("C(=N)c1ccccc1"))
print(reward, abs(reward), type(reward))

input_dim = 10
hidden_dim = 32
latent_dim = 64
valid_actions = env.get_possible_actions()
print("No. of valid actions: "+str(len(valid_actions)))
num_actions = 20*350*20
print("No. of possible actions: "+str(num_actions))

import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=DeprecationWarning)
train_gflownet(env, input_dim, hidden_dim, latent_dim, num_actions, valid_actions, 5000)		
		
		
		
		
