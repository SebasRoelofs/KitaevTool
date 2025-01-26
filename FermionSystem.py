from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import block_diag, null_space

from typing import Callable
from functools import partial
from IPython.display import display, Math,Markdown

import numpy as np
import matplotlib.pyplot as plt

import time

def hamming_weight(states: np.ndarray):
    '''
    Constant-time method for counting number of 1's in a number in binary.
    Works up to 32-bit integers. 
        
    Shoutout to stackoverflow.

    Args:
        states (np.ndarray): Array of integers
    Returns:
        np.ndarray: Number of 1's in each inputs' binary representation
    '''
    count = states - ((states >> 1) & 0o33333333333) - ((states >> 2) & 0o11111111111)
    return ((count + (count >> 3)) & 0o30707070707) % 63

############################################################################################
#### Adapted code snippets from Chun-Xiao for generating current from a Transition rate matrix
#### by solving rate equation
def n_F(E, mu, kBT):
    energy = (E - mu) / kBT
    n = (np.exp(energy) + 1)**-1
    return n

def get_P_old(rate_total, N_state):
    rate_matrix = np.zeros((N_state+1, N_state))
    rate_matrix[0:N_state, 0:N_state] = rate_total
    for k in range(N_state):
        rate_matrix[k, k] = -np.sum(rate_total[:, k])
    rate_matrix[N_state, :] = np.ones(N_state)
    
    right_vec = np.zeros((N_state+1, 1))
    right_vec[N_state, 0] = 1
    P_vec = np.linalg.pinv(rate_matrix) @ right_vec
    return P_vec

## Minor speedup compared to above code
def get_P_vectorized(rate_total):
    N_states = np.shape(rate_total)[0]

    rate_matrix = np.zeros((N_states+1, N_states))
    rate_matrix[0:N_states,0:N_states] = rate_total
    np.fill_diagonal(rate_matrix, -np.sum(rate_total, axis=0)) 
    rate_matrix[N_states,:] = np.ones(N_states)

    right_vec = np.zeros((N_states+1,1))
    right_vec[N_states,0] = 1
    P_vec = np.linalg.pinv(rate_matrix) @ right_vec
    return P_vec

def get_current(rate_plus_list, rate_minus_list, P_vec, num_of_leads):
    Is = np.zeros(num_of_leads)
    for j in range(num_of_leads):
        Is[j] = np.sum((rate_plus_list[j] - rate_minus_list[j]) @ P_vec)
    return Is

def get_Is(num_of_leads, Tsq_plus_list, Tsq_minus_list, gammas, mus, Es_ba, kBT):
    rate_plus_list = []
    rate_minus_list = []
    rate_total = 0
    for k in range(num_of_leads):
        rate_plus = gammas[k] * Tsq_plus_list[k] * n_F(Es_ba, mus[k], kBT)
        rate_minus = gammas[k] * Tsq_minus_list[k] * (np.ones(np.shape(Es_ba)) - n_F(-Es_ba, mus[k], kBT))
        rate_plus_list.append(rate_plus)
        rate_minus_list.append(rate_minus)
        rate_total += rate_plus + rate_minus
   
    P_vec = get_P_old(rate_total = rate_total, N_state=np.shape(rate_total)[0])
    #P_vec = get_P(rate_total=rate_total, N_state = np.shape(rate_total)[0])

    Is = get_current(rate_plus_list, rate_minus_list, P_vec, num_of_leads)
    return Is

### n_F(mu=0) is calculated a redundant amount of times above
### Below gives a (very minimal) speed-up when number of leads gets larger
### By calculated the Nf terms only once and passing them to get_Is
def get_Is_vectorized(num_of_leads, Tsq_plus_list, Tsq_minus_list, gammas, mus, Nf_0_plus,Nf_0_min,Nf_mu_plus,Nf_mu_min):
    nF_plus_array = np.array([Nf_mu_plus*gammas[i] if mus[i] != 0 else Nf_0_plus*gammas[i] for i in range(len(mus))])
    nF_min_array = np.array([Nf_mu_min*gammas[i] if mus[i] != 0 else Nf_0_min*gammas[i] for i in range(len(mus))])

    rate_plus_list = Tsq_plus_list * nF_plus_array
    rate_minus_list = Tsq_minus_list * nF_min_array

    rate_total = np.sum(rate_plus_list,axis=0)+ np.sum(rate_minus_list,axis=0)

    P_vec = get_P_vectorized(rate_total = rate_total)

    Is = get_current(rate_plus_list, rate_minus_list, P_vec, num_of_leads)
    return Is
########################################################################################

class FermionSystem:
    def __init__(self, N: int, store_fock_states: bool=True):
        '''
        Class for handling fermionic operator and states logic with binary operations
        Args: 
            N: the number of Fermionic sites in the system
            pos_bits: the number of bits required to store the position an operator acts on
            store_fock_states: whether to generate a list of all integers representing the fock states
            fock_states: list of all possible states (integers) in a system of size N
                         In principle no need to generate this list, can choose to make it optional 
                         since memory becomes an issue otherwise for large N
        '''
        self.N = N
        self.pos_bits = int(np.ceil(np.log2(2*N)))
        if store_fock_states:
            self.fock_states = np.arange(0,2**(2*N), dtype=np.int32)
 
    def operator(self,type: str, position: int, spin: str):
        '''
        Verbose method for constructing an operator, passes the call on to 
        self.construct_operator
        Args:
            type (str): specify whether to make an annihilation or creation operator
            position (int): the fermionic site to act on
            spin (str): 'up' or 'down'
        Returns:
            oper (int): integer representing the operator
        '''
        creation = (type=='creation')
        spin = 0 if (spin=='down') else 1
        oper = self.construct_operator(position, spin,creation)
        return oper
        
    def construct_operator(self,position: int, spin: int, creation: bool = True):
        ''' 
        Creates the integer representing an operator. The first bits store the bit position of 
        the operator to act on, which depends on position and spin
        The largest bit is set or unset depending on whether it stores a creation or annihilation operator
        Args:
            position (int): fermionic site of operator to act on
            spin (int): 0 = spin down, 1 = spin up
            creation(bool): True = creation, False = annihiliation

        Returns:
            oper (int): integer representing an operator
        '''
        shift = (position<<1) + spin
        oper = shift + (creation << self.pos_bits)
        return oper

    def act_oper(self,oper: int,states: np.ndarray):
        '''
        Applies an operator to a list of states, to generate the new states
        A destructed state is shown as -1, to distuinguish it from the empty state (0)
        (Alternative option is to shift each state up by 1)

        Args:
            oper (int): integer representing an operator
            states (ndarray): list of integers representing Fock states
        Returns:
            new_states: list of the states after applying the operator
            signs: list of signs resulting from applying the operator 
        '''
        ## Check the creation/annihilation bit
        check_bit = ((1 << self.pos_bits) & oper != 0)
    
        ## Get the position of the bit to flip
        flip_bit_pos = oper & int('1'* self.pos_bits,2)
        flip_bit = (1<<flip_bit_pos)

        ## Calculate relative signs of the operator
        parity_bits = (states & (int('1'*(flip_bit_pos+1),2)>>1))
        signs = hamming_weight(states&parity_bits) &0B1
        signs = signs*-2+1
        
        ## Flip the bit that the operator acts on
        new_states = states ^ flip_bit
    
        ## Check the original state of the flipped bit with the creation/annihilation bit
        destroyed = (flip_bit&states != 0) == check_bit
        new_states[destroyed] = -1 ## -1 = destroyed state (0 is already taken by the empty state)
                                   ## Benefit: any state *-1 will also be <0
        return new_states, signs

    def act_oper_list(self,oper_list, states,rel_sign=1):
        '''
        Apply a sequence of operators to a list of states,
        removes any 'destroyed' terms
        Args:
            oper_list: list of ints representing a sequence of states
            states: list of ints representing Fock states
            rel_sign: the relative sign of the operator
        Returns:
            states: list of the original states which were not destroyed
            new_states: resulting new states
            signs: relative sign after applying the sequence of operators
        '''
        signs = np.full(len(states),1)
        new_states = states
        for oper in oper_list:
            new_states, new_parity = self.act_oper(oper,new_states)
            filter_states = np.where(new_states > -1)
            new_states = new_states[filter_states]
            signs = signs[filter_states]
            new_parity = new_parity[filter_states]
            states = states[filter_states]
            signs *= new_parity
        return states,new_states,signs*rel_sign

    def normal_order(self,oper_list: list):
        ''' 
        Place sequence of operators in normal order, tracking the sign
        Normal order = sorting the list of integers from smallest to largest
        Sorting done with a bubble sort. 
        To do: handle the presence of same-site operators (for now ignored)
        Args:
            oper_list (list)
        '''
        ferm_sign=1
        for i in range(len(oper_list)-1):
            flag_swap = False
            for j in range(len(oper_list)-1):
                if oper_list[j] > oper_list[j+1]:
                    flag_swap=True
                    temp = oper_list[j+1]
                    oper_list[j+1] = oper_list[j] 
                    oper_list[j] = temp
                    ferm_sign*= -1
            ## if no swap took place, list is sorted
            if not flag_swap:
                return oper_list,ferm_sign
        return oper_list,ferm_sign

    def state_to_oper_list(self,state:int):
        '''
        Convert from a state to the list of creation operators 
        to create the state from vaccuum.
        '''
        oper_list = []
        pos = 0
        while state:
            if (state & 0B1):
                oper = self.construct_operator(int(np.floor(pos/2)), pos%2, creation=True)
                oper_list.append(oper)
            state = state >> 1
            pos += 1
        return oper_list


    def state_to_state(self, state_1:int, state_2: int):
        '''
        Given 2 states, returns the sequence of operators
        to go from state_1 to state_2 (in normal order).
        '''
        state_diff = state_1^state_2
        pos = 0
        oper_list = []
        while state_diff:
            if (state_diff & 0B1):
                creation = state_2 & 0B1
                oper = self.construct_operator(int(np.floor(pos/2)), pos%2, creation=creation)
                oper_list.append(oper)
            state_diff = state_diff >> 1
            state_1 = state_1 >> 1
            state_2 = state_2 >> 1
            pos+= 1
        normal_ordered, sign = self.normal_order(oper_list)
        return normal_ordered,sign

    '''
    Version that works with single-operator sequences. More general version included below
    def bra_oper_ket(self, states, phi, operators):
        size = len(phi[:,0])
        T_ij = np.zeros((size,size), dtype=complex)
        zero_col = np.zeros(size, dtype=complex)
        for oper in operators:
            new_states,signs = self.act_oper(oper, states)

            trans_array = [phi[:,np.where(new_states == states[i])[0]].T[0]*signs[np.where(new_states == states[i])[0]] if (states[i] in new_states) else zero_col for i in range(len(states))]
            T_ij += np.conj(phi) @ trans_array
        return T_ij
    '''

    def bra_oper_ket(self, states: list, phi: np.ndarray , operators:list):
        '''
        Calculate <phi|operator|phi> matrix for an operator sequence
        If multiple operator sequences are included, the contributions will be summed
        Args:
            states: list of states to operate on
            phi: array of weights of the states
            operators: list of operator lists to apply
        '''
        size = len(phi[:,0])
        T_ij = np.zeros((size,size), dtype=complex)
        zero_col = np.zeros(size, dtype=complex)
        trans_array = np.zeros((len(phi[0]), len(phi)), dtype='complex')

        for oper_list in operators:
            old_states, new_states,signs = self.act_oper_list(oper_list, states)
            
            col_indices = [np.where(states == old_states[i])[0][0] for i in range(len(old_states)) if new_states[i] in states]
            row_indices = [np.where(states == new_states[i])[0][0] for i in range(len(new_states)) if new_states[i] in states]
            relevant_signs = [[signs[i]] for i in range(len(signs)) if new_states[i] in states]
            
            if len(row_indices) > 0:
                trans_array[row_indices] = np.transpose(phi[:,col_indices])*relevant_signs
                T_ij += np.conj(phi) @ trans_array
        return T_ij

    def vis_oper(self,oper:int):
        '''
        Convert an integer representing an operator to a readable string
        Args:
            oper (int): integer representing an operator
        Returns: 
            (str): string visualizing the operator

        '''
        spins = ['\u2193', '\u2191']
    
        creation = ((1 << self.pos_bits) & oper != 0)
        act_pos = oper & int('1'* self.pos_bits,2)
        act_pos_proper = np.floor(act_pos/2)
        spin_vis = spins[act_pos % 2]
    
        if creation:
            return '$c^{\u2020}_{'+f'{int(act_pos_proper)},{spin_vis}'+'}$'
        else:
            return '$c_{'+f'{int(act_pos_proper)},{spin_vis}'+'}$'

   
    def vis_oper_list(self,oper_list,displ=False):
        '''
        Convert a sequence of operators to a readable string
        '''
        full_str=''
       
        for oper in list(reversed(oper_list)):
            full_str+= self.vis_oper(oper)
        if displ:
            display(Markdown(full_str))
        else:
            return full_str

    

    def vis_state(self,state, displ=False):
        '''
        Show a 'state' in a readable fashion
        args:
            state (int): the Fock state to visualize
            displ (bool): if True, prints the state
                          if False, returns a string
        '''
        n=state
        if isinstance(n,np.ndarray):
            if len(n)==0:
                return '0'
            elif n[0] == -1:
                return '0'
        elif n<0:
            return '0'
        
        str = '\u007C'
        sgn = 1
        count = 0
        while n:
            check_up = n & 1 
            n >>= 1
            check_down = n & 1
            n >>= 1
            if check_up and check_down: 
                str += '\u2193\u2191,'
            elif check_up:
                str += '\u2193,'
            elif check_down:
                str += '\u2191,'
            else:
                str += '0,'
            count += 1
        while count < self.N:
            str += '0,'
            count +=1 
        str = str[:-1]
        str += '\u3009'
        if displ:
            display(Markdown(str))
        else:
            return str

    def vis_state_list(self, states,weights = None, displ=False):
        '''
        Visualise a superposition of Fock states
        Args:
            states (list): list of integers representign states
            weights (floats): the weight of each state
            displ (bool): if True prints the visualisation
                          if False returns the string
        '''
        str = ''
        if weights is not None:
            for w,s in zip(weights,states):
                w = np.round(w,3)
                if w !=0:
                    str += f'{"+" if w>0 else ""}{w.real if w.imag==0 else w}'+self.vis_state(s, displ=False)
        else:
            for state in states:
                str += self.vis_state(state, displ=False) + ' '
        if displ:
            display(Markdown(str))
        else:
            return str

    @staticmethod
    def oper_list_to_str(oper_list):
        '''
        Convert a list of integers representing operators 
        to a string representation
        '''
        return '.'.join(map(str, oper_list))

    @staticmethod
    def oper_str_to_list(oper_str):
        '''
        Convert a string representing operators
        to a list of operators
        '''
        return list(map(int, oper_str.split('.')))

class ParitySystem(FermionSystem):
    def __init__(self, N: int,H_generator: Callable, H_mapping: Callable, sparse_function: Callable=None, Ez_inf: bool = False,U_inf:bool = False):

        '''
        Class for specifically handling systems where a division into odd and even states is useful
        Args:
            N (int): the number of fermionic sites
            H_generator (Callable): a function that creates the list of operators representing H
            H_mapping (Callable): a fuction that assigns readable labels to the operators
            sparse_function (Callable): the function of LinearOperator to use when using scipy's sparse solver
            Ez_inf (bool): whether to restrict the subspaces to infinite Ez 
            U_inf (bool): whether to restrict the subspaces to infinute U
        '''
        super().__init__(N)
        self.odd_states,self.even_states = self.restrict_and_sort_fockspace(Ez_inf = Ez_inf, U_inf=U_inf)
        self.fock_states = np.append(self.even_states,self.odd_states)
        self.odd_states_hash = {num:idx for idx,num in enumerate(self.odd_states)}
        self.even_states_hash = {num:idx for idx,num in enumerate(self.even_states)}

        ## Initialise H formatting dictionary for readability and control purposes
        self.H,self.H_signs = H_generator()
        self.H_vals, self.H_symbols,self.H_types = H_mapping()
        self.sparse_function =sparse_function

        ## Gather Odd Hamiltonian
        self.odd_terms = self.act_H_on_subspace(self.odd_states,self.odd_states_hash)
        self.odd_vals = np.array([self.H_vals[type] for type in self.odd_terms[3]])

        ## Gather Even Hamiltonian
        self.even_terms = self.act_H_on_subspace(self.even_states,self.even_states_hash)
        self.even_vals = np.array([self.H_vals[type] for type in self.even_terms[3]])

    def restrict_and_sort_fockspace(self,Ez_inf: bool, U_inf: bool):
        '''
        Sorts Fock states into 'even' and 'odd' parities (determined by count of 1's in binary)
        Optionally allows restricting the Fock space to infinite U or infinite Ez 
        Args:
            Ez_inf (bool): If True, excludes states with a 'spin up' set
            U_inf (bool): If True, excludes states with both 'spin up' and 'spin down' set for a single site
        Returns:
            odd_states, even_states (nd.array): Nnon-excluded fock states sorted by parity
        '''
        all_states = self.fock_states        
        
        ## If Ez is infinite -> exclude all states with a 1 in the odd powers of 2 position
        if Ez_inf:
            for spin_up_in_state in [2**(2*n+1) for n in range(self.N)]:
                mask_states = ((all_states & spin_up_in_state) == 0)
                all_states = all_states[(mask_states)]
        
        ## If otherwise U is infinite -> exclude all states with both 1's set for a given site
        elif U_inf:
            for spin_up_in_state,spin_down_in_state in zip([2**(2*n)for n in range(self.N)], [2**(2*n+1)for n in range(self.N)]):
                mask_states_odd = ((all_states & spin_up_in_state)!=0)
                mask_states_even = ((all_states & spin_down_in_state)!=0)
                mask_states = ~(mask_states_odd & mask_states_even)
                all_states = all_states[(mask_states)]
    
        ## Create masks for odd and even parity
        even_states_mask = (hamming_weight(all_states)%2 ==0)
        odd_states_mask = ~even_states_mask
    
        ## Seperate the states into odd and even parity
        even_states = all_states[even_states_mask]
        odd_states = all_states[odd_states_mask]
        return odd_states,even_states

    def act_H_on_subspace(self, states: np.ndarray,states_hash: dict):
        '''
        Applies the stored hamiltonian to a set of Fock states
        Args:
            states (ndarray[int]): list of Fock states to act on
            states_hash (dict): dictionary mapping a state to its list index in 'states'
        Returns:
            rows (ndarray[int]): row indices of non-zero terms
            cols (ndarray[int]): column indices of non-zero terms
            pars (ndarray[int]): relative signs of operators for the non-zero terms
            type (ndarray[str]): the types of operators giving rise to non-zero terms
        '''
        rows,cols,pars,type = [],[],[],[] 
        for rel_sign, h in zip(self.H_signs,self.H):
            type_str = self.oper_list_to_str(h)
            old_states,new_states,parities = self.act_oper_list(h, states,rel_sign=rel_sign)
            subspace_filt = [state in states_hash for state in new_states]
            old_states = old_states[subspace_filt]
            new_states = new_states[subspace_filt]
            parities = parities[subspace_filt]
            type.extend([type_str]*len(parities))
            rows.extend([states_hash.get(state) for state in old_states])
            cols.extend([states_hash.get(state) for state in new_states])
            pars.extend(parities.tolist())
        return np.array(rows, dtype=np.int32),np.array(cols,dtype=np.int32),np.array(pars),np.array(type)


    def H_to_array(self, parity):
        '''
        Converts lists of non-zero row/column indices to a 2D array
        Used for smaller systems to do np.linalg.eigh on the Hamiltonian in array form
        Args:
            parity (str): which parity sector to construct the hamiltonian for
        Returns:
            None
        '''
        if parity == 'odd':
            states = self.odd_states
            pos_data = self.odd_terms
            val_data = self.odd_vals
        else:
            states = self.even_states
            pos_data = self.even_terms
            val_data = self.even_vals
        
        arr = np.zeros((len(states),len(states)),dtype=complex)
        idx=0
        for row,col,par in zip(pos_data[0],pos_data[1],pos_data[2]):
            arr[row,col] += par*val_data[idx]
            arr[col,row] += np.conj(par*val_data[idx])
            idx+=1

        if parity == 'odd':
            self.H_odd = arr
        else:
            self.H_even = arr

    def update_H_param(self, type: str, new_val: float, update_matrix=False):
        '''
        Change the value of the hamiltonian parameter
        Args:
            type (str): 'readable' variable name
            new_val (float): the new value to set
            update_matrix (bool): if True, also updates the array forms
                                (requires that the arrays have been generated)
        Returns:
            None
        '''
        ## Diagonal H terms are counted twice, store only halve the value
        ## Note: this is the only non-general part of this Class as it assumes
        ## things about the variable names. 
        ## would be better to get the factor of 1/2 generally for diagonal terms
        ## but checking for diagonality will add some time
        if type[0] == 'm' or type[0] == 'U':
            new_val = new_val/2
            
        ## Grab 'non-verbose' parameter name
        type_key = self.H_types[type] 

        ## Grab the parameters old value
        old_val = self.H_vals[type_key]

        ## Store the new value
        self.H_vals[type_key] = new_val 

        if update_matrix:
            Hs = [self.H_even, self.H_odd]
        
        ## Update the lists tracking the non-zero terms for the even and odd sectors
        i=0
        for terms, values in zip([self.even_terms, self.odd_terms], [self.even_vals, self.odd_vals]):
            type_match = (terms[3] == type_key)
            replace_value_indices = np.where(type_match)[0]
            signs = terms[2][replace_value_indices]
            values[replace_value_indices] = signs*new_val 
            
            ## If required, update the matrix forms
            if update_matrix:
                H=Hs[i]
                for r,c,p in zip(terms[0][type_match],terms[1][type_match],terms[2][type_match]):
                    H[r,c] -= old_val*p
                    H[r,c] += new_val*p
                    H[c,r] -= np.conj(old_val*p)
                    H[c,r] += np.conj(new_val*p)
            i+=1

    def update_H_param_list(self, types: list[str], new_val: float, update_matrix=False):
        '''
        Change the value of multiple Hamiltonian parameters
        Args:
            types (list[str]): 'readable' variable name
            new_val (float): the new value to set
            update_matrix (bool): if True, also updates the array forms
                                (requires that the arrays have been generated)
        Returns:
            None
        '''
        ## Diagonal H terms are counted twice, store only halve the value
        ## Note: this is the only non-general part of this Class as it assumes
        ## things about the variable names. 
        ## would be better to get the factor of 1/2 generally for diagonal terms
        ## but checking for diagonality will add some time

        if types[0][0] == 'm' or types[0][0] == 'U':
            new_val = new_val/2
            
        ## Grab 'non-verbose' parameter name
        type_keys = [self.H_types[type] for type in types]

        ## Grab the parameters old value
        old_val = self.H_vals[type_keys[0]]

        ## Store the new values
        for type_key in type_keys:
            self.H_vals[type_key] = new_val 

        if update_matrix:
            Hs = [self.H_even, self.H_odd]

        ## Update the lists tracking the non-zero terms for the even and odd sectors
        i=0
        for terms, values in zip([self.even_terms, self.odd_terms], [self.even_vals, self.odd_vals]):
            type_match = np.isin(terms[3], type_keys)
            replace_value_indices = np.where(type_match)[0]
            signs = terms[2][replace_value_indices]
            values[replace_value_indices] = signs*new_val 
            
            ## If required, update the matrix forms
            if update_matrix:
                H=Hs[i]
                for r,c,p in zip(terms[0][type_match],terms[1][type_match],terms[2][type_match]):
                    H[r,c] -= old_val*p
                    H[r,c] += new_val*p
                    H[c,r] -= np.conj(old_val*p)
                    H[c,r] += np.conj(new_val*p)
            i+=1
    
    def solve_system(self, method='linalg', n_values=None,):
        '''
            Calculate the eigenvalues and eigenstates of the odd and even parity sectors
            Args:
                method (str): 
                    'linalg': uses full matrix arrays and np.linalg.eigh to calculate values
                    'sparse': uses scipys LinearOperator and the lists of non-zero terms to calculate values
                    The latter becomes faster for larger N, when only requiring a small number of n_values
                n_values (int):
                    Required in combination with method='sparse'-> specifies the number n of smallest eigenvalues
                    and eigenvectors to calculate.
                    method='linalg' will always return the full set of solutions
            Returns:
                E_odd: list of eigenvalues of the odd parity sector
                E_even: list of eigenvalues of the even parity sector
                E_phi_odd: array of normalized odd eigenvector weights
                E_phi_even: array of normalized even eigenvector weights
        '''
        
        if method == 'linalg':
            ## Check if the matrices have been constructed, otherwise generate them
            if not hasattr(self, 'H_odd'):
                self.H_to_array('odd')
            if not hasattr(self, 'H_even'):
                self.H_to_array('even')
            E_odd,E_even,E_phi_odd,E_phi_even = self.solve_linalg()

        elif method == 'sparse':
            ## If no n_values is provided, get the maximum allowed number
            if n_values is None:
                n_values = max([len(self.odd_states-1), len(self.even_states-1)])
            E_odd,E_even,E_phi_odd,E_phi_even = self.solve_sparse(n_values)

        return E_odd,E_even,E_phi_odd,E_phi_even

    def solve_linalg(self):
        '''
        Uses numpy's linalg.eigh to obtain all eigenvalues and eigenvectors
        '''
        E_odd, E_phi_odd = np.linalg.eigh(self.H_odd, UPLO = 'U')
        E_phi_odd = np.transpose(E_phi_odd)
        E_phi_odd = np.round(E_phi_odd,10) ## Truncate numerical errors

        E_even, E_phi_even = np.linalg.eigh(self.H_even,  UPLO = 'U')
        E_phi_even = np.transpose(E_phi_even)
        E_phi_even = np.round(E_phi_even,10)  ## Truncate numerical errors

        ## obtain the ground states of the system (allow degeneracy within some treshhold)
        return E_odd.real,E_even.real, E_phi_odd, E_phi_even

    def solve_sparse(self, n_values):
        '''
        Uses scipy's LinearOperator to calculate the n_values number of lowest eigenvalues and eigenvectors
        '''
        ## Construct LinearOperator for odd parity sector
        M_odd = LinearOperator((len(self.odd_states), len(self.odd_states)), 
                               matvec=partial(self.sparse_function, rows=self.odd_terms[0], cols=self.odd_terms[1],vals=self.odd_vals*self.odd_terms[2]))

        ## Construct LinearOperator for even parity sector
        M_even = LinearOperator((len(self.even_states), len(self.even_states)), 
                            matvec=partial(self.sparse_function, rows=self.even_terms[0], cols=self.even_terms[1],vals=self.even_vals*self.even_terms[2]))

        ## n_values cannot be larger than dimension if system -1
        n_values_odd = min([n_values, len(self.odd_states)-1])
        n_values_even = min([n_values, len(self.even_states)-1])

        ## Obtain n lowest eigenvalues and eigenvectors (up to n_values)
        E_odd, phi_odd = eigsh(M_odd, k=n_values_odd, which='SA')
        E_even, phi_even = eigsh(M_even, k=n_values_even, which='SA')
        return E_odd,E_even, np.transpose(phi_odd),np.transpose(phi_even)

    @staticmethod
    def sel_ground_states(E_odd: list, E_even: list, threshold: float  = 0.001):
        ''' 
        Determine the idx of the lowest ground state, between the odd and even sector
        Selects multiple in case of degeneracies, within the given threshhold
        Args:
            E_odd (list): odd eigenvalues
            E_even (list): even eigenvalues
            thresshold (float): the threshold within which to consider a degeneracy
        Returns:
            odd_gs (list): indexes of odd groundstates
            even_gs (list): indexes of even groundstates
            min_E (float): the lowest eigenvalue
        '''
        min_e_odd = E_odd[np.argmin(E_odd)]
        min_e_even = E_even[np.argmin(E_even)]
       
        min_e = min(min_e_odd,min_e_even)
        even_gs = []
        odd_gs = []
        for idx,e in enumerate(E_odd):
            if np.abs(e-min_e) < threshold:
                odd_gs.append(idx)
        for idx,e in enumerate(E_even):
            if np.abs(e-min_e) < threshold:
                even_gs.append(idx)
        return odd_gs,even_gs, min_e


    def eigenstates(self, only_ground_states=False, only_energies = False):
        '''
        Calculate and display the eigenstates of the system, sorted by energy

        '''
        E_odd,E_even,E_phi_odd, E_phi_even = self.solve_system(method='linalg')
        odd_gs, even_gs,E_lowest = self.sel_ground_states(E_odd,E_even) ## which states to label as ground state

        ## Merge the odd and even states
        merged_list = []
        i=0
        for E,phi,gs in zip([E_odd,E_even], [E_phi_odd, E_phi_even],[odd_gs,even_gs]):
            for idx in np.arange(len(E)):
                if idx in gs:
                    merged_list.append((E[idx],phi[idx], 'ground',i))
                else:
                    merged_list.append((E[idx],phi[idx], 'excited',i))
            i+= 1 
        sorted_merged_list = sorted(merged_list, key=lambda x: x[0])

        ## Print the sorted states
        states_list = [self.odd_states, self.even_states]
        parity_list = ['odd', 'even']
        for state in sorted_merged_list:
            label = f'({state[2]}) ({parity_list[state[3]]})'
            if only_ground_states:
                if state[2] != 'ground':
                    continue
                else:
                    if only_energies:
                        print(f"{label:<20} E = {np.round(state[0],2)}")
                    else:
                        print(f"{label:<20} E = {np.round(state[0],2)}  \u007C\u03C6\u3009= {self.vis_state_list(states_list[state[3]],  state[1])}")
            else:
                if only_energies:
                    print(f"{label:<20} E = {np.round(state[0],2):<10.2f}")
                else:
                    print(f"{label:<20} E = {np.round(state[0],2):<10.2f} \u007C\u03C6\u3009= {self.vis_state_list(states_list[state[3]],  state[1])}")


    def lowest_transitions_sorted(self,site: int, method: str='linalg',n_values: int = 1, threshold: float = 1):
        ''' 
        Set-up for calculating possible single-electron transitions between odd/even ground states and the excited states
        for adding holes/electrons to a given site
        Included for speed compared to solving the entire rate equation

        Args:
            site: the fermionic site to calculate transitions for
            method: method to use for solving the hamiltonian
            n_values: number of lowest eigenvalues to obtain, only relevant if method='sparse'
            thresshold: range within which to consider a groundstate degenerate
        '''
        E_odd,E_even, phi_odd,phi_even = self.solve_system(method=method,n_values=n_values)
        odd_gs_idx, even_gs_idx,min_E = self.sel_ground_states(E_odd,E_even, threshold=threshold)
        T_all = []
        weights_all = []
        for idx in odd_gs_idx:
            T_odd_to_even, weights = self.lowest_transitions(E_odd[idx],phi_odd[idx], self.odd_states, E_even, phi_even, self.even_states, site)
            T_all.extend(T_odd_to_even)
            weights_all.extend(weights)
            
        for idx in even_gs_idx:
            T_even_to_odd, weights = self.lowest_transitions(E_even[idx],phi_even[idx], self.even_states, E_odd, phi_odd, self.odd_states, site)
            T_all.extend(T_even_to_odd)
            weights_all.extend(weights)
        return T_all,weights_all

    
    def lowest_transitions(self, E_gs, phi_gs, gs_states, E_excited, phi_excited, excited_states, site):
        '''
        Calculate single-electron transitions at a given site, from the ground states to the excited states
        Included for speed compared to solving the entire rate equation
        Args:
            E_gs (float): energy of the ground state
            phi_gs (1Darray): weights of the ground state Fock basis
            gs_states (1Darray): the Fock basis of ground state sector
            E_excited (1Darray): energies of the excited states
            phi_excited (2Darray): array of weights of excited states Fock basis
            excited_states (1Darray):  the fock basis of the excited state sector
            site (int): site to consider for single electron transitions
        '''
        gs_filt = (phi_gs != 0)
        gs_states = gs_states[gs_filt]
        phi_gs = phi_gs[gs_filt]

        excited_states_array = np.array([excited_states for _ in range(len(phi_excited))])
        es_shape = np.shape(phi_excited)
        phi_excited = np.conj(phi_excited)

        positive = [self.operator('creation', site,'up'), self.operator('creation',site,'down')]
        negative = [self.operator('annihilation', site, 'up'), self.operator('annihilation',site,'down')]

        energies = []
        weights = []
        
        i=0
        for sign,oper in zip([1,1,-1,-1],positive+negative):
            new_states,parities = self.act_oper(oper, gs_states)
            transitions = np.full(es_shape,0.0)
            for state, gs_weight, parity in zip(new_states, phi_gs, parities):
                if state>-1:
                    state_match = (excited_states_array == state)
                    transitions = state_match*phi_excited*gs_weight*parity + transitions

            total = np.sum(transitions, axis=1)
            energies.extend([sign*(E_gs-E_excited[idx]) for idx,p in enumerate(total) if p!=0])
            weights.extend([p for p in total if p!=0])
        return energies,weights

    def show_hamiltonian_numeric(self, parity, width):
        '''
        Show the filled Hamiltonian matrix
        '''
        if parity=='odd':
            if not hasattr(self, 'H_odd'):
                self.H_to_array('odd')
            H = self.H_odd
        else:
            if not hasattr(self, 'H_even'):
                self.H_to_array('even')
            H = self.H_even
        #H = self.H_to_array(parity)
        fig,ax = plt.subplots(1,figsize =(width/1.2,width/1.2))
        ax.set_xlim([0,width])
        ax.set_ylim([width,0])
        for i in np.arange(width):
            for j in np.arange(width):
                t = H[j][i]
                t = np.round(complex(t),2)
                if t.imag == 0:
                    t = t.real
                ax.text(i,j,t, horizontalalignment = 'center')
        ax.axis('off')
        
    def show_hamiltonian(self,parity, numeric=False):
        '''
        Show the symbolic Hamiltonian matrix
        To do: add the symbols on the diagonal
        '''
        if parity == 'odd':
            states = self.odd_states
            pos_data = self.odd_terms
            val_data = self.odd_vals
        else:
            states = self.even_states
            pos_data = self.even_terms
            val_data = self.even_vals

        strs = ''
        for state in states:
            strs+= self.vis_state(state) + ', '
        print(f'Basis: {strs[:-2]}')
        if numeric:
            self.show_hamiltonian_numeric(parity, len(states))
            return
        width = len(states)
        fig,ax = plt.subplots(1,figsize =(width/1.2,width/1.2))
        ax.set_xlim([0,width])
        ax.set_ylim([width,0])

        ## Fill in known non-diagonal terms
        for idx, r in enumerate(pos_data[0]):
            i = r
            j = pos_data[1][idx]
            if i==j:
                continue
            sign = pos_data[2][idx]
            type = pos_data[3][idx]
            if sign == 1:
                t_print =  self.H_symbols[f'{type}']
            else:
                t_print = '-' +  self.H_symbols[f'{type}']
            ax.text(i,j,t_print, horizontalalignment = 'center')
            ax.text(j,i,t_print, horizontalalignment = 'center')

        ## list of row and column pairs
        rc_pairs = [[r,c] for r,c in zip(pos_data[0],pos_data[1]) if r!=c]
        
        ## Fill in remaining zeroes
        for i in range(len(states)):
            for j in range(i,len(states)):
                if [i,j] in rc_pairs:
                    continue
                else:
                    ax.text(i,j, '0', horizontalalignment = 'center')
                    if i !=j:
                        ax.text(j,i, '0', horizontalalignment = 'center')
    
        ax.axis('off')

    '''
    ### Old version, slightly faster implementation added below
    def rate_equation_old(self, sites,bias_range, lead_params, truncate_lim = 20, method = 'linalg'):
        ## Solve for energies and wavefunctions
        E_odd, E_even, phi_odd, phi_even = self.solve_system(method=method, n_values=truncate_lim)

        ## Merge the odd and even sections into a block diagonal matrix with only the lowest eigenvectors and eigenvalues
        E_even_trunc, E_odd_trunc = self.N_lowest_states(E_even,E_odd, truncate_lim=truncate_lim)
        E = np.append(E_even[:E_even_trunc], E_odd[:E_odd_trunc])
        phi = block_diag(phi_even[:E_even_trunc], phi_odd[:E_odd_trunc])

        G_matrix = np.zeros((len(sites), len(sites), len(bias_range)))

        Es_a, Es_b = np.meshgrid(E, E)
        Es_ba = Es_b - Es_a
        Tsq_plus_list, Tsq_minus_list = [], []
        ## For each desired site, get transition rate matrix
        for site in sites:
            operators = [self.operator('creation',site,'up'), self.operator('creation',site,'down')] ## Create spin-up and spin-down
            Tsq_plus = np.abs(self.bra_oper_ket(self.fock_states,phi, operators))**2 
            Tsq_minus = Tsq_plus.T 
            Tsq_plus_list.append(Tsq_plus)
            Tsq_minus_list.append(Tsq_minus)

        ## Generate currents from transition rate matrix
        for j in range(len(sites)):
            for i, Vbias in enumerate(bias_range):
                mus = np.zeros(len(sites))
                mus[j] = Vbias - lead_params['dV']
                Is0 = get_Is(len(sites), Tsq_plus_list, Tsq_minus_list, lead_params['gammas'], mus, Es_ba, lead_params['kBT'])
               
                mus = np.zeros(len(sites))
                mus[j] = Vbias + lead_params['dV']
                Is1 = get_Is(len(sites), Tsq_plus_list, Tsq_minus_list, lead_params['gammas'], mus, Es_ba, lead_params['kBT'])
                
                gs = 2 * np.pi * (Is1 - Is0) / (2*lead_params['dV'])
                G_matrix[:, j, i] = gs
                
        return G_matrix
    '''

    def rate_equation(self, sites,bias_range, lead_params, truncate_lim = 100, method = 'linalg'):
        ## Solve for energies and wavefunctions
        E_odd, E_even, phi_odd, phi_even = self.solve_system(method=method, n_values=truncate_lim)

        ## Merge the odd and even sections into a block diagonal matrix with only the lowest eigenvectors and eigenvalues
        E_even_trunc, E_odd_trunc = self.N_lowest_states(E_even,E_odd, truncate_lim=truncate_lim)
        E = np.append(E_even[:E_even_trunc], E_odd[:E_odd_trunc])
        phi = block_diag(phi_even[:E_even_trunc], phi_odd[:E_odd_trunc])

        G_matrix = np.zeros((len(sites), len(sites), len(bias_range)))

        Es_a, Es_b = np.meshgrid(E, E)
        Es_ba = Es_b - Es_a
        Es_ab = -Es_ba
        Tsq_plus_list, Tsq_minus_list = [], []
        ## For each desired site, get transition rate matrix
        for site in sites:
            operators = [[self.operator('creation',site,'up')], [self.operator('creation',site,'down')]] ## Create spin-up and spin-down

            Tsq_plus = np.abs(self.bra_oper_ket(self.fock_states, phi, operators))**2 
            Tsq_minus = Tsq_plus.T 
            Tsq_plus_list.append(Tsq_plus)
            Tsq_minus_list.append(Tsq_minus)

        ## These terms needed for solving rate equation are constant 
        kBT = lead_params['kBT']
        Nf_0_plus = n_F(Es_ba, 0, kBT)
        Nf_0_min = np.ones(np.shape(Es_ba)) - n_F(Es_ab, 0, kBT)

        ## Generate currents from transition rate matrix
        for i, Vbias in enumerate(bias_range):
            ## These terms are specific to each Vbias
            mu_minus = Vbias - lead_params['dV']
            Nf_mu_plus_minus = n_F(Es_ba, mu_minus, kBT)
            Nf_mu_min_minus = np.ones(np.shape(Es_ba)) - n_F(Es_ab, mu_minus, kBT)

            mu_plus = Vbias + lead_params['dV']
            Nf_mu_plus_plus = n_F(Es_ba, mu_plus, kBT)
            Nf_mu_min_plus = np.ones(np.shape(Es_ba)) - n_F(Es_ab, mu_plus, kBT)

            for j in range(len(sites)):
                mus = np.zeros(len(sites))
                mus[j] = mu_minus
                Is0 = get_Is_vectorized(len(sites), Tsq_plus_list, Tsq_minus_list, lead_params['gammas'], mus, Nf_0_plus,Nf_0_min,Nf_mu_plus_minus,Nf_mu_min_minus)
                
                mus = np.zeros(len(sites))
                mus[j] = mu_plus
                Is1 = get_Is_vectorized(len(sites), Tsq_plus_list, Tsq_minus_list, lead_params['gammas'], mus, Nf_0_plus,Nf_0_min, Nf_mu_plus_plus,Nf_mu_min_plus)
                
                gs = 2 * np.pi * (Is1 - Is0) / (2*lead_params['dV'])
                G_matrix[:, j, i] = gs
        return G_matrix

    @staticmethod
    def N_lowest_states(E_even,E_odd, truncate_lim):
        sorted_E = np.sort(np.append(E_odd,E_even))
        trunc = min([truncate_lim, len(E_odd)+len(E_even)])
        lowest_E_vals = (sorted_E[:trunc])[::-1]

        E_odd_trunc = np.where(E_odd == lowest_E_vals[np.argmax(np.isin(lowest_E_vals,E_odd))])[0][-1]+1
        E_even_trunc =  np.where(E_even == lowest_E_vals[np.argmax(np.isin(lowest_E_vals,E_even))])[0][-1]+1
        return E_even_trunc, E_odd_trunc