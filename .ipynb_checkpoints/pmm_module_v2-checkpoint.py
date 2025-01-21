# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import numpy.linalg as LA
import itertools


def get_h_2(c_labels, c_signs, strength, bases, index_from_basis):
    """
    When c_labels = [i, j] and c_signs = [+1, -1], it means H = strength * c^dg_i c_j (+ H.c. if needed).
    """
    dim = len(bases)
    H = np.zeros((dim, dim), dtype=complex)
    if c_labels[0] > c_labels[1]:  # we first apply c operators with a larger c_label
        cflip = 1
    else:
        cflip = 0

    for n in range(dim):
        restart = False
        num_of_anticom = 0
        vec = list(bases[n])
        num_of_anticom += np.sum(vec[0:c_labels[0]]) + np.sum(vec[0:c_labels[1]])
        
        vec[c_labels[1]] += c_signs[1]
        if any(k < 0 for k in vec) or any(k > 1 for k in vec):
            restart = True
        if restart:
            continue
            
        vec[c_labels[0]] += c_signs[0]
        if any(k < 0 for k in vec) or any(k > 1 for k in vec):
            restart = True
        if restart:
            continue

        m = index_from_basis[tuple(vec)]
        H[m, n] = strength * (-1)**(num_of_anticom + cflip)
        if abs(m - n) > 1e-5:
            H[n, m] = H[m, n].conj()
    return H


def get_h_coulomb(n_labels, strength, bases, index_from_basis):
    """
    We only consider coulomb terms like U * n_i * n_j
    When n_labels = [i, j], it means H = strength * n_i * n_j.
    """
    dim = len(bases)
    H = np.zeros((dim, dim), dtype=complex)
    for n in range(len(bases)):
        vec = list(bases[n])
        n_i, n_j = vec[n_labels[0]], vec[n_labels[1]]
        H[n, n] = strength * n_i * n_j
    return H


def get_c_matrix(c_labels, c_signs, bases, index_from_basis):
    """
    When c_labels = [i] and c_signs = [+1], it means <basis|c^dg_i|basis> (no h.c.).
    When c_labels = [j] and c_signs = [-1], it means <basis|c_j|basis> (no h.c.).
    """
    dim = len(bases)
    c_matrix = np.zeros((dim, dim), dtype=complex)
    
    for n in range(dim):
        restart = False
        num_of_anticom = 0
        vec = list(bases[n])
        num_of_anticom += np.sum(vec[0:c_labels[0]])
        
        vec[c_labels[0]] += c_signs[0]
        if any(k < 0 for k in vec) or any(k > 1 for k in vec):
            restart = True
        if restart:
            continue

        m = index_from_basis[tuple(vec)]
        c_matrix[m, n] = (-1)**(num_of_anticom)
    return c_matrix


def sort_parity_states(es, wfs):
    """
    Even-parity space comes first.
    """
    N_even = len(wfs[:, 0])//2
    es_even, es_odd = [], []
    wfs_even, wfs_odd = [], []
    
    for n in range(len(es)):
        if np.sum(np.abs(wfs[0:N_even, n])**2) > 0.99:
            es_even.append(es[n])
            wfs_even.append(wfs[:,n])
        else:
            es_odd.append(es[n])
            wfs_odd.append(wfs[:,n])
    return es_even, wfs_even, es_odd, wfs_odd





def get_d_dg(orbitals, bases, index_from_basis):
    d_dg_list = []
    for n in range(len(orbitals)):
        d_dg = []
        for i in orbitals[n]:
            d_dg.append(get_c_matrix(c_labels=[i], c_signs=[+1], bases=bases, index_from_basis=index_from_basis))
        d_dg_list.append(d_dg)
    return d_dg_list


def n_F(E, mu, kBT):
    energy = (E - mu) / kBT
    n = (np.exp(energy) + 1)**-1
    return n


def get_P(rate_total, N_state):
    rate_matrix = np.zeros((N_state+1, N_state))
    rate_matrix[0:N_state, 0:N_state] = rate_total
    for k in range(N_state):
        rate_matrix[k, k] = -np.sum(rate_total[:, k])
    rate_matrix[N_state, :] = np.ones(N_state)
    
    right_vec = np.zeros((N_state+1, 1))
    right_vec[N_state, 0] = 1
    P_vec = LA.pinv(rate_matrix) @ right_vec
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
    P_vec = get_P(rate_total=rate_total, N_state=np.shape(Es_ba)[0])
    Is = get_current(rate_plus_list, rate_minus_list, P_vec, num_of_leads)
    return Is


def get_G_matrix(es, wfs, d_dg_list, bias_range, lead_params):
    '''
    inputs:
    es, wfs: eigenenergies and eigenfunctions of the isolated system.
    d_dg_list: matrices of d_dg coupled to the leads
    bias_range: 1D array/list for the range of bias voltage to scan. len(bias_range) should be >= 1.
    lead_params: a dictionary including num_of_leads, gammas, orbitals, kBT, dV
    outputs: G_matrix, with shape of num_of_leads x num_of_leads x len(bias_range)
    '''
    num_of_leads = lead_params['num_of_leads']
    gammas = lead_params['gammas']
    orbitals = lead_params['orbitals']
    kBT = lead_params['kBT']
    dV = lead_params['dV']
    
    G_matrix = np.zeros((num_of_leads, num_of_leads, len(bias_range)))

    ### the following part has no Vbias dependence
    Es_a, Es_b = np.meshgrid(es, es)
    Es_ba = Es_b - Es_a

    Tsq_plus_list = []
    Tsq_minus_list = []
    for j in range(num_of_leads):
        Tsq_plus = 0
        for i in range(len(orbitals[j])):
            Tsq_plus += np.abs(wfs.T.conj() @ d_dg_list[j][i] @ wfs)**2
        Tsq_minus = Tsq_plus.T
        Tsq_plus_list.append(Tsq_plus)
        Tsq_minus_list.append(Tsq_minus)

    ### the following part has both lead and Vbias dependence
    for j in range(num_of_leads):
        for i, Vbias in enumerate(bias_range):
            mus = np.zeros(num_of_leads)
            mus[j] = Vbias
            Is0 = get_Is(num_of_leads, Tsq_plus_list, Tsq_minus_list, gammas, mus, Es_ba, kBT)

            mus = np.zeros(num_of_leads)
            mus[j] = Vbias + dV
            Is1 = get_Is(num_of_leads, Tsq_plus_list, Tsq_minus_list, gammas, mus, Es_ba, kBT)

            gs = 2 * np.pi * (Is1 - Is0) / dV
            G_matrix[:, j, i] = gs
            
    return G_matrix


# +
def H_bdg_kitaev_N(N, mus, ts, Deltas, phis):

    assert N >= 2, "N must be larger than 2."
    assert len(mus) == N, "len(mus) is not equal to N."
    assert len(ts) == N-1, "len(ts) is not equal to N-1."
    assert len(Deltas) == N-1, "len(Deltas) is not equal to N-1."
    assert len(phis) == N-1, "len(phis) is not equal to N-1."
    
    H_bdg = np.zeros((2*N, 2*N), dtype=complex)
    H_normal = np.zeros((N, N), dtype=complex)
    H_Delta = np.zeros((N, N), dtype=complex)

    for j in range(N):
        H_normal[j, j] = mus[j]

    for j in range(N-1):
        H_normal[j, j+1] = ts[j]
        H_normal[j+1, j] = ts[j]

        H_Delta[j, j+1] = -Deltas[j] * np.exp(1j * phis[j])
        H_Delta[j+1, j] = Deltas[j] * np.exp(1j * phis[j])
        

    H_bdg[0:N, 0:N] = H_normal
    H_bdg[N:, N:] = -H_normal.conj()
    H_bdg[0:N, N:] = H_Delta
    H_bdg[N:, 0:N] = H_Delta.T.conj()
    return H_bdg

def get_S_matrix(omega, H_bdg, W, N):
    W_dg = W.conj().T
    GF = LA.inv(omega * np.eye(2*N) - H_bdg + 0.5 * 1j * W @ W_dg)
    S_matrix = np.eye(2*N) - 1j * W_dg @ GF @ W
    return S_matrix

def get_G0_matrix(S_matrix, N):
    
    G0_matrix = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            if i == j:
                same_lead = 1
            else:
                same_lead = 0
            G0_matrix[j, i] = same_lead - abs(S_matrix[j, i])**2 + abs(S_matrix[j+N, i])**2
    return G0_matrix

def get_GT(G0, omega_range, T):
    domega = omega_range[1] - omega_range[0]
    df_dE = (4 * T)**-1 * np.cosh(omega_range/(2*T))**-2
    GT = np.convolve(a=G0, v=df_dE, mode='same') * domega
    return GT



















