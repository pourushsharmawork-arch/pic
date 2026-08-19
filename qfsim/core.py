"""Core functionality of the package.
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import least_squares
np.set_printoptions(precision=4, suppress=True)

from . import one_two_hardware_decomposition
from . import two_one_hardware_decomposition
from . import misc

##################################################################
# Optical Unitary Scattering Matrices
##################################################################
# Beam-splitter and Phase-shifter
def U_PS(α, β):
    """
    Two-mode phase shifter.
    """
    return np.array([
        [np.exp(1j * α), 0],
        [0, np.exp(1j * β)]
    ], dtype=complex)




def U_BS(θ, ϕ=0):
    """
    Beam splitter with i-reflection convention:
        [[cos theta, i sin theta],
         [i sin theta, cos theta]]
    """
    return np.array([
        [np.cos(θ), 1j * np.exp(-1j*ϕ) * np.sin(θ)],
        [1j * np.exp(1j*ϕ) * np.sin(θ), np.cos(θ)]
    ], dtype=complex)


def U2mmi(ξ = 0.5):
    return np.array([
        [np.sqrt(ξ), 1j * np.sqrt(1-ξ)],
        [1j * np.sqrt(1-ξ), np.sqrt(ξ)]
    ])

def U2mzi(θ, ϕ,  ξlist=[0.5, 0.5]):

    return U_PS(ϕ, 0) @ U2mmi(ξlist[0]) @ U_PS(θ, 0) @ U2mmi(ξlist[1]) 

def loss_matrix(η1, η2):
    """
    Diagonal amplitude-loss matrix.
    eta1^2, eta2^2 are intensity efficiencies.
    """
    if not (0 <= η1 <= 1 and 0 <= η2 <= 1):
        raise ValueError("Efficiencies eta1 and eta2 must lie in [0,1].")

    return np.diag([η1, η2]).astype(float)


def lossy_U2mzi(θ, ϕ, ηlist, ξlist):
    """
    Effective non-unitary 2x2 transfer matrix with input and output losses.
    """
    U = U2mzi(θ, ϕ, ξlist)

    L_in = loss_matrix(ηlist[0], ηlist[1])
    L_out = loss_matrix(ηlist[2], ηlist[3])

    return L_in @ U @ L_out

###################################################################################
# U(4) Decomposition To Quanfluence MZI Architecture
###################################################################################
def decompose_U4_rectangular(
    U: np.ndarray,
    *,
    tol: float = 1e-12,
    decomp : str = '1212',
    max_search_nodes: int = 100_000,
    numerical_fallback: bool = False,
    n_restarts: int = 16,
    seed: int | None = 0,
    return_diagnostics: bool = False 
):
    """
    Decompose a 4x4 unitary into a '1212' decomposition 
    or '2121' decomposition.
    Returns:
        blocks : list of block dictionaries
        D      : final diagonal phase matrix
        W      : diagonalized matrix after applying adjoints
    """


    if decomp == '1212' :
        blocks, D, W = one_two_hardware_decomposition.decompose_U4_rectangular(U = U,
                                            tol = tol,
                                            max_search_nodes = max_search_nodes,
                                            numerical_fallback = numerical_fallback,
                                            n_restarts = n_restarts,
                                            seed = seed,
                                            return_diagnostics = return_diagnostics) 

    elif decomp == "2121" :
        blocks, D, W = two_one_hardware_decomposition.decompose_U4_rectangular(U = U,
                                                    tol = tol,
                                                    max_search_nodes = max_search_nodes,
                                                    numerical_fallback = numerical_fallback,
                                                    n_restarts = n_restarts,
                                                    seed = seed,
                                                    return_diagnostics = return_diagnostics) 

    else :
        raise ValueError("Decomposition entered is not feasible.")
    
    return blocks, D, W
###########################################################

def reconstruct_from_rectangular(blocks, D):
    """
    Reconstructs the complete unitary back.
    """
    N = D.shape[0]
    U_rec = np.eye(N, dtype=complex)

    for block in blocks:
        U_rec = U_rec @ block["T_full"]

    U_rec = U_rec @ D

    return U_rec


def decomposition_diagnostics(U, blocks, D, W):
    """
    Check reconstruction and diagonalization errors.
    """
    U_rec = reconstruct_from_rectangular(blocks, D)

    reconstruction_error = np.linalg.norm(U - U_rec, ord="fro")
    offdiag_error = np.linalg.norm(W - np.diag(np.diag(W)), ord="fro")
    D_unitarity_error = np.linalg.norm(D.conj().T @ D - np.eye(4), ord="fro")

    return {
        "reconstruction_error": reconstruction_error,
        "offdiag_error": offdiag_error,
        "D_unitarity_error": D_unitarity_error,
        "success": reconstruction_error < 1e-10
    }


def print_decomposition_diagnostics(U_target, decomp='1212'):
    blocks, D, W = decompose_U4_rectangular(U_target, decomp=decomp)
    U_reconstructed = reconstruct_from_rectangular(blocks, D)
    
    diagnostics = decomposition_diagnostics(U_target, blocks, D, W)

    print("Block parameters:")
    for block in blocks:
        print(
            f"{block['label']}, "
            f"theta = {block['theta']:.12f}, "
            f"phi = {block['phi']:.12f}"
        )

    print("\nFinal diagonal phase matrix D:")
    print(D)

    print("\nDiagnostics:")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")
    
    print("\nReconstruction check:")
    print(np.allclose(U_target, U_reconstructed, atol=1e-10))

    return None


###################################################################################
# Decompose U(4) to Quanfluence architecture, put losses in each MZI and 
# then find the effective matrix of Quanfluence architecture.  
###################################################################################
def lossy_U4_circuit(U, losses, mmi_imbalance, decomp='1212'):

    blocks, D, W = decompose_U4_rectangular(U, decomp=decomp)

    D_list = np.angle(np.diag(D))
    theta_list = []
    phi_list = []

    for mzi in blocks:
        theta_list.append(mzi["theta"])
        phi_list.append(mzi["phi"])

    x = np.concatenate([theta_list, phi_list, D_list])

    M = build_u4_row_mesh(x, losses, mmi_imbalance, decomp)

    return M


###################################################################################
# Calibrating the 6-MZIs (angles) to find the ideal matrix that can be implemented 
# by the Quanfluence architecture. 
###################################################################################
def frob_innerproduct(A, B):
    if A.shape != B.shape:
        raise ValueError("Shapes of input matrices not equal.")

    return np.trace(A.conj().T @ B)


def frob_norm(A, B):
    return np.sqrt(
        frob_innerproduct(A, B)
    )

def best_scalar(M, target):
    """
    Return c minimizing ||M - c target||_F.
    """
    if misc.is_unitary(target) == False:
        print(f"**Warning** : The target \n {target}, \n is not unitary.")
        
    return np.vdot(target, M) / np.vdot(target, target)



def build_u4_row_mesh(x, losses, mmi_imbalance, decomp='1212'):
    """
    Build the 4x4 row-action transfer matrix of the rectangular mesh.

    Row-vector convention:

        E_out = E_in @ M.

    Mesh order is decided by the decomposition.

    With loss, each two-mode block is replaced by

        L_in U_cell L_out

    because the input is a row vector.
    """
    eta = losses
    xi = mmi_imbalance
    if eta.shape != (6, 4):
        raise ValueError(f"Shape of losses is {eta.shape} not (6, 4)")

    if xi.shape != (6, 2):
        raise ValueError(f"Shape of rotation losses is {xi.shape} not (12, 2)")
    
    if x.shape != (16, ):
        raise ValueError(f"Shape of parameters is {x.shape} not (16, )")

    theta_list = x[0:6]
    phi_list = x[6:12]
    D_list = x[12:16]
            
    D = np.diag(np.exp(1j * D_list))

    if decomp=='1212':
        L1 = block_diag(
            np.eye(1),
            lossy_U2mzi(θ=theta_list[0], ϕ=phi_list[0], ηlist=eta[0], ξlist=xi[0]),
            np.eye(1),
        )

        L2 = block_diag(
            lossy_U2mzi(θ=theta_list[1], ϕ=phi_list[1], ηlist=eta[1], ξlist=xi[1]),
            lossy_U2mzi(θ=theta_list[2], ϕ=phi_list[2], ηlist=eta[2], ξlist=xi[2]),
        )

        L3 = block_diag(
            np.eye(1),
            lossy_U2mzi(θ=theta_list[3], ϕ=phi_list[3], ηlist=eta[3], ξlist=xi[3]),
            np.eye(1),
        )

        L4 = block_diag(
            lossy_U2mzi(θ=theta_list[4], ϕ=phi_list[4], ηlist=eta[4], ξlist=xi[4]),
            lossy_U2mzi(θ=theta_list[5], ϕ=phi_list[5], ηlist=eta[5], ξlist=xi[5]),
        )

        M = L1 @ L2 @ L3 @ L4 

    elif decomp == '2121':
        L1 = block_diag(
                    lossy_U2mzi(θ=theta_list[0], ϕ=phi_list[0], ηlist=eta[0], ξlist=xi[0]),
                    lossy_U2mzi(θ=theta_list[1], ϕ=phi_list[1], ηlist=eta[1], ξlist=xi[1]),
                )
        
        L2 = block_diag(
            np.eye(1),
            lossy_U2mzi(θ=theta_list[2], ϕ=phi_list[2], ηlist=eta[2], ξlist=xi[2]),
            np.eye(1),
        )

        L3 = block_diag(
                    lossy_U2mzi(θ=theta_list[3], ϕ=phi_list[3], ηlist=eta[3], ξlist=xi[3]),
                    lossy_U2mzi(θ=theta_list[4], ϕ=phi_list[4], ηlist=eta[4], ξlist=xi[4]),
                )
        

        L4 = block_diag(
            np.eye(1),
            lossy_U2mzi(θ=theta_list[5], ϕ=phi_list[5], ηlist=eta[5], ξlist=xi[5]),
            np.eye(1),
        )

        M = L1 @ L2 @ L3 @ L4 

    else:
        raise ValueError("Decomposition entered is not feasible.")
    
    return M @ D


def transfer_metrics(M, target):
    """
    Compare a lossy transfer matrix M to a target unitary.

    The relevant question is often whether M is proportional to target:

        M approximately (best_scalar * target).
    """
    c = best_scalar(M, target)
    shape_error = frob_norm(M - c * target, M - c * target) / frob_norm(target, target)

    fidelity = (abs(frob_innerproduct(target, M)) ** 2) / (
        frob_innerproduct(target, target).real * frob_innerproduct(M, M).real
    )

    singular_values = np.linalg.svd(M, compute_uv=False)
    throughput = frob_innerproduct(M, M).real / M.shape[0]

    if throughput > 1e-15:
        nonunitarity = frob_norm(
            M.conj().T @ M - throughput * np.eye(M.shape[0]), 
            M.conj().T @ M - throughput * np.eye(M.shape[0])
        ) / (throughput * np.sqrt(M.shape[0]))
    else:
        nonunitarity = np.inf

    return {
        "best_scalar": c,
        "abs_best_scalar": abs(c),
        "shape_error": shape_error,
        "coherent_fidelity": fidelity,
        "throughput": throughput,
        "singular_values": singular_values,
        "nonunitarity": nonunitarity,
    }


# ---------------------------------------------------------------------
# Final Calibration
# ---------------------------------------------------------------------
def residual_vector(x, target, losses, mmi_imbalance, decomp='1212'):
    """
    Real residual vector for calibration.

    If remove_global_loss=True, minimize ||M - c target||.
    This tells whether the lossy device implements the right unitary shape
    up to an overall attenuation.
    """
    M = build_u4_row_mesh(x, losses, mmi_imbalance, decomp=decomp)
    
    
    c = best_scalar(M, target)
    R = M - c * target
   
    return np.concatenate([R.real.reshape(-1), R.imag.reshape(-1)])


def optimise(target, losses, mmi_imbalance, decomp='1212', max_nfev=5000):
    """
    Tune MZI parameters so that the lossy mesh approximates the target.
    """

    blocks, D, W = decompose_U4_rectangular(target, decomp=decomp)

    p0 = []
    theta_list = []
    phi_list = []

    for mzi in blocks:
        theta_list.append(mzi["theta"])
        phi_list.append(mzi["phi"])

    p0 = np.concatenate([p0, theta_list, phi_list])
    p0 = np.concatenate([p0, np.angle(np.diag(D))])
    
    result = least_squares(
        lambda p: residual_vector(p, target, 
                                  losses=losses, mmi_imbalance=mmi_imbalance,
                                    decomp=decomp),
        p0,
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    
    M_opt = build_u4_row_mesh(result.x, 
                              losses=losses, mmi_imbalance=mmi_imbalance,
                                decomp=decomp)
    
    return {
        "p_opt": result.x,
        "M_opt": M_opt,
        "optimizer_result": result,
        "metrics": transfer_metrics(M_opt, target),
    }



###################################################################################
# SENSITIVITY ANALYSIS : Once, optimized which of 
# the MZI parameters are most sensitive to losses.
###################################################################################
def finite_difference_jacobian(f, x, step=1e-6):
    """Central finite-difference Jacobian of a real vector function."""
    x = np.asarray(x, dtype=float)
    f0 = f(x)
    J = np.zeros((len(f0), len(x)), dtype=float)

    for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = step
        J[:, k] = (f(x + dx) - f(x - dx)) / (2 * step)

    return J


def parameter_names(decomp):

    if decomp == '2121' :
        CELL_NAMES = ["Ta12", "Ta34", "Ta23", "Tb12", "Tb34", "Tb23"]
    elif decomp == '1212':
        CELL_NAMES = ["Ga23", "Ga12", "Ga34", "Gb23", "Gb12", "Gb34"]
    else:
        raise ValueError("Decomposition entered not feasible.")
    
    names = []
    for name in CELL_NAMES:
        names += [f"{name}.theta", f"{name}.phi"]
    names += ["D.xi1", "D.xi2", "D.xi3", "D.xi4"]
    return names


def loss_names(decomp):

    if decomp == '2121':
        CELL_NAMES = ["Ta12", "Ta34", "Ta23", "Tb12", "Tb34", "Tb23"]
    elif decomp == '1212':
        CELL_NAMES = ["Ga23", "Ga12", "Ga34", "Gb23", "Gb12", "Gb34"]
    else:
        raise ValueError("Decomposition entered not feasible.")
    
    names = []
    for name in CELL_NAMES:
        names += [
            f"{name}.eta_in_1",
            f"{name}.eta_in_2",
            f"{name}.eta_out_1",
            f"{name}.eta_out_2",
        ]
    return names

def mmi_imbalance_names(decomp):

    if decomp == '2121':
        CELL_NAMES = ["Ta12", "Ta34", "Ta23", "Tb12", "Tb34", "Tb23"]
    elif decomp == '1212':
        CELL_NAMES = ["Ga23", "Ga12", "Ga34", "Gb23", "Gb12", "Gb34"]
    else:
        raise ValueError("Decomposition entered not feasible.")
    
    names = []
    for name in CELL_NAMES:
        names += [
            f"{name}.xi_out",
            f"{name}.xi_in"
        ]
    return names

def sensitivity(target, p, losses, mmi_imbalance, decomp, step_p=1e-6, step_eta=1e-5, step_xi=1e-5):
    """
    First-order sensitivity of the residual to parameters and losses.

    Returns two Jacobians:

        J_p    = d residual / d MZI-parameter
        J_eta  = d residual / d amplitude transmissivity
        J_xi   = d residual / d mmi imbalance
    The residual is M - c target, so global attenuation is ignored.
    """
    losses = np.asarray(losses, dtype=float)

    def f_p(xx):
        return residual_vector(xx, target, losses, mmi_imbalance, decomp)

    def f_eta(eta_flat):
        eta = np.clip(eta_flat.reshape(6, 4), 1e-12, 1.0)
        return residual_vector(p, target, eta, mmi_imbalance, decomp)

    def f_xi(xi_flat):
        xi = np.clip(xi_flat.reshape(6, 2), 1e-12, 1.0)
        return residual_vector(p, target, losses, xi, decomp)


    J_p = finite_difference_jacobian(f_p, p, step=step_p)
    J_eta = finite_difference_jacobian(f_eta, losses.reshape(-1), step=step_eta)
    J_xi = finite_difference_jacobian(f_xi, mmi_imbalance.reshape(-1), step=step_xi)

    param_norms = np.linalg.norm(J_p, axis=0)
    loss_norms = np.linalg.norm(J_eta, axis=0)
    mmi_imbalance_norms = np.linalg.norm(J_xi, axis=0)
    svals = np.linalg.svd(J_p, compute_uv=False)

    return {
        "J_parameters": J_p,
        "J_losses": J_eta,
        "J_mmi_imbalance" : J_xi,
        "parameter_singular_values": svals,
        "parameter_condition_number": svals[0] / svals[-1] if svals[-1] > 1e-14 else np.inf,
        "ranked_parameters": sorted(
                    zip(parameter_names(decomp), param_norms), 
                    key=lambda x: x[1], reverse=True
                     ),
        "ranked_losses": sorted(
                        zip(loss_names(decomp), loss_norms),
                        key=lambda x: x[1], reverse=True
                    ),
        "ranked_losses": sorted(
                            zip(mmi_imbalance_names(decomp), mmi_imbalance_norms),
                            key=lambda x: x[1], reverse=True
                        )
    }