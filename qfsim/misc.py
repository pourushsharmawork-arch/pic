"""
Miscellaneous functions that need to be defined.
"""
import numpy as np
from numpy.typing import NDArray

##################################################################
# Essential Variables For The Sim.
##################################################################
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

##################################################################
# Basic qubit gates
##################################################################
Hadamard2x2 = (1 / np.sqrt(2)) * np.array([
                                 [1,1],
                                 [1,-1]
                                   ])

Hadamard4x4 = np.kron(Hadamard2x2, Hadamard2x2)

sigmaX = np.array([
    [0,1],
    [1,0]
])

sigmaY = np.array([
    [0, -1j],
    [1j, 0]
])

sigmaZ = np.array([
    [1, 0],
    [0, -1]
])


CZ = np.diag([1, 1, 1, -1])

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
]) 

##################################################################
# Essential Mathematical Entitities
##################################################################
def wrap_to_pi(angle: FloatArray | float) -> FloatArray:
    """Return the representative of ``angle`` in the half-open interval [-pi, pi)."""

    value = np.asarray(angle, dtype=float)
    return (value + np.pi) % (2.0 * np.pi) - np.pi

def T_local(theta, phi):
    """
    Givens Rotation Matrix

        T(theta, phi)
        =
        [[ exp(-i phi) cos(theta),  -sin(theta) ],
         [ sin(theta),               exp(i phi) cos(theta) ]]

    This is an SU(2) rotation with two parameters.
    """
    return np.array([
        [np.exp(-1j * phi) * np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.exp(1j * phi) * np.cos(theta)]
    ], dtype=complex)


def Givens(theta, phi):
    """
    Alias For T_local
    """
    return T_local(theta, phi)

def embed_T(N, i, j, T2):
    """
    Embed a 2x2 block T2 into an NxN identity matrix.

    i, j are zero-indexed mode labels.
    """
    M = np.eye(N, dtype=complex)
    idx = [i, j]

    for a in range(2):
        for b in range(2):
            M[idx[a], idx[b]] = T2[a, b]

    return M


def embed_Givens(N, i, j, T2):
    """
    Alias for embed_T
    """
    return embed_T(N, i, j, T2)


##################################################################
# Basic Functions To Run  Tests
##################################################################
def is_unitary(U, tol=1e-10):
    """
    Check U^dagger U = I.
    """
    U = np.asarray(U, dtype=complex)
    N = U.shape[0]
    return np.allclose(U.conj().T @ U, np.eye(N), atol=tol)


def random_unitary(N: int, seed: int | None = None) -> np.ndarray:
    """Generate a Haar-distributed numerical unitary using QR decomposition."""
    rng = np.random.default_rng(seed)

    Z = (
        rng.normal(size=(N, N))
        + 1j * rng.normal(size=(N, N))
    )

    Q, R = np.linalg.qr(Z)
    diagonal = np.diag(R)
    phases = np.ones_like(diagonal)

    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])

    return Q @ np.diag(np.conj(phases))