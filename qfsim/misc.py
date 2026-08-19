"""
Miscellaneous functions that need to be defined.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import math



from collections.abc import Mapping
from enum import Enum
import inspect
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

# try:
#     import numpy as np
# except ImportError:
#     np = None

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

def real_rotation(angle: float) -> ComplexArray:
    """Return a small real parasitic two-mode rotation."""

    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=complex)


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


##################################################################################
##################################################################################
def basis_state(left_state: str, right_state: str) -> np.ndarray:
    """
    Returns the 2D row matrix representation of a 4-qubit computational basis state.
    """
    # Map each 2-bit string to its corresponding standard basis vector index
    mapping = {
        '00' : [1, 0, 0, 0],
        '01' : [0, 1, 0, 0],
        '10' : [0, 0, 1, 0],
        '11' : [0, 0, 0, 1]
    }
    
    if (left_state not in mapping) or (right_state not in mapping):
        raise ValueError("Input must be one of '00', '01', '10', or '11'.")

    # Creating a 2D array with shape (1, 16) explicitly forces a row matrix
    
    output = np.kron(
        np.array([mapping[left_state]]),
        np.array([mapping[right_state]])
    )
    
    return output 


def printstate(ψ):
    psi = ψ[0].tolist()
    basisstates = [
        '0000',
        '0001',
        '0010',
        '0011',
        '0100',
        '0101',
        '0110',
        '0111',
        '1000',
        '1001',
        '1010',
        '1011',
        '1100',
        '1101',
        '1110',
        '1111',
    ]        
    
    for i in range(0,len(basisstates)):
        print(f"{complex(
        round(psi[i].real, 2),
        round(psi[i].imag, 2)
        ):>15}   |{basisstates[i]}>")
##################################################################################
##################################################################################


def pretty_print_object(
    obj: Any,
    *,
    title: str | None = None,
    include_private: bool = True,
    include_properties: bool = False,
    max_depth: int | None = None,
    console: Console | None = None,
) -> Tree:
    """
    Display an object as a coloured recursive tree.

    Supports:
      - nested class instances
      - dictionaries
      - lists, tuples, sets and frozensets
      - dataclasses
      - __dict__
      - inherited __slots__
      - NumPy arrays
      - cyclic/shared references
      - optional @property evaluation
    """

    output_console = console or Console()
    seen: dict[int, str] = {}

    class AttributeErrorValue:
        def __init__(self, error: Exception):
            self.error = error

        def __repr__(self) -> str:
            return (
                f"<unavailable: {type(self.error).__name__}: "
                f"{self.error}>"
            )

    def safe_repr(value: Any) -> str:
        try:
            return repr(value)
        except Exception as error:
            return (
                f"<unrepresentable {type(value).__qualname__}: "
                f"{type(error).__name__}: {error}>"
            )

    def type_name(value: Any) -> str:
        return type(value).__qualname__

    def is_numpy_array(value: Any) -> bool:
        return np is not None and isinstance(value, np.ndarray)

    def is_numpy_scalar(value: Any) -> bool:
        return np is not None and isinstance(value, np.generic)

    def is_scalar(value: Any) -> bool:
        return (
            value is None
            or type(value) in {
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                bytearray,
            }
            or isinstance(value, (Enum, AttributeErrorValue))
            or is_numpy_scalar(value)
            or inspect.isroutine(value)
            or inspect.ismodule(value)
            or inspect.isclass(value)
        )

    def scalar_style(value: Any) -> str:
        if value is None:
            return "italic bright_black"
        if isinstance(value, bool):
            return "bold bright_yellow"
        if isinstance(value, str):
            return "bright_green"
        if isinstance(value, (int, float, complex)):
            return "bright_blue"
        if isinstance(value, Enum):
            return "bright_magenta"
        if isinstance(value, AttributeErrorValue):
            return "bold red"
        return "white"

    def scalar_label(name: str, value: Any) -> Text:
        label = Text()
        label.append(name, style="bold bright_cyan")
        label.append(" = ", style="dim")
        label.append(safe_repr(value), style=scalar_style(value))
        label.append(f"   ‹{type_name(value)}›", style="dim italic")
        return label

    def object_summary(value: Any) -> str:
        if is_numpy_array(value):
            return (
                f"ndarray · shape={value.shape} · dtype={value.dtype}"
            )

        if isinstance(value, Mapping):
            count = len(value)
            word = "entry" if count == 1 else "entries"
            return f"{type_name(value)} · {count} {word}"

        if isinstance(value, (list, tuple, set, frozenset)):
            count = len(value)
            word = "item" if count == 1 else "items"
            return f"{type_name(value)} · {count} {word}"

        return type_name(value)

    def object_label(name: str, value: Any) -> Text:
        label = Text()
        label.append(name, style="bold bright_cyan")
        label.append("  ")
        label.append(object_summary(value), style="bold bright_magenta")
        return label

    def reference_label(name: str, value: Any, original_path: str) -> Text:
        label = object_label(name, value)
        label.append("  ↪ ", style="bright_yellow")
        label.append(f"reference to {original_path}", style="italic yellow")
        return label

    def get_attributes(value: Any) -> dict[str, Any]:
        attributes: dict[str, Any] = {}

        # Standard instance attributes.
        try:
            attributes.update(vars(value))
        except (TypeError, AttributeError):
            pass
        except Exception as error:
            attributes["<__dict__>"] = AttributeErrorValue(error)

        # Slotted attributes, including inherited slots.
        for cls in type(value).__mro__:
            slots = cls.__dict__.get("__slots__", ())

            if isinstance(slots, str):
                slots = (slots,)

            for name in slots:
                if name in {"__dict__", "__weakref__"}:
                    continue
                if name in attributes:
                    continue
                if not include_private and name.startswith("_"):
                    continue

                try:
                    attributes[name] = getattr(value, name)
                except AttributeError:
                    continue
                except Exception as error:
                    attributes[name] = AttributeErrorValue(error)

        # Optionally evaluate properties.
        if include_properties:
            for cls in type(value).__mro__:
                for name, descriptor in cls.__dict__.items():
                    if not isinstance(descriptor, property):
                        continue
                    if name in attributes:
                        continue
                    if not include_private and name.startswith("_"):
                        continue

                    try:
                        attributes[name] = getattr(value, name)
                    except Exception as error:
                        attributes[name] = AttributeErrorValue(error)

        if not include_private:
            attributes = {
                name: value
                for name, value in attributes.items()
                if not name.startswith("_")
            }

        return attributes

    def add_child(
        parent: Tree,
        name: str,
        value: Any,
        path: str,
        depth: int,
    ) -> None:
        if is_scalar(value):
            parent.add(scalar_label(name, value))
            return

        object_id = id(value)

        if object_id in seen:
            parent.add(reference_label(name, value, seen[object_id]))
            return

        seen[object_id] = path
        branch = parent.add(object_label(name, value))

        if max_depth is not None and depth >= max_depth:
            branch.add(
                Text("… maximum depth reached", style="italic yellow")
            )
            return

        expand(branch, value, path, depth)

    def expand(
        branch: Tree,
        value: Any,
        path: str,
        depth: int,
    ) -> None:
        if is_numpy_array(value):
            array_string = np.array2string(
                value,
                threshold=value.size,
                separator=", ",
                max_line_width=100,
                suppress_small=False,
            )
            branch.add(Text(array_string, style="bright_white"))
            return

        if isinstance(value, Mapping):
            if not value:
                branch.add(Text("∅ empty mapping", style="dim italic"))
                return

            for key, item in value.items():
                key_repr = safe_repr(key)
                add_child(
                    branch,
                    f"[{key_repr}]",
                    item,
                    f"{path}[{key_repr}]",
                    depth + 1,
                )
            return

        if isinstance(value, (list, tuple)):
            if not value:
                branch.add(Text("∅ empty sequence", style="dim italic"))
                return

            for index, item in enumerate(value):
                add_child(
                    branch,
                    f"[{index}]",
                    item,
                    f"{path}[{index}]",
                    depth + 1,
                )
            return

        if isinstance(value, (set, frozenset)):
            if not value:
                branch.add(Text("∅ empty set", style="dim italic"))
                return

            for index, item in enumerate(sorted(value, key=safe_repr)):
                add_child(
                    branch,
                    f"{{{index}}}",
                    item,
                    f"{path}{{{index}}}",
                    depth + 1,
                )
            return

        attributes = get_attributes(value)

        if not attributes:
            branch.add(Text(safe_repr(value), style="white"))
            return

        for attribute_name, attribute_value in attributes.items():
            add_child(
                branch,
                attribute_name,
                attribute_value,
                f"{path}.{attribute_name}",
                depth + 1,
            )

    # Construct root.
    if is_scalar(obj):
        root = Tree(scalar_label("value", obj))
    else:
        root = Tree(
            object_label("root", obj),
            guide_style="bright_blue",
        )
        seen[id(obj)] = "root"
        expand(root, obj, "root", 0)

    heading = title or type_name(obj)

    panel = Panel(
        root,
        title=Text(f" {heading} ", style="bold bright_white"),
        subtitle=Text(" recursive object inspector ", style="dim"),
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=False,
    )

    output_console.print(panel)
    return None