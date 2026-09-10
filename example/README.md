# Generating Operator YAML Files

`natto` can generate three sets of operators, each stored as a YAML file.
The `generate_*.py` scripts in this directory generate the corresponding YAML files:

- `generate_coupling_operators.py` generates `coupling_operators.yaml`
- `generate_reduction_operators.py` generates `reduction_operators.yaml`
- `generate_harmonic_operators.py` generates `harmonic_operators.yaml`

Run a script directly to generate the corresponding YAML file, e.g.:

```bash
python generate_coupling_operators.py
```

Each script allows configuring which ranks or tensor types to include; edit the
parameters at the top of the script before running.

The three are described below.


# harmonic_operators.yaml

The harmonic operator `H` takes the polyadic of a unit vector to the Cartesian
harmonic of a given weight, the Cartesian counterpart of a spherical harmonic.
The file contains the symbolic and numerical values of `H` along with the einsum
rule to apply it.

The data is organized in the following way:

{weight-normalization:
    {"symbolic": symbolic expression for the harmonic operator H,
    "numerical": numerical values of H,
    "rule": einsum rule to apply H to the copies of a unit vector
    }
}

- In "weight-normalization", weight is the weight of the harmonic, and
  normalization can be `none` or `unity`. Under `unity` the weight-fold
  contraction of the harmonic with a unit vector is the Legendre polynomial of
  the angle between the two.


# coupling_operators.yaml

The coupling operator `K` performs the Clebsch-Gordan-like coupling of two natural
tensors of weights l1 and l2 into a new natural tensor of weight l3.
The file contains the symbolic and numerical values of `K` along with the einsum rule
to apply it.

The data is organized in the following way:

{l1-l2-l3-normalization:
    {"symbolic": symbolic expression for the coupling operator K,
    "numerical": numerical values of K,
    "rule": einsum rule to apply K to the two input natural tensors
    }
}

- In "l1-l2-l3-normalization", l1 and l2 are the weights of the two input natural
  tensors, l3 is the weight of the output natural tensor, and normalization can be
  `none` or `unity`, indicating whether K is normalized or not.


# reduction_operators.yaml

These operators decompose a physical Cartesian tensor (e.g. polarizability, elasticity)
into its natural tensor components and reconstruct it back. Specifically:

- `extraction` extracts a natural tensor component from the physical tensor.
- `embedding` embeds a natural tensor component back into the physical tensor space.
- `decomposition` is their composition, taking the physical tensor straight to its
  weight-`l`, channel-`p` part without forming the natural tensor.

The file contains the symbolic and numerical values of the three for each
physical tensor and each natural tensor component (labeled by its weight).

The data is organized in the following way:

{physical_tensor_name:
    {"rank": rank of the physical tensor,
     "symmetry": symmetry of the physical tensor,
     "operators":
        weight:
            {"embedding": [{"symbolic": symbolic expression,
                   "numerical": numerical values }],
             "extraction": [{"symbolic": symbolic expression,
                   "numerical": numerical values }],
             "decomposition": [{"symbolic": symbolic expression,
                   "numerical": numerical values }]
            }
    }
}

- "physical_tensor_name" is the name of the physical tensor, such as "polarizability" "
elasticity".
- "weight" is the weight of the natural tensor in the decomposition of the physical
  tensor, which is a positive integer.
