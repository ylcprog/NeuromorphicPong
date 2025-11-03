"""
Converter for a custom Keras model into a spiking neural network.

This module constructs a spiking Nengo network that replicates the behavior of a
pre-trained Keras model consisting of:
- A single gated dense encoder layer (256 units)
- A mu-head dense layer producing a continuous action value

The conversion uses **individual Nengo Ensembles** rather than the high-level
`LayerConverter`, allowing fine-grained control over synapses, biases, and scaling.
A constant-bias node and a custom input-preprocessing lambda inject fixed values
into the observation vector (hard-coded for ``OBS_LENGTH`` = 38 or 45).

Key design choices:
- ``gain`` and ``bias`` are fixed to mimic the original TensorNode behavior.
- Firing-rate amplitudes are scaled (``*0.002``) to match SoftLIFRate training.
- Synaptic time constants are tuned for stability (e.g., 0.02 s on the mu layer).

The resulting network exposes:
- ``obs_input``: Node to feed Unity observations
- ``action_output``: Node emitting the continuous action scalar
- Probes for debugging (input, encoder spikes, action)

Intended for online control in a Unity ML-Agents loop.
"""

import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# Constants (mirrored from training / environment)
# -----------------------------------------------------------------------------
SYNAPSE = 0.02  # Default synaptic time constant for most connections
OBS_LENGTH = 45  # Observation vector length (38 or 45 supported)


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------
def create_spiking_nengodl_model(keras_model: tf.keras.Model) -> nengo.Network:
    """
    Build a spiking NengoDL network from a pre-trained Keras model.

    The function manually extracts weights from two layers (``encoder_0`` and ``mu``)
    and wires them into Nengo Ensembles. Fixed constants are injected into the
    observation vector via ``set_constants``.

    Parameters
    ----------
    keras_model : tf.keras.Model
        Loaded Keras model containing layers named ``encoder_0`` and ``mu``.

    Returns
    -------
    nengo.Network
        Fully configured network with input/output nodes and diagnostic probes.
    """

    # -------------------------------------------------------------------------
    # Helper: inject hard-coded constants into the observation vector
    # -------------------------------------------------------------------------
    def set_constants(t: float, x: np.ndarray) -> np.ndarray:
        """
        Modify the incoming observation vector in-place with environment-specific
        constants. Supports both OBS_LENGTH=38 and OBS_LENGTH=45.
        """
        if OBS_LENGTH == 38:
            x[1] = -4.2
        else:  # OBS_LENGTH == 45
            x[1] = -4.2
            x[2] = 0.0
            x[3] = -1.259889
            x[4] = -0.4856576
            x[5] = -7.226348
            x[6] = -1.259889
            x[7] = -0.4856576
            x[8] = -7.226348
        return x

    # -------------------------------------------------------------------------
    # Network definition
    # -------------------------------------------------------------------------
    with nengo.Network() as model:
        # ---------------------------------------------------------------------
        # Input node – receives raw Unity observations
        # ---------------------------------------------------------------------
        input_node = nengo.Node(
            output=set_constants,
            size_in=OBS_LENGTH,
            size_out=OBS_LENGTH,
            label="obs_input",
        )

        # ---------------------------------------------------------------------
        # Bias node – constant 1.0 for affine transformations
        # ---------------------------------------------------------------------
        bias_one = nengo.Node(output=1.0, label="bias_one")

        # ---------------------------------------------------------------------
        # Encoder layer (256 neurons, replicating a gated dense layer)
        # ---------------------------------------------------------------------
        encoder_weights = keras_model.get_layer("encoder_0").get_weights()
        encoder_W, encoder_b = encoder_weights[0], encoder_weights[1]

        dense_node = nengo.Ensemble(
            n_neurons=256,
            dimensions=1,
            gain=nengo.dists.Choice([1.0]),
            bias=nengo.dists.Choice([0.0]),
            label="encoder_0",
        )
        # Input → neurons (transposed weights)
        nengo.Connection(
            input_node,
            dense_node.neurons,
            transform=encoder_W.T,
            synapse=0.005,
        )
        # Bias injection
        nengo.Connection(
            bias_one,
            dense_node.neurons,
            transform=encoder_b.reshape(-1, 1),
            synapse=None,
        )

        # ---------------------------------------------------------------------
        # Mu head (200 neurons → continuous action scalar)
        # ---------------------------------------------------------------------
        mu_weights = keras_model.get_layer("mu").get_weights()
        mu_W, mu_b = mu_weights[0], mu_weights[1]

        mu_node = nengo.Ensemble(
            n_neurons=200,
            dimensions=1,
            radius=1.4,  # Keeps represented values in ~[-1.4, 1.4]
            label="mu_head",
        )
        # Encoder spikes → mu ensemble
        nengo.Connection(
            dense_node.neurons,
            mu_node,
            transform=mu_W.T * 0.002,  # Amplitude scaling from training
            synapse=SYNAPSE,
        )
        # Bias for mu
        nengo.Connection(
            bias_one,
            mu_node,
            transform=mu_b.reshape(-1, 1),
            synapse=None,
        )

        # ---------------------------------------------------------------------
        # Action output node (raw scalar, no tanh – caller applies if needed)
        # ---------------------------------------------------------------------
        action_output = nengo.Node(size_in=1, label="action_output")
        nengo.Connection(mu_node, action_output, synapse=None)

        # ---------------------------------------------------------------------
        # Probes for introspection
        # ---------------------------------------------------------------------
        model.obs_input = input_node
        model.action_output = action_output

        model.input_probe = nengo.Probe(input_node, synapse=0.01)
        model.action_probe = nengo.Probe(action_output, synapse=0.01)

        # Spike raster of first 9 encoder neurons (useful for debugging)
        model.encoder_spikes = nengo.Probe(dense_node.neurons[:9], synapse=0.05)
        model.encoder_rates = nengo.Probe(dense_node.neurons[:9])

    return model
