"""
Nengo-based controller for a Unity ML-Agents environment.

This script integrates a spiking neural network (converted from a Keras model)
with a custom neuromorphic velocity estimator and a basal-ganglia/thalamus action-selection
loop. The network receives observations from Unity, estimates ball velocities using a
100 ms synaptic delay, processes the state through the SNN, and outputs discrete actions
(left / no-move / right) to the game.

Key features:
- Neuromorphic velocity computation (current - delayed observation)
- Population encoding of board + ball positions
- Basal ganglia / thalamus for action selection
- Unity ↔ Nengo bridge via a stepped node
"""

import nengo
import nengo_dl
import numpy as np
from tensorflow.keras.models import load_model


# Local utilities
from Utilities.unity_utils import (
    initialize_unity_environment,
    unity_environment_step,
)
from Utilities.ANN_to_SNN_conversion import (
    #create_spiking_nengodl_model,
    OBS_LENGTH,
)

from Utilities.nengoDL_utils_LIF_no_memory_one_layer import (
    create_spiking_nengodl_model,
    OBS_LENGTH,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Unity executable and trained Keras model
GAME_FILE_PATH = r"C:\Users\yisra\Dummy\Builds\velocity_observations\manual_velocity_with_noise\Dummy.exe"
KERAS_MODEL_PATH = "keras_models/reconstructed_model_LIF_simple_velocity_one_layer.h5"

# Unity runtime flags
UNITY_NO_GRAPHICS = False
UNITY_WORKER_ID = 0
UNITY_TIME_SCALE = 1.0
UNITY_CAPTURE_FRAME_RATE = 50

# Simulation timing
SIMULATION_TIME_STEP = 100  # ms between Unity steps
TOTAL_SIMULATION_TIME = 150  # seconds
NUM_RUNS = 1  # number of independent episodes

# Neural / synaptic parameters
TAU_SYN = 0.005  # synaptic time constant (t)
NO_MOVE_THRESHOLD = 0.2  # constant input to the "no-move" BG channel
NUM_BALLS = 4  # balls tracked in the environment
N_NEURONS = 300  # neurons per main ensemble

# =============================================================================
# GLOBAL STATE (shared across the Nengo node and simulation loop)
# =============================================================================

last_move: int = 0
past_state: np.ndarray = np.zeros(OBS_LENGTH)
reward: float = 0.0

env = None  # Unity environment handle
keras_session = None  # TensorFlow session for the converted Keras model
behavior_name: str = ""  # ML-Agents behavior identifier


# =============================================================================
# UNITY ↔ NENGO BRIDGE
# =============================================================================

def unity_node_function(t: float, data: np.ndarray) -> np.ndarray:
    """
    Nengo Node that synchronizes Unity steps with the Nengo simulation.

    Unity is only stepped every ``SIMULATION_TIME_STEP`` ms. Between steps the node
    returns the most recent observation.

    Parameters
    ----------
    t : float
        Current simulation time (seconds).
    data : np.ndarray
        Action value produced by the thalamus (size 1).

    Returns
    -------
    np.ndarray
        Latest observation vector from Unity (length ``OBS_LENGTH``).
    """
    global past_state, reward

    # Determine whether a new Unity step is required
    step_now = int(t * (1000 / SIMULATION_TIME_STEP))
    step_prev = int((t - 0.001) * (1000 / SIMULATION_TIME_STEP))
    if step_now == step_prev:
        return past_state

    # Execute action in Unity and retrieve new state / reward
    action = data[0]
    observation, step_reward = unity_environment_step(env, behavior_name, action)
    reward += step_reward
    past_state = observation
    return past_state


# =============================================================================
# NETWORK CONSTRUCTION
# =============================================================================

def create_nengo_model() -> nengo.Network:
    """
    Build the full spiking controller network.

    Returns
    -------
    nengo.Network
        Configured network ready for simulation.
    """
    # -------------------------------------------------------------------------
    # Index matrices: route Unity observations to the appropriate neural dimensions
    # -------------------------------------------------------------------------
    # Balls occupy the last 36 entries of the observation vector:
    #   [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
    balls_idxs = np.zeros((OBS_LENGTH, 2 * NUM_BALLS))
    velocity_idxs = np.zeros((OBS_LENGTH, 2 * NUM_BALLS))

    row = OBS_LENGTH - 36
    for col in range(0, 2 * NUM_BALLS, 2):
        # Position
        balls_idxs[row, col] = 1.0  # x
        balls_idxs[row + 1, col + 1] = 1.0  # y
        # Velocity
        velocity_idxs[row + 2, col] = 1.0  # vx
        velocity_idxs[row + 3, col + 1] = 1.0  # vy
        row += 4 # skip two rows each pair (x,y,vx,vy)

    # Board x-position is the first observation dimension. The y-position is constant.
    board_and_ball_idxs = np.hstack((np.eye(OBS_LENGTH, 1), balls_idxs))

    # -------------------------------------------------------------------------
    # Network definition
    # -------------------------------------------------------------------------
    model = nengo.Network(label="Controller")
    with model:
        # Unity interface
        game = nengo.Node(unity_node_function, size_in=1, size_out=OBS_LENGTH)

        # Converted Keras → NengoDL spiking model
        nengodl_model = create_spiking_nengodl_model(keras_session)

        # ---------------------------------------------------------------------
        # State ensemble: board + ball positions (radius tuned for unity scaling)
        # ---------------------------------------------------------------------
        state = nengo.Ensemble(
            n_neurons=N_NEURONS,
            dimensions=2 * NUM_BALLS + 1,
            radius=1.4,
        )
        nengo.Connection(
            game,
            state,
            transform=0.125 * board_and_ball_idxs.T,  # scaled to [-1,1]
            synapse=None,  # instantaneous
        )

        # ---------------------------------------------------------------------
        # Velocity estimator: current - 100 ms delayed observation
        # ---------------------------------------------------------------------
        velocities = nengo.Ensemble(n_neurons=N_NEURONS, dimensions=2 * NUM_BALLS)
        # Remove board dimension for velocity computation
        state_to_velocity = np.vstack([np.zeros((1, 2 * NUM_BALLS)), np.eye(2 * NUM_BALLS)])

        nengo.Connection(state, velocities, transform=8 * state_to_velocity.T, synapse=0.005)  # rescaling applied
        nengo.Connection(state, velocities, transform=-8 * state_to_velocity.T, synapse=0.105)  # rescaling applied

        # ---------------------------------------------------------------------
        # Feed observations into the SNN
        # ---------------------------------------------------------------------
        nengo.Connection(
            velocities,
            nengodl_model.obs_input,
            transform=velocity_idxs,
            synapse=0.01,
        )
        nengo.Connection(
            state,
            nengodl_model.obs_input,
            transform=8 * board_and_ball_idxs, # rescaling applied
            synapse=0.01,
        )

        # ---------------------------------------------------------------------
        # Action selection ensembles
        # ---------------------------------------------------------------------
        left_ensemble = nengo.Ensemble(
            n_neurons=100,
            dimensions=1,
            encoders=nengo.dists.Choice([[-1]]),
            intercepts=nengo.dists.Uniform(0.3, 1),
            label="Left Movement",
        )
        right_ensemble = nengo.Ensemble(
            n_neurons=100,
            dimensions=1,
            encoders=nengo.dists.Choice([[1]]),
            intercepts=nengo.dists.Uniform(0.3, 1),
            label="Right Movement",
        )
        no_movement_ensemble = nengo.Node(NO_MOVE_THRESHOLD)

        nengo.Connection(nengodl_model.action_output, left_ensemble, synapse=TAU_SYN)
        nengo.Connection(nengodl_model.action_output, right_ensemble, synapse=TAU_SYN)

        # ---------------------------------------------------------------------
        # Basal ganglia / thalamus decision loop
        # ---------------------------------------------------------------------
        bg = nengo.networks.BasalGanglia(dimensions=3, label="Basal Ganglia")
        thalamus = nengo.networks.Thalamus(dimensions=3, label="Thalamus")

        # BG receives inverted left signal (negative → positive utility)
        nengo.Connection(left_ensemble, bg.input[0], transform=-1, synapse=TAU_SYN)
        nengo.Connection(no_movement_ensemble, bg.input[1], synapse=TAU_SYN)
        nengo.Connection(right_ensemble, bg.input[2], synapse=TAU_SYN)

        nengo.Connection(bg.output, thalamus.input, synapse=TAU_SYN)

        # Thalamus drives Unity actions
        nengo.Connection(thalamus.output[0], game, transform=-1, synapse=TAU_SYN)  # left
        nengo.Connection(thalamus.output[1], game, transform=0, synapse=TAU_SYN)  # none
        nengo.Connection(thalamus.output[2], game, transform=1, synapse=TAU_SYN)  # right

        # ---------------------------------------------------------------------
        # Probes for analysis / debugging
        # ---------------------------------------------------------------------
        model.nengodl_model = nengodl_model
        model.p_thal = nengo.Probe(thalamus.output, synapse=0.01)
        model.p_game = nengo.Probe(game, synapse=0.01)
        model.p_vel = nengo.Probe(velocities, synapse=0.005)

    return model


# =============================================================================
# SIMULATION LOOP
# =============================================================================

def run_simulation() -> None:
    """Execute the full experiment for ``NUM_RUNS`` episodes."""
    global env, reward, keras_session, behavior_name

    for run_idx in range(NUM_RUNS):
        reward = 0.0

        # -----------------------------------------------------------------
        # Initialise external systems
        # -----------------------------------------------------------------
        keras_session = load_model(KERAS_MODEL_PATH, compile = False)
        env, behavior_name = initialize_unity_environment(
            GAME_FILE_PATH,
            capture_frame_rate=UNITY_CAPTURE_FRAME_RATE,
            no_graphics=UNITY_NO_GRAPHICS,
            time_scale=UNITY_TIME_SCALE,
            worker_id=UNITY_WORKER_ID,
        )

        # -----------------------------------------------------------------
        # Build and run the Nengo network
        # -----------------------------------------------------------------
        model = create_nengo_model()
        print(f"\n--- Starting run {run_idx + 1}/{NUM_RUNS} ---")
        with nengo_dl.Simulator(model) as sim:
            sim.run(TOTAL_SIMULATION_TIME)

        print(f"Run {run_idx + 1} finished – total reward: {reward:.2f}")

        # -----------------------------------------------------------------
        # Cleanup
        # -----------------------------------------------------------------
        env.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_simulation()
