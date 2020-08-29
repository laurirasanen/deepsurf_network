# =============================================================================
# >> IMPORTS
# =============================================================================
# Python
import rpyc

if __name__ == "__main__":
    conn = rpyc.connect("localhost", 18811)
    network = conn.root.Network()

    # construct a fake state
    state_distances = []
    state_surfaces = []
    for x in range(0, 92):
        state_distances.append(1000.0)
        state_surfaces.append(0)
    state_velocity = [0.0, 0.0, 0.0]
    state = (state_distances, state_surfaces, state_velocity,)

    action = network.get_action(state)
    print(action)
