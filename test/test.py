# =============================================================================
# >> IMPORTS
# =============================================================================
# Python
import rpyc
import time

if __name__ == "__main__":
    conn = rpyc.connect("localhost", 18811)
    network = conn.root.Network()

    # construct a fake state
    time_start = time.time()
    state = []
    for x in range(0, 92):
        state.append(1000.0)
    for x in range(0, 92):
        state.append(0.0)
    state.extend([0.0, 0.0, 0.0])
    print(f"action create time: {(time.time() - time_start) * 1000.0}ms")

    time_start = time.time()
    action = network.get_action(state)
    print(f"action: {action}, time: {(time.time() - time_start) * 1000.0}ms")

    time_start = time.time()
    network.post_action(1.0, state, True)
    print(f"post time: {(time.time() - time_start) * 1000.0}ms")