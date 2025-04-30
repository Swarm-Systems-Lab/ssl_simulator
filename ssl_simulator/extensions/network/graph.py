"""
"""

__all__ = ["Graph"]

import numpy as np

from ssl_simulator.math import build_B, build_L_from_B

#######################################################################################

class Graph:
    def __init__(self, N, Z):
        self.N = N
        self.agents_status = np.ones(N, dtype=bool)
        self.alive = N

        self.Z = Z
        self.set_Z(Z)

    def set_Z(self, Z):
        """
        Set the new Z and build the Laplacian matrix
        """
        self.Z = Z
        self.B = build_B(Z, self.N)
        self.gen_L()

        # Update the graph description
        descr = ""
        for i in range(self.N):
            count = 0
            for edge in Z:
                count += int(i in edge)
            descr += rf"{i} - {count} connections" + "\n"
        self.descr = descr

    def gen_L(self):
        """
        Generate the Laplacian matrix considering agent status.
        """
        B_kill = np.copy(self.B)
        
        # Zero out columns in B corresponding to inactive (dead) agents
        for i in np.where(self.agents_status == 0)[0]:
            for j in range(B_kill.shape[1]):
                if B_kill[i, j] != 0:
                    B_kill[:, j] = 0

        self.B = B_kill

        # Generate the Laplacian matrix
        self.L = build_L_from_B(self.B)
        self.Lb = np.kron(self.L, np.eye(2))

        # Compute algebraic connectivity (smallest non-zero eigenvalue)
        eig_vals = np.linalg.eigvalsh(self.L)
        nonzero_eigs = eig_vals[eig_vals > 1e-7]
        self.lambda2 = np.min(nonzero_eigs) if nonzero_eigs.size > 0 else 0

    def kill_agents(self, agents_index):
        """
        Update the Lalplacian matrix to kill the connections of the
        specified agents, and update their status to (0)-"non-active"
        """
        if not isinstance(agents_index, list):
            agents_index = [agents_index]

        # Update the agents status
        self.agents_status[agents_index] = 0
        self.alive = np.sum(self.agents_status)

        # Generate the new Laplacian matrix
        self.gen_L()
    
    def __repr__(self):
        return self.descr
        

#######################################################################################

# Example usage
if __name__ == "__main__":
    graph = Graph(10, ((0,1),(1,2)))
    print(graph.B)
    print(graph.L)
