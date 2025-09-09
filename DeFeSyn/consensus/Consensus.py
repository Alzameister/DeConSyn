from copy import deepcopy


class Consensus:
    """
    Asynchronous consensus with dynamic epsilon (learning factor) per Carrascosa et al. (2024).
    Works elementwise over dicts of numpy arrays / tensors.
    """

    def __init__(self, model_type="CTGAN", alpha: float = 0.5):
        self.model_type = model_type
        # Dynamic epsilon state
        self.alpha = float(alpha)  # safety factor in (0,1)
        self.degree = 0                 # d_i (neighbors count)
        self.eps = 1.0                  # ε_i^t
        self.prev_eps = 1.0             # ε_i^{t-1}, for correction term
        self.all_eps = []            # history of eps_i^t, for logging/debugging
        # Snapshot of weights at start of consensus window (x_i^0)
        self.x0 = None                  # same structure as weights

    def set_degree(self, degree: int):
        """Initialize/reset ε_i when topology changes (Eq. (4): ε ≤ 1/d_i)."""
        if self.degree != 0:
            return

        self.degree = max(1, int(degree))
        self.eps = self.alpha / self.degree  # strict version of Eq. (4), is valid because epsilon is less than 1/d_i
        self.prev_eps = self.eps

    def start_consensus_window(self, x_i: dict):
        """Capture x_i^0 for the correction term (Eq. (10)/(13))."""
        self.x0 = deepcopy(x_i)
        self.prev_eps = self.eps  # first step in this window will see eps^{t+1}/eps^{t} = 1 ⇒ zero correction
        self.all_eps.append(self.eps)  # log current epsilon

    def step_with_neighbor(self, x_i: dict, x_j: dict, eps_j: float) -> dict:
        """
        Apply asynchronous consensus with dynamic epsilon:
        ε_i^{t+1} = min(ε_i^t, ε_j^t)   (Eq. 12)
        x update with correction term   (Eq. 13)
        """
        if self.model_type != "CTGAN":
            raise ValueError(f"Model type {self.model_type} not supported.")

        if self.x0 is None:
            self.x0 = deepcopy(x_i)

        # Update epsilon_i by the min rule (pairwise specialization of Eq. (5))
        new_eps = min(self.eps, float(eps_j))

        # Precompute correction factor: (1 - ε^{t+1}/ε^{t})
        if self.prev_eps <= 0:
            corr = 0.0
        else:
            corr = 1.0 - (new_eps / self.prev_eps)

        def blend_dict(A, B, A0):
            out = {}
            for k, a in A.items():
                b = B[k]
                a0 = A0[k]
                # consensus part: (1-ε) * a + ε * b
                cons = a + new_eps * (b - a)
                # correction term: (1 - ε^{t+1}/ε^{t}) * (a - a0)
                out[k] = cons - corr * (a - a0)
            return out

        gen_i = x_i['generator']
        gen_j = x_j['generator']
        dis_i = x_i['discriminator']
        dis_j = x_j['discriminator']

        x0_gen = self.x0['generator']
        x0_dis = self.x0['discriminator']

        updated = {
            'generator': blend_dict(gen_i, gen_j, x0_gen),
            'discriminator': blend_dict(dis_i, dis_j, x0_dis)
        }

        self.prev_eps = self.eps
        self.eps = new_eps

        return updated

    def get_eps(self) -> float:
        return float(self.eps)