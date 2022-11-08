import numpy as np
from scipy.linalg import expm

class KalmanFilter:
    def __init__(self, A, B, C, D=None, t_step=.1, t_sim=60):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.nx = A.shape[0]
        self.ny = C.shape[0]
        self.nu = B.shape[1]
        self.t_step = t_step
        self.t_sim = t_sim
        self.N_sim = int(t_sim / t_step)

       
        self.x_data =           np.zeros((self.N_sim, self.nx, 1))
        self.x_data_observer =  np.zeros((self.N_sim, self.nx, 1))
        self.y_data =           np.zeros((self.N_sim, self.ny, 1))
        self.P_data =           np.zeros((self.N_sim, self.nx, self.nx))
        self.time =             np.arange(self.N_sim)*t_step

    def run(self, x0, x0_observer, sig_q, sig_r, sig_p, uk):

        vsig_q = np.ones(self.nx) * sig_q  # Process noise standard deviation
        vsig_r = np.ones(self.ny) * sig_r  # Measurement noise standard deviation
        vsig_p = np.ones(self.nx) * sig_p  # Initial state estimate standard deviation

        Q = np.diag(vsig_q ** 2)      # Process noise covariance
        R = np.diag(vsig_r ** 2)      # Measurement noise covariance
        P0 = np.diag(vsig_p ** 2)     # Initial state estimate covariance

        self.x_data[0] = x0
        self.x_data_observer[0] = x0_observer
        self.y_data[0] = self.C @ x0 + np.random.normal(0, sig_r, self.ny)
        self.P_data[0] = P0

        WX = np.random.normal(0, sig_q, (self.N_sim, self.nx, 1)) # Process noise
        WY = np.random.normal(0, sig_r, (self.N_sim, self.ny, 1)) # Measurement noise

        for i in range(self.N_sim-1):
            Pk = self.P_data[i]
            xk = self.x_data[i]
            xk_observer = self.x_data_observer[i]
            yk = self.y_data[i]

            # Correction step
            Lk = Pk @ self.C.T * np.linalg.inv(self.C @ Pk @ self.C.T + R)
            xk_observer_correct = xk_observer + Lk @ (yk - self.C @ xk_observer)
            Pk_correct = (np.eye(self.nx) - Lk @ self.C) @ Pk

            # Prediction step
            xk_observer_next = self.A @ xk_observer_correct + self.B @ uk
            Pk_next = self.A @ Pk_correct @ self.A.T + Q

            # Simulation step
            x_next = self.A @ xk + self.B @ uk + WX[i]
            y_next = self.C @ x_next + WY[i]

            # Store results
            self.x_data[i + 1] = x_next
            self.x_data_observer[i + 1] = xk_observer_next
            self.y_data[i + 1] = y_next
            self.P_data[i + 1] = Pk_next

def get_sys(step_size):
    # Define system
    nx = 3
    nu = 1
    ny = 1

    # Model parameter
    kAB = 1.5
    kBC = 3
    kCB = 2
    dvdt = 1
    VR = 10

    # Continous-time model in state space form
    A = np.array([[-kAB - dvdt / VR, 0, 0],
                [kAB, -kBC - dvdt / VR, kCB],
                [0, kBC, -kCB - dvdt / VR]])
    B = np.array([[dvdt / VR], [0], [0]]).reshape(nx, nu)
    C = np.array([[0, 1, 0]]).reshape(ny, nx)


    # Discretizing continous system
    A_dc = expm(A * step_size)
    B_dc = np.linalg.inv(A) @ (expm(A * step_size) - np.eye(nx)) @ B

    return A_dc, B_dc, C, nx, nu, ny


class Normal:
    def __init__(self, mu, Sigma):
        # Store inputs as class attributes, ensure that inputs are of type numpy with correct dimensions.
        self.mu = np.atleast_2d(mu)
        self.Sigma = np.atleast_2d(Sigma)
        
        # Compute values that are constant
        
        # 1) Number of dimensions
        self.n_dim = self.mu.size
        # 2) normalizing constant
        self.c = np.sqrt(1/(((2*np.pi)**self.n_dim)*np.linalg.det(self.Sigma)))
        # 3) Inverse covariance
        self.Sigma_inv = np.linalg.inv(self.Sigma)

        
    def pdf(self, x):
        # difference between mu and x:
        dx = x-self.mu
        # argument of the exponential
        e = np.diag(-0.5*dx.T@self.Sigma_inv@dx)
        # probability
        p = self.c*np.exp(e)
        
        return p

    def __mul__(self, other):

        # Product with other Gaussian
        if isinstance(other, Normal):
            # Get mu1 and m2 from both classes
            mu1 = self.mu
            mu2 = other.mu
            
            # Sigma1_inv and Sigma2_inv
            Sigma1_inv = self.Sigma_inv
            Sigma2_inv = other.Sigma_inv
            
            # Compute new Sigma
            Sigma12_inv = (Sigma1_inv+Sigma2_inv)
            Sigma12 = np.linalg.inv(Sigma12_inv)
            
            # Compute new mean 
            mu12 = Sigma12@(Sigma1_inv@mu1+Sigma2_inv@mu2)
        # Product with constant (matrix or scalar)
        elif isinstance(other, (np.ndarray, float)):
            A = np.atleast_2d(other)
            mu12 = A@self.mu
            Sigma12 = A.T@self.Sigma@A
        else:
            raise ValueError('Invalid type for multiplication')
        
        
        # Return new normal distribution        
        return Normal(mu=mu12, Sigma=Sigma12)

    def __add__(self, other):
        # Sum with other Gaussian
        if isinstance(other, Normal):
            # Get mu1 and m2 from both classes
            mu1 = self.mu
            mu2 = other.mu
            
            mu12 = mu1+mu2
            Sigma12 = self.Sigma+other.Sigma

        # Return new normal distribution        
        return Normal(mu=mu12, Sigma=Sigma12)


    