
Originally developed for molecular dynamics under the name hybrid Monte Carlo \cite{Duane_1987}, Hamiltonian Monte Carlo (HMC) is now commonly used for the subset of sampling problems where gradients of the posterior $p(\mathbf{m}|\mathbf{d}_\text{obs})$ with respect to the model parameters $\mathbf{m}$ are easy to compute. The cost of generating independent samples with HMC under increasing dimension $n$ grows as $\mathcal{O}(n^{5/4})$ \cite{Neal_2011}, whereas it grows as $\mathcal{O}(n^2)$ for  standard Metropolis-Hastings \cite{Creutz_1988}.

HMC constructs a Markov chain over an $n$-dimensional probability density function $p(\mathbf{m})$ using classical Hamiltonian mechanics \cite{Landau_1976}. The algorithm regards the current state $\mathbf{m}$ of the Markov chain as the location of a physical particle in $n$-dimensional space $\mathbb{M}$. It moves under the influence of a potential energy, $U$, which is defined as
\begin{eqnarray}
	U (\mathbf{m}) = - \ln p(\mathbf{m})\,.
\end{eqnarray}
In the case of a Gaussian probability density $p$, the potential energy $U$ is equal to the least-squares misfit $\chi(\mathbf{m})$. To complete the physical system, the state of the Markov chain needs to be artificially augmented with momentum variables $\mathbf{p}$ for every dimension and a generalized mass for every dimension pair.
The collection of resulting masses are contained in a positive definite mass matrix $\mathbf{M}$ of dimension $n\times n$. The momenta and the mass matrix define the kinetic energy of a model as
\begin{eqnarray}\label{eq:kinetic-energy}
	K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}\,.
\end{eqnarray}
In the HMC algorithm, the momenta $\mathbf{p}$ are drawn randomly from a multivariate Gaussian with covariance matrix $\mathbf{M}$. The location-dependent potential and kinetic energies constitute the total energy or Hamiltonian of the system,
\begin{eqnarray}
	H(\mathbf{m}, \mathbf{p}) = U(\mathbf{m}) + K(\mathbf{p}).
\end{eqnarray}
Hamilton's equations
\begin{eqnarray}
	\frac{d \mathbf{m}}{d \tau}  =  \frac{\partial H}{\partial \mathbf{p}}\,, \qquad
	\frac{d \mathbf{p}}{d \tau}  =  - \frac{\partial H}{\partial \mathbf{m}}\,.
\end{eqnarray}
determine the position of the particle as a function of the artificial time variable $\tau$. We can simplify Hamilton's equations using the fact that kinetic and potential energy depend only on momentum and location, respectively,
\begin{eqnarray}
	\frac{d \mathbf{m}}{d \tau}  =  \mathbf{M}^{-1} \mathbf{p}\,,\label{eq:mass-location} \qquad
	\frac{d \mathbf{p}}{d \tau}  =  - \frac{\partial U}{\partial \mathbf{m}}\,.\label{eq:momentum-potential}
\end{eqnarray}
Evolving $\mathbf{m}$ over time $\tau$ generates another possible state of the system with new position $\mathbf{\tilde{m}}$, momentum $\mathbf{\tilde{p}}$, potential energy $\tilde{U}$, and kinetic energy $\tilde{K}$. Due to the conservation of energy, the Hamiltonian is equal in both states. Successively drawing random momenta and evolving the system generates a distribution of the possible states of the system. Thereby, HMC samples the joint momentum and model space, referred to as phase space. As we are not interested in the momentum component of phase space, we marginalize over the momenta by simply dropping them. This results in samples drawn from $p(\mathbf{m})$.

If one could solve Hamilton's equations exactly, every proposed state would be a valid sample of $p(\mathbf{m})$. Since Hamilton's equations for non-linear forward models cannot be solved analytically, the system must be integrated numerically. Suitable integrators are symplectic, meaning that time reversibility, phase space partitioning and volume preservation are satisfied \cite{Neal_2011,Fichtner_2019}. However, the Hamiltonian is generally not preserved exactly when explicit time-stepping schemes are used. In this work, we employ the leapfrog method as described in \citeA{Neal_2011}. As the Hamiltonian is not preserved, the time evolution generates samples not exactly proportional to the original distribution. A Metropolis-Hastings correction step is therefore applied at the end of numerical integration.

In summary, samples are generated starting from a random model $\mathbf{m}$ in the following way:
%--------------------------------------------------------------------------------------------------------------------------------------------------------
\begin{enumerate}
	\item Propose momenta $\mathbf{p}$ according to the Gaussian with mean $\mathbf{0}$ and covariance $\mathbf{M}$;
	\item Compute the Hamiltonian $H$ of model $\mathbf{m}$ with momenta $\mathbf{p}$;
	\item Propagate $\mathbf{m}$ and $\mathbf{p}$ for some time $\tau$ to $\tilde{\mathbf{m}}$ and $\tilde{\mathbf{p}}$, using the discretized version of Hamilton's equations and a suitable numerical integrator;
	\item Compute the Hamiltonian $\tilde{H}$ of model $\tilde{\mathbf{m}}$ with momenta $\mathbf{\tilde{p}}$;
	\item Accept the proposed move $\mathbf{m} \rightarrow \tilde{\mathbf{m}}$ with probability
	      \begin{eqnarray}
		      \label{eq:Acceptance}
		      p_\text{accept} = \min \left( 1, \exp ( H-\tilde{H} ) \right)\,.
	      \end{eqnarray}
	\item If accepted, use (and count) $\tilde{\mathbf{m}}$ as the new state. Otherwise, keep (and count) the previous state. Then return to 1.
\end{enumerate}
%--------------------------------------------------------------------------------------------------------------------------------------------------------
The main factor influencing the acceptance rate of the algorithm is the conservation of energy, $H$, along the trajectory.
If the leapfrog integration has too large time steps, or the gradients of the misfit function are computed incorrectly (e.g., by badly discretizing the forward model), $H$ is less well conserved, and the algorithm's acceptance rate decreases.

The main cost of HMC, compared to other MCMC samplers, is the computation of the gradient $\partial U/\partial \mathbf{m}$ at every step in the leapfrog propagation. When gradients can be computed easily, HMC can provide improved performance for two reasons: (1) the reduced cost of generating independent samples, that is, the avoidance of random-walk behaviour \cite{Neal_2011}, and (2) the better scaling of HMC with increasing dimension \cite{Creutz_1988,Neal_2011}.

The tuning parameters in HMC are simulation time $\tau$ and the mass matrix $\mathbf{M}$. HMC has the potential to inject additional knowledge about the distribution $p$ via the mass matrix in order to enhance convergence significantly. At the same time, the abundance of tuning parameters also creates potential for choosing inefficient settings, leading to sub-optimal convergence. \citeA[]{Fichtner_2018c} and \citeA[]{Fichtner_2019} both illustrate how to create relevant mass matrices for tomographic inverse problems.

We adapt the specific tuning strategy for the mass matrix in this study depending on the target, as illustrated in the following  sections. However, for all targets we choose the size of the discrete time steps empirically such that the acceptance rate is close to the optimum of 65 \% \cite{Neal_2011}. This typically results in needing approximately 10 leap-frog steps per proposal, i.e. requiring this many forward and adjoint solves per proposal.
