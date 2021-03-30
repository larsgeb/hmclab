The Hamiltonian Monte Carlo algorithm
=====================================

Originally developed for molecular dynamics under the name hybrid Monte Carlo :cite:p:`Duane_1987`, Hamiltonian Monte Carlo (HMC) is now commonly used for the subset of sampling problems where gradients of the posterior :math:`p(\mathbf{m}|\mathbf{d}_\text{obs})` with respect to the model parameters :math:`\mathbf{m}` are easy to compute. The cost of generating independent samples with HMC under increasing dimension :math:`n` grows as :math:`\mathcal{O}(n^{5/4})` :cite:p:`Neal_2011`, whereas it grows as :math:`\mathcal{O}(n^2)` for  standard Metropolis-Hastings :cite:p:`Creutz_1988`.

Theory
******

HMC constructs a Markov chain over an :math:`n`-dimensional probability density function :math:`p(\mathbf{m})` using classical Hamiltonian mechanics :cite:p:`Landau_1976`. The algorithm regards the current state :math:`\mathbf{m}` of the Markov chain as the location of a physical particle in :math:`n`-dimensional space :math:`\mathbb{M}`. It moves under the influence of a potential energy, :math:`U`, which is defined as

.. math::
   :nowrap:

   \begin{eqnarray}
       U (\mathbf{m}) = - \ln p(\mathbf{m})\,.
   \end{eqnarray}

In the case of a Gaussian probability density :math:`p`, the potential energy :math:`U` is equal to the least-squares misfit :math:`\chi(\mathbf{m})`. To complete the physical system, the state of the Markov chain needs to be artificially augmented with momentum variables :math:`\mathbf{p}` for every dimension and a generalized mass for every dimension pair.
The collection of resulting masses are contained in a positive definite mass matrix :math:`\mathbf{m}` of dimension :math:`n\times n`. The momenta and the mass matrix define the kinetic energy of a model as

.. math::
   :nowrap:

   \begin{eqnarray}
       K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}\,.
   \end{eqnarray}

In the HMC algorithm, the momenta :math:`\mathbf{p}` are drawn randomly from a multivariate Gaussian with covariance matrix :math:`\mathbf{m}`. The location-dependent potential and kinetic energies constitute the total energy or Hamiltonian of the system,

.. math::
   :nowrap:

   \begin{eqnarray}
   	H(\mathbf{m}, \mathbf{p}) = U(\mathbf{m}) + K(\mathbf{p}).
   \end{eqnarray}

Hamilton's equations

.. math::
   :nowrap:
   
   \begin{eqnarray}
       \frac{d \mathbf{m}}{d \tau}  =  \frac{\partial H}{\partial \mathbf{p}}\,, \qquad
       \frac{d \mathbf{p}}{d \tau}  =  - \frac{\partial H}{\partial \mathbf{m}}\,,
   \end{eqnarray}

determine the position of the particle as a function of the artificial time variable :math:`\tau`. We can simplify Hamilton's equations using the fact that kinetic and potential energy depend only on momentum and location, respectively,

.. math::
   :nowrap:
   
   \begin{eqnarray}
       \frac{d \mathbf{m}}{d \tau}  =  \mathbf{M}^{-1} \mathbf{p}\,,\qquad
       \frac{d \mathbf{p}}{d \tau}  =  - \frac{\partial U}{\partial \mathbf{m}}\,.
   \end{eqnarray}

Evolving :math:`\mathbf{m}` over time :math:`\tau` generates another possible state of the system with new position :math:`\mathbf{\tilde{m}}`, momentum :math:`\mathbf{\tilde{p}}`, potential energy :math:`\tilde{U}`, and kinetic energy :math:`\tilde{K}`. Due to the conservation of energy, the Hamiltonian is equal in both states. Successively drawing random momenta and evolving the system generates a distribution of the possible states of the system. Thereby, HMC samples the joint momentum and model space, referred to as phase space. As we are not interested in the momentum component of phase space, we marginalize over the momenta by simply dropping them. This results in samples drawn from :math:`p(\mathbf{m})`.

From theory to code
*******************

If one could solve Hamilton's equations exactly, every proposed state would be a valid sample of :math:`p(\mathbf{m})`. Since Hamilton's equations for non-linear forward models cannot be solved analytically, the system must be integrated numerically. Suitable integrators are symplectic, meaning that time reversibility, phase space partitioning and volume preservation are satisfied :cite:p:`Neal_2011,Fichtner_2019`. However, the Hamiltonian is generally not preserved exactly when explicit time-stepping schemes are used. In this work, we employ the leapfrog method as described in :cite:t:`Neal_2011`. As the Hamiltonian is not preserved, the time evolution generates samples not exactly proportional to the original distribution. A Metropolis-Hastings correction step is therefore applied at the end of numerical integration.

In summary, samples are generated starting from a random model :math:`\mathbf{m}` in the following way:


.. admonition:: The Hamiltonian Monte Carlo algorithm

   1. Propose momenta :math:`\mathbf{p}` according to the Gaussian with mean :math:`\mathbf{0}` and covariance :math:`\mathbf{m}`;
   2. Compute the Hamiltonian :math:`H` of model :math:`\mathbf{m}` with momenta :math:`\mathbf{p}`;
   3. Propagate :math:`\mathbf{m}` and :math:`\mathbf{p}` for some time :math:`\tau` to :math:`\tilde{\mathbf{m}}` and :math:`\tilde{\mathbf   {p}}`, using the discretized version of Hamilton's equations and a suitable numerical integrator;
   4. Compute the Hamiltonian :math:`\tilde{H}` of model :math:`\tilde{\mathbf{m}}` with momenta :math:`\mathbf{\tilde{p}}`;
   5. Accept the proposed move :math:`\mathbf{m} \rightarrow \tilde{\mathbf{m}}` with probability
   
   .. math::
      :nowrap:
   
      \begin{eqnarray}
          p_\text{accept} = \min \left( 1, \exp ( H-\tilde{H} ) \right)\,.
      \end{eqnarray}
   
   6. If accepted, use (and count) :math:`\tilde{\mathbf{m}}` as the new state. Otherwise, keep (and count) the previous state. Then return    to 1.

HMC compared
************

The main factor influencing the acceptance rate of the algorithm is the conservation of energy, :math:`H`, along the trajectory.
If the leapfrog integration has too large time steps, or the gradients of the misfit function are computed incorrectly (e.g., by badly discretizing the forward model), :math:`H` is less well conserved, and the algorithm's acceptance rate decreases.

The main cost of HMC, compared to other MCMC samplers, is the computation of the gradient :math:`\partial U/\partial \mathbf{m}` at every step in the leapfrog propagation. When gradients can be computed easily, HMC can provide improved performance for two reasons: (1) the reduced cost of generating independent samples, that is, the avoidance of random-walk behaviour :cite:p:`Neal_2011`, and (2) the better scaling of HMC with increasing dimension :cite:p:`Creutz_1988,Neal_2011`.

Tuning parameters
*****************

The tuning parameters in HMC are simulation time :math:`\tau` and the mass matrix :math:`\mathbf{m}`. HMC has the potential to inject additional knowledge about the distribution :math:`p` via the mass matrix in order to enhance convergence significantly. At the same time, the abundance of tuning parameters also creates potential for choosing inefficient settings, leading to sub-optimal convergence. :cite:t:`Fichtner_2018c` and :cite:t:`Fichtner_2019` both illustrate how to create relevant mass matrices for tomographic inverse problems.

We adapt the specific tuning strategy for the mass matrix in this study depending on the target, as illustrated in the following  sections. However, for all targets we choose the size of the discrete time steps empirically such that the acceptance rate is close to the optimum of 65 \% :cite:p:`Neal_2011`. This typically results in needing approximately 10 leap-frog steps per proposal, i.e. requiring this many forward and adjoint solves per proposal.

Further reading
***************

.. bibliography::

