# **Master Technical Specification: Project 32k-XAI Cognitive Simulation Platform**

**Document Status:** Aggressive / Strict / Final Architectural Blueprint  
**Project Leads:**

> * **Kian Mansouri Jamshidi** (Lead AI Architect, XAI Protocol Engineering, High-Performance Computing)  
> * **Sabihe Khorshidi** (Lead Cognitive Psychologist, Psychometric Mapping, Empirical Validation)

## **1\. Architectural Genesis and Paradigm Shift**

The contemporary landscape of Multi-Agent Reinforcement Learning (MARL) is fundamentally flawed for sociological and psychological simulation. Deep Neural Networks (DNNs) act as impenetrable black boxes, representing human behavior as homogeneous, latent-space approximations devoid of empirical psychological grounding.  
Project 32k-XAI proposes an aggressive paradigm shift: a **strictly deterministic, highly heterogeneous, Level-3 Mechanistically Interpretable Multi-Agent Platform**. We will instantiate an ecology of 32,000 autonomous agents. Every agent’s "brain" is a unified system of differential equations and Markov Decision Processes (MDPs), the parameters of which are strictly bounded by real-world human covariance matrices. There are no black boxes. Every action, betrayal, trust allocation, and system failure can be traced back to a specific cognitive parameter (e.g., temporal discounting, working memory, psychopathy), yielding a 100% transparent "glass-box" simulation.

### **1.1 The Transdisciplinary Synergy**

The architectural viability of this platform relies on the strict division of labor and epistemological synthesis between the two project leads:

> * **Sabihe Khorshidi (Cognitive Psychology Lead):** Responsible for the absolute validity of the psychological constructs. Sabihe oversees the translation of raw behavioral data (BART, Stop-Signal, Daw's Two-Step) into computational parameters. She dictates the *theoretical bounds*—mapping empirical DASS-21 scores to algorithmic exploration temperatures, and SD3 Dark Triad traits to social utility functions. Sabihe ensures the agents are mathematically functioning as humans do, not as hyper-rational economic robots.  
> * **Kian Mansouri Jamshidi (XAI & Computational Lead):** Responsible for the execution, scaling, and algorithmic translation. Kian engineers the vectorized tensor architectures required to run 32,000 parallel Markov models simultaneously at sub-millisecond latencies. Kian is tasked with building the XAI logging protocols, the Numba/JAX optimization frameworks, and the generative Gaussian Copula that scales Sabihe’s validated psychometric covariance matrix into 32,000 unique synthetic arrays.

## **2\. The Empirical Substrate: Hierarchical Parameter Extraction**

We reject the generation of arbitrary synthetic agents. The genome of the 32k population must be explicitly tethered to the multi-modal data pipeline secured in Phase 1\.  
**Sabihe’s Protocol:** Before Kian can initialize the population, the raw TSV/CSV files must undergo Hierarchical Bayesian Analysis (HBA) to extract subject-level posterior distributions.

### **2.1 Dimensional Mappings**

> 1. **Base Personality (OCEAN) & Dark Triad (SD3):** Act as static global priors. For instance, high Conscientiousness mathematically limits the variance in the delay discounting parameter \$k\$.  
> 2. **Psychopathology (DASS-21):** Acts as dynamic modulators. Stress directly throttles the working-memory capacity of the agent, suppressing Model-Based (MB) tree-search algorithms and forcing a reversion to Model-Free (MF) habitual responses.  
> 3. **Social Valuation (Trust/Ultimatum Games):** Calibrates the exact distribution of the Fehr-Schmidt inequity aversion coefficients (\$\\alpha, \\beta\$).  
> 4. **Risk & Inhibitory Control (UCLA ds000030):**  
   * *BART:* Extracted parameters govern the agent’s coefficient of risk tolerance \$\\rho\$.  
   * *Stop-Signal:* Data is fitted to a **Drift-Diffusion Model (DDM)** to extract drift rate (\$v\$), boundary separation (\$a\$), and non-decision time (\$t\_0\$). This computationally defines the agent's impulsivity under time pressure.  
> 5. **Reinforcement Learning Architecture (Daw's Two-Step Task):** Extracts the master control weight \$\\omega\$, defining the precise ratio of goal-directed vs. habitual learning for every individual.

## **3\. The Generative Engine: Multivariate Gaussian Copula**

To scale the empirically extracted parameters (from \$N \\approx 500\$ real humans) to \$N \= 32,000\$ synthetic agents without losing the complex non-linear correlations of human psychology, **Kian** will implement a strict Multivariate Gaussian Copula.

### **3.1 Mathematical Implementation**

Let \$X \= \[X\_1, X\_2, \\dots, X\_d\]\$ be the \$d\$-dimensional vector of psychological parameters (e.g., \$X\_1 \= \\text{Neuroticism}\$, \$X\_2 \= \\omega\$, \$X\_3 \= \\text{Delay Discounting } k\$).  
Since biological distributions are not purely normal (e.g., discount rates follow gamma distributions, Dark Triad scores are right-skewed), we cannot simply use a multivariate normal generator.

> 1. **Marginal Transformation:** Convert empirical observations to uniform distributions via their empirical cumulative distribution functions (eCDF): \$U\_i \= F\_i(X\_i)\$.  
> 2. **Gaussian Mapping:** Map these to standard normal variables: \$Z\_i \= \\Phi^{-1}(U\_i)\$.  
> 3. **Covariance Extraction:** Compute the Pearson correlation matrix \$\\Sigma\$ of the latent variables \$Z\$.  
> 4. **32k Synthesis:** Kian will draw 32,000 samples from the multivariate normal distribution \$N(0, \\Sigma)\$.  
> 5. **Inversion:** Apply the inverse CDFs to return the values to their true psychometric spaces: \$\\hat{X}\_i \= F\_i^{-1}(\\Phi(\\hat{Z}\_i))\$.

**Rigor Output:** The result is a \$32,000 \\times d\$ matrix. Agent \#14,052 might have clinically high anxiety (DASS), severely impaired inhibitory control (from Stop-Signal DDM \$a\$ parameter), and high Machiavellianism. These traits are probabilistically linked exactly as they are in the actual clinical population.

## **4\. The Mechanistic Cognitive Architecture (The "Brain")**

Deep neural networks are strictly banned from the decision-making loop. Every XAI agent’s brain is a continuous loop of transparent mathematical functions mapped directly to the psychometric matrix.

### **4.1 Module 1: Fehr-Schmidt Social Utility Function**

An agent \$i\$'s perceived value of a systemic state is not merely its own resource accumulation \$x\_i\$, but is filtered through its social parameters (derived from Ultimatum/Trust data):

\$\$U\_i(X) \= x\_i \- \\frac{\\alpha\_i}{N-1} \\sum\_{j \\neq i} \\max(x\_j \- x\_i, 0\) \- \\frac{\\beta\_i}{N-1} \\sum\_{j \\neq i} \\max(x\_i \- x\_j, 0)\$\$

> * \$\\alpha\_i\$ (Envy): Bounds the agent's hostility toward richer neighbors (correlated via Copula to Neuroticism/Psychopathy).  
> * \$\\beta\_i\$ (Guilt/Altruism): Bounds the agent's willingness to share (correlated to Agreeableness).

### **4.2 Module 2: State-Space Dual-System Learning (Daw Model)**

When navigating the environment to maximize \$U\_i\$, the agent evaluates paths using a hybrid learning algorithm.  
**Model-Free (Habitual) System:** Updates via simple Temporal Difference (TD) error:

\$\$Q\_{MF}(s,a) \\leftarrow Q\_{MF}(s,a) \+ \\alpha\_{MF} \[r \+ \\gamma \\max\_{a'} Q\_{MF}(s',a') \- Q\_{MF}(s,a)\]\$\$  
**Model-Based (Goal-Directed) System:** Computes forward-planning based on transition probability matrices \$P(s' \\vert{} s,a)\$:

\$\$Q\_{MB}(s,a) \= \\sum\_{s'} P(s' \\vert{} s,a) \\max\_{a'} Q\_{MB}(s',a')\$\$  
**The Arbitration Node (\$\\omega\$):** The ultimate action value is the weighted sum:

\$\$Q\_{net}(s,a) \= \\omega\_i Q\_{MB}(s,a) \+ (1 \- \\omega\_i) Q\_{MF}(s,a)\$\$  
*Crucial Sabihe-Kian Integration:* Sabihe's psychological constraints dictate that \$\\omega\_i\$ is not static. Kian will program \$\\omega\_i\$ to dynamically decay under environmental stress (simulated DASS-21 activation). A highly stressed agent loses the cognitive capacity for tree-search (\$Q\_{MB}\$) and reverts to blind habit (\$Q\_{MF}\$).

### **4.3 Module 3: Action Translation (Exploration vs. Exploitation)**

Values translate into probabilities via a Softmax function:

\$\$P(a\_k) \= \\frac{\\exp(\\tau\_i \\cdot Q\_{net}(s,a\_k))}{\\sum \\exp(\\tau\_i \\cdot Q\_{net}(s,a\_j))}\$\$  
Where \$\\tau\_i\$ (Inverse Temperature) is derived from the agent's Openness to Experience and baseline exploration parameters.

## **5\. System Execution and The Massively Parallel Sandbox**

The computational bottleneck is running 32,000 complex MDP solvers simultaneously.

### **5.1 Kian's Tensor Vectorization Protocol**

Object-Oriented Programming (OOP) with 32k separate class Agent(): instances via standard Python loops will cause massive CPU bottlenecking. Kian will construct the environment using **Data-Oriented Design (DOD)** via JAX or NumPy/Numba.

> * All 32,000 \$Q\$-tables are flattened into a global tensor \$\\mathbf{Q}\$ of shape (32000, States, Actions).  
> * All state transitions and reward calculations are executed as parallel matrix multiplications on the GPU.  
> * *Topology:* Agents will be linked using adjacency matrices to simulate social graphs (Small-World networks), allowing them to play Trust Games and resource trading simultaneously across the network graph.

## **6\. The Pure XAI Protocol: Mechanistic Interpretability**

The ultimate deliverable of this platform is the absolute elimination of algorithmic opacity. The system will possess an **Interrogation API**.  
When a macro-phenomenon emerges in the simulation (e.g., an economic crash, a breakdown of trust, a segregation of agent sub-populations), the researcher can freeze the simulation at tick \$t\$ and query any Agent \$A\_i\$.  
The XAI output will mathematically trace the decision:

> 1. **Log Output:** "Agent 14,052 betrayed Agent 8,991 at \$t=450\$."  
> 2. **Utility Trace:** "Calculated \$Q\_{net}\$ for \[Betray\] was 8.4, \$Q\_{net}\$ for \[Cooperate\] was 4.2."  
> 3. **Algorithmic Breakdown:** "Despite \$Q\_{MB}\$ showing long-term value for cooperation, the agent's DASS-21 Stress parameter spiked to 94th percentile due to recent resource loss. This forced arbitration weight \$\\omega\_i \\to 0.05\$, blinding the agent to the future and relying purely on immediate Model-Free (\$Q\_{MF}\$) greed."  
> 4. **Social Trace:** "Furthermore, baseline Fehr-Schmidt parameter \$\\alpha\_i \= 0.82\$ (Envy) generated negative utility for cooperating with an agent perceived to have higher resources."

## **7\. Master Execution Roadmap & Role Assignments**

### **Sprint 1: Empirical Data Ingestion & HBA Fitting**

> * **Sabihe Khorshidi:** Clean and standardize the HCP OCEAN, SD3, and DASS-21 datasets. Identify extreme outliers. Map the conceptual linkages between psychological traits and algorithmic hyper-parameters.  
> * **Kian Mansouri Jamshidi:** Write parsers for the BIDS/TSV formats downloaded from OpenNeuro (ds000030, ds003745) and OSF. Implement Bayesian fitting scripts (PyMC or Stan) to extract individual \$\\alpha, \\beta, \\omega, \\tau, \\rho, k\$ values from the raw human trial logs.

### **Sprint 2: The Copula Core**

> * **Sabihe Khorshidi:** Review the aggregate parameter distributions. Validate that the correlations mathematically make psychological sense (e.g., ensuring high psychopathy correlates with high Fehr-Schmidt envy and low Ultimatum offers).  
> * **Kian Mansouri Jamshidi:** Build the Multivariate Gaussian Copula pipeline. Feed the validated empirical parameters in, and generate the master 32k\_Agent\_Genome.h5 matrix.

### **Sprint 3: The Engine Sandbox**

> * **Kian Mansouri Jamshidi:** Engineer the JAX/Numba vectorized tensor environment. Build the state-space matrices, the reward distribution algorithms, and the network graph adjacency arrays.  
> * **Sabihe Khorshidi:** Design the "Scenarios" (e.g., Resource Scarcity, Wealth Inequality shocks). Define how environmental state changes map to the agents' DASS-21 stress triggers.

### **Sprint 4: The XAI Dashboard & Testing**

> * **Kian Mansouri Jamshidi:** Build the Interrogation API. Ensure that at every tick, matrices of \$Q\$, \$\\tau\$, and \$\\omega\$ can be logged and visualized in real-time.  
> * **Both Leads:** Run the first 32,000-agent systemic test. Monitor the emergent dynamics, pause the simulation, and use the XAI framework to causally explain macro-behaviors using the deterministic micro-equations.