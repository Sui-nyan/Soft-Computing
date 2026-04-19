## 1. Optimization problem

Let the objective be

$$
\min_{x \in D} f(x),
$$

where the feasible domain is the box

$$
D = \prod_{j=1}^d [\ell_j, u_j] \subset \mathbb{r}^d.
$$

Each root agent lives in $ \mathbb{r}^d $, with position constrained by projection onto $D$. 

## 2. Root population state

At iteration (t), root (i) has state

$$
R_i(t) = \Big(x_i(t), e_i(t),, p_i(t),, f_i(t),, s_i(t)\Big),
$$

where:

* $x_i(t) \in D$: current position,
* $e_i(t) \ge 0$: energy,
* $p_i(t)$: personal best position,
* $f_i(t) = f(x_i(t))$: current fitness,
* $s_i(t) \in {\text{ALIVE}, \text{DEAD}}$: status. 

The personal best satisfies

$$
p_i(t) = \arg\min_{\tau \le t} f(x_i(\tau)).
$$

The algorithm also maintains the global best visited solution

$$
x_*(t) = \arg\min_{i,\tau \le t} f(x_i(\tau)).
$$


## 3. Moisture field

The environment is a dynamic scalar field $M(t)(x)$, represented as a Gaussian kernel sum plus a baseline:

$$
M(t)(x)
=

M_0 + \sum_{k=1}^{K_t} a_k(t)
\exp!\left(
-\frac{|x-c_k(t)|^2}{2\sigma^2}
\right),
$$

where:

* $M_0$ > 0$: base moisture,
* $c_k(t)$: center of patch (k),
* $a_k(t)$: strength of patch (k) (positive = attractive, negative = depleted),
* $\sigma > 0$: kernel width. 

The gradient used for hydrotropism is

$$
\nabla M(t)(x)
=

\sum_{k=1}^{K_t}
a_k(t)
\exp!\left(
-\frac{|x-c_k(t)|^2}{2\sigma^2}
\right)
\left(
-\frac{x-c_k(t)}{\sigma^2}
\right).
$$

This is exactly the Gaussian gradient implemented in the code. 

Patch strengths evaporate over time:

$$
a_k(t+1) = (1-\rho), a_k(t),
$$

where $\rho \in (0,1)$ is the evaporation rate, and patches with very small magnitude are removed. 


## 4. Direction components

Each living root computes a movement direction from five normalized components.

Define the normalization operator

$$
\operatorname{norm}(v)=
\begin{cases}
\dfrac{v}{|v|}, & |v|>\varepsilon,[4pt]
0, & \text{otherwise}.
\end{cases}
$$



For root (i), the components are:

### Hydrotropism

$$
h_i(t) = \operatorname{norm}!\big(\nabla M(t)(x_i(t))\big).
$$

### Gravity

Given gravity vector (g),
$$
g^\sharp = \operatorname{norm}(g).
$$

### Random exploration

$$
r_i(t) = \operatorname{norm}(\xi_i(t)), \qquad \xi_i(t) \sim \mathcal{N}(0,I_d).
$$

### Competition repulsion

For repulsion radius (R), the repulsion from other alive roots is

$$
q_i(t)
=

\sum_{j\ne i}
\mathbf{1}{0<|x_i(t)-x_j(t)|<R}
\left(1-\frac{|x_i(t)-x_j(t)|}{R}\right)
\frac{x_i(t)-x_j(t)}{|x_i(t)-x_j(t)|},
$$

then normalized as

$$
c_i(t) = \operatorname{norm}(q_i(t)).
$$

### Personal-best attraction

$$
b_i(t) = \operatorname{norm}(p_i(t) - x_i(t)).
$$

These are the exact ingredients used in `calculate_direction`. 


## 5. Movement rule

Let the weights be

$$
(w_h, w_g, w_r, w_c)
$$

for hydrotropism, gravity, randomness, and competition, and let (w_b) be the personal-best weight.

Then the unnormalized direction is

$$
v_i(t)
=

w_h h_i(t)

* w_g g^\sharp
* w_r r_i(t)
* w_c c_i(t)
* w_b b_i(t).
$$

If $v_i(t)$ is numerically zero, a random Gaussian direction is substituted. The final direction is

$$
d_i(t) = \operatorname{norm}(v_i(t)).
$$



The adaptive step length is

$$
\eta_i(t)
=

\Delta \cdot \operatorname{clip}!\left(\frac{e_i(t)}{5},,0.25,,2.0\right),
$$

where $\Delta$ is the base step size. 

The proposed new position is

$$
\tilde{x}_i(t+1) = x_i(t) + \eta_i(t) d_i(t),
$$

followed by box projection:

$$
x_i(t+1) = \Pi_D!\left(\tilde{x}_i(t+1)\right).
$$


## 6. Fitness improvement and energy update

After moving, the new fitness is

$$
f_i(t+1) = f(x_i(t+1)).
$$

Improvement is defined for minimization as

$$
\Delta f_i(t) = f_i(t) - f_i(t+1).
$$

The algorithm uses **relative improvement**

$$
\gamma_i(t)
=

\operatorname{clip}!\left(
\frac{\max(\Delta f_i(t),0)}{|f_i(t)|+\varepsilon},
0,1
\right).
$$

Then energy updates according to

$$
e_i(t+1)
=

e_i(t) + \alpha \gamma_i(t) - c_{\text{step}},
$$

where (\alpha) is the reward factor and (c_{\text{step}}) is the movement cost. If (e_i(t+1) \le 0), the root dies. 

This is one of the central mechanisms of the optimizer: good moves replenish energy, while poor moves gradually kill agents.


## 7. Personal-best update

If the new position improves the root’s historical best, then

$$
\text{if } f_i(t+1) < f(p_i(t)), \quad
p_i(t+1) = x_i(t+1),
$$

otherwise

$$
p_i(t+1) = p_i(t).
$$


## 8. Moisture deposition and depletion

The algorithm modifies the field based on root behavior.

### Positive deposit from successful movement

If $\gamma_i(t) > 0$, add a positive moisture patch at $x_i(t+1)$ with strength

$$
a_{\text{dep}} = \beta_{\text{dep}} , \gamma_i(t),
$$

where $\beta_{\text{dep}}$ is `deposit_scale`. 

### Extra deposit for new personal best

If a new personal best is found, add another deposit of size

$$
a_{\text{best}} = 0.5,\beta_{\text{dep}}.
$$


### Depletion from repeated visitation

Every move adds a negative patch at the current location:

$$
a_{\text{deplete}} = -\beta_{\text{depl}},
$$

where $\beta_{\text{depl}}$ is `depletion_scale`. 

So the field learns which regions are promising, but also discourages overcrowding and repeated exploitation.


## 9. Splitting rule

If a root has enough energy,

$$
e_i(t) \ge e_{\text{split}},
$$

and population capacity allows it, then it can split into children. 

Suppose $m_i$ children are created. Let $\lambda \in (0,1)$ be the split ratio. Then:

* total child energy:
  $$
  e_{\text{child,total}} = \lambda e_i(t),
  $$

* each child receives
  $$
  e_{\text{child}} = \frac{\lambda e_i(t)}{m_i},
  $$

* the parent keeps
  $$
  e_i(t+1) = (1-\lambda)e_i(t).
  $$



Each child is placed near the parent:

$$
x_{i,k}(t)
=

\Pi_D!\left(
x_i(t) + \delta, \hat{u}_{i,k}(t)
\right),
$$

where $\delta$ is `child_offset`, and

$$
\hat{u}_{i,k}(t)
=

\operatorname{norm}!\left(
\hat{d}*i(t) + 0.3,\zeta*{i,k}(t)
\right),
\qquad
\zeta_{i,k}(t) \sim \mathcal{N}(0,I_d),
$$

with $\hat{d}_i(t)$ being the parent’s last movement direction. 

This means children inherit the parent’s search tendency with random perturbation.

## 10. Compact formulation of one iteration

A compact iteration for each living root (i) is:

$$
d_i(t) =
\operatorname{norm}\Big(
w_h h_i(t) + w_g g^\sharp + w_r r_i(t) + w_c c_i(t) + w_b b_i(t)
\Big),
$$

$$
x_i(t+1) =
\Pi_D!\left(
x_i(t) + \Delta \cdot \operatorname{clip}!\left(\frac{e_i(t)}{5},0.25,2\right)d_i(t)
\right),
$$

$$
\gamma_i(t)=
\operatorname{clip}!\left(
\frac{\max(f(x_i(t))-f(x_i(t+1)),0)}{|f(x_i(t))|+\varepsilon},
0,1
\right),
$$

$$
e_i(t+1) = e_i(t) + \alpha \gamma_i(t) - c_{\text{step}}.
$$

Then:

* deposit positive moisture if $\gamma_i(t) > 0$,
* deposit bonus moisture if personal best improves,
* always deplete the visited region,
* kill root if $e_i(t+1) \le 0$. 

## 11. Shortend version

$$
\min_{x\inD} f(x), \qquad D=\prod_{j=1}^d[\ell_j,u_j]
$$

$$
M(t)(x)=M_0+\sum_{k=1}^{K_t} a_k(t)
\exp!\left(-\frac{|x-c_k(t)|^2}{2\sigma^2}\right)
$$

$$
d_i(t)=\operatorname{norm}!\left(
w_h \operatorname{norm}(\nabla M(t)(x_i(t)))
+w_g\hat g+w_r\hat\xi_i(t)
+w_c\hat q_i(t)
+w_b\operatorname{norm}(p_i(t)-x_i(t))
\right)
$$

$$
x_i(t+1)=\Pi_D!\left(x_i(t)+\Delta_i(t) d_i(t)\right),
\qquad
\Delta_i(t)=\Delta\cdot \operatorname{clip}!\left(\frac{e_i(t)}{5},0.25,2\right)
$$

$$
\gamma_i(t)=
\operatorname{clip}!\left(
\frac{\max(f(x_i(t))-f(x_i(t+1)),0)}{|f(x_i(t))|+\varepsilon},
0,1
\right)
$$

$$
e_i(t+1)=e_i(t)+\alpha\gamma_i(t)-c_{\text{step}}
$$

with splitting applied when $e_i^{(t)}\ge e_{\text{split}}$. 
