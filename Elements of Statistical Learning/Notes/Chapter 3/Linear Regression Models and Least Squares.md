# Model assumptions

Suppose we have a dataset $\\{(x_i, y_i)\\}_{i=1}^{N}$ where $x_i = {({x_i}_j)^T}\_{j=1}^{\ p} \in \mathbb{R}^p$ and $y_i \in \mathbb{R}$. We assume the outputs depend linearly on the inputs governed by the equation:

### $$y_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij} + \epsilon_i, \ \ \ \ \ \ \ \ \ i=1, 2, ..., N$$

where $\beta = {(\beta_i)^T}\_{i=0}^{\ p} \in \mathbb{R}^{p+1}$ is the unknown parameter vector and $\epsilon = {(\epsilon_i)^T}\_{i=1}^{N} \in \mathbb{R}^{N}$ is the sampling error vector. We suppose that every $y_i$ is independent with $E(\epsilon_i) = 0$ and $Var(\epsilon_i) = \sigma^2$, so that:

### $$E(Y|X) = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}$$
### $$Var(Y|X) = \sigma^2$$

If we set $x_{i0} = 1 \  \forall i = 1, 2, ..., N$, then we can write our model equation in the matrix form:

### $$y = X\beta + \epsilon$$

where $y$ is the output vector, $\beta$ is the parameter vector, $\epsilon$ is the error vector and $X \in \mathbb{R}^{N \times (p+1)}$ is the matrix given by $(X)_{ij} = x\_{ij}$ where $i = 1, ..., N$ and $j = 0, 1, 2, ..., p$.

We wish to find $\hat{\beta}$, an estimate of the parameter $\beta$. We use the method of least squares, where we minimize the sum of squared errors:

### $$RSS(\beta) = \sum_{i=1}^{N} \epsilon_i^2 = \sum_{i=1}^{N} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2$$

or, in matrix form,

### $$RSS(\beta) = (y - X\beta)^T (y - X\beta)$$
