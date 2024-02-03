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

### $$RSS(\beta) = (y - X\beta)^T (y - X\beta) = \langle \epsilon, \epsilon \rangle$$

## Theorem 1 
### The estimate $\hat{\beta}$ that minimizes the sum of squared errors satisfies the equation
### $$X^T X\hat{\beta} = X^T y$$

**Proof**: Given $\epsilon: \mathbb{R}^{p+1} \rightarrow \mathbb{R}^N$ and $RSS: \mathbb{R}^{p+1} \rightarrow \mathbb{R}$ defined by $\epsilon(\beta) = y - X\beta$ and $RSS(\beta) = \langle \epsilon, \epsilon \rangle$, respectively, the derivative of $RSS$ is given by:

### $$[D(\langle \epsilon, \epsilon \rangle)] = 2 \epsilon^T [D(\epsilon)] = -2(y - X\beta)^T X$$

It is clear that $RSS$ cannot be maximized, but nonetheless it can be shown by taking the second derivative, giving us $[D^2(RSS)] = 2 X^T X$. Assuming the columns of $X$ are independent (i.e. its rank equals $p + 1$), $X^T X$ is invertible and so is positive definite, thus the Hessian matrix is positive implying that $RSS$ achieves its minimum at the critical value $\hat{\beta}$.

Setting $[D(\langle \epsilon, \epsilon \rangle)] = 0$ and denoting $\hat{\beta}$ the vector satisfying this equation, we get $X^T X\hat{\beta} = X^T y$ as required, or,

### $$\hat{\beta} = (X^T X)^{-1} X^T y$$

$$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \square$$

Our predicted values of $y$, denoted by $\hat{y}$, s given by:

### $$\hat{y} = X\hat{\beta} = X(X^T X)^{-1} X^T y = Hy$$

where $H = X(X^T X)^{-1} X^T$, called the hat matrix.

## Theorem 2
### The variance of $\hat{\beta}$ is given by
### $$Var(\hat{\beta}) = (X^T X)^{-1} \sigma^2$$

**Proof**: Since every $y_i$ is independent and have variance $\sigma^2$, then $Var(y) = \sigma^2 I$, where $I$ is the identity matrix, and so:

### $$Var(\hat{\beta}) = Var((X^T X)^{-1} X^T y) = (X^T X)^{-1} X^T Var(y) [(X^T X)^{-1} X^T]^T = (X^T X)^{-1} X^T X (X^T X)^{-1} \sigma^2 = (X^T X)^{-1} \sigma^2$$
as required.
$$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \square$$

Let $\hat{\epsilon}_i$ denote the error between the observed value and the predicted value of the $i^{th}$ output, called the **residual**, given by $\epsilon_i = y_i - \hat{y}_i$. By letting $\hat{\epsilon} = y - \hat{y}$ be the residual vector, the expectation is given by $E(\hat{\epsilon}) = E(y) - E(\hat{y}) = X\beta - E(H y) = X\beta - HX\beta = X\beta - X(X^T X)^{-1} X^T X\beta = X\beta - X\beta = 0$ so the residual is an unbiased estimator of the error $\epsilon$.

## Theorem 3
### The mean sum - of - squared residuals given by:
### $$\hat{\sigma}^2 = \frac{1}{N - p -1} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$
### is an unbiased estimator of the error variance $\sigma^2$.

**Proof**: We will find the expectation of the sum of squared residuals, denoted by SSE, and show that dividing by $N - p - 1$ gives us $\sigma^2$. 

First, write $\hat{\epsilon} = y - \hat{y} = y - Hy = (I - H)y$. So $SSE = \hat{\epsilon}^T \hat{\epsilon} = y^T (I - H)^T (I - H) y = y^T (I - H)^2 y = y^T (I - H)y = y^T y - y^T Hy$ (this follows because the matrix $I - H$ is symmetric and idempotent). Since $y = X\beta + \epsilon$, we substitute and simplify:

### $$y^T y - y^T Hy = (X\beta + \epsilon)^T (X\beta + \epsilon) - (X\beta + \epsilon)^T H (X\beta + \epsilon)$$
### $$= (X\beta + \epsilon)^T (X\beta + \epsilon) - (X\beta + \epsilon)^T (X\beta + H \epsilon)$$
### $$= (X\beta + \epsilon)^T \epsilon - (X\beta + \epsilon)^T H \epsilon$$
### $$= (\beta^T X^T - \beta^T X^T H) \epsilon + \epsilon^T \epsilon - \epsilon^T H \epsilon$$
### $$= (\beta^T X^T - \beta^T X^T (X(X^T X)^{-1} X^T)) \epsilon + \epsilon^T \epsilon - \epsilon^T H \epsilon$$
### $$= (\beta^T X^T - \beta^T X^T) \epsilon + \epsilon^T \epsilon - \epsilon^T H \epsilon$$
### $$= \epsilon^T \epsilon - \epsilon^T H \epsilon$$

We have:

### $$E(\epsilon^T \epsilon) = \sum_{i=1}^{N} E(\epsilon_i^2) = \sum_{i=1}^{N} (\sigma^2 + E(\epsilon_i)^2) = N\sigma^2$$
### $$E(\epsilon^T H \epsilon) = E(\sum_{i=1}^{N} \sum_{j=1}^{N} (\epsilon^T)\_{1i} (H)\_{ij} (\epsilon)\_{j1}) = \sum_{i=1}^{N} \sum_{j=1}^{N} (H)\_{ij} E(\epsilon\_{j1} \epsilon\_{i1}) = \sum_{i=1}^{N} (H)_{ii} E(\epsilon_i^2) = \sigma^2 Tr(H)$$

But since $H = X(X^T X)^{-1} X^T$, the trace is given by $Tr(H) = Tr(X(X^T X)^{-1} X^T) = Tr(X^T X(X^T X)^{-1}) = Tr(I_{(p+1) \times (p+1)}) = p + 1$. So $E(\epsilon^T H \epsilon) = (p + 1) \sigma^2$.

Thus $E(SSE) = E(\epsilon^T \epsilon) - E(\epsilon^T H \epsilon) = (N - p - 1) \sigma^2$. Dividing SSE by (N - p - 1) and putting:

### $$\hat{\sigma}^2 = \frac{SSE}{N - p - 1} = \frac{1}{N - p -1} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

concludes the proof.
$$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \square$$
