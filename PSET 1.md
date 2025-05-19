
---
***Show:*** 
	$w^* = \arg\min_w \|Xw - y\|_2^2 = (X^T X)^{-1} X^T y.$

---
	$w^* = \arg\min_w \|Xw - y\|_2^2$

We rewrite it as a summation:
	$w^* = \arg\min_w \frac{1}{n}\sum_{i=1}^{n}(x^{(i)}w-y^{(i)})^2$

The gradient of our loss function can thus be described as:
	$∇_wL(w)=\vec[\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, ..., \frac{\partial L}{\partial w_d}]$

So each with respect to each $w_j$:
	$\frac{\partial}{\partial w_j}\frac{1}{n}\sum_{i=1}^{n}(x^{(i)}w-y^{(i)})^2=\frac{2}{n}\sum_{i=1}^{n}(x^{(i)}w-y^{(i)})(x_j^i)$, by the chain rule

We set it such that:
	$∇_wL(w)=0$
	$0=\frac{2}{n}\sum_{i=1}^{n}(x^{(i)}w-y^{(i)})(x_j^i)$
	$=X^T(Xw-y)$, by rewriting in matrix form
	$=X^TXw-X^Ty$, by expansion
	$X^TXw=X^Ty$, by rearranging
	$w=(X^TX)^{-1}X^Ty$

---
***Show:*** 
	$∇_wL(w)=(\sigma(wx)-y)x$, given $L(w)=-(ylog(\sigma(wx))+(1-y)log(1-\sigma(wx))$

---
We have sigmoid activation function:
	$\sigma(z)=\frac{1}{1+e^{-z}}$

First we show that $\frac{\partial}{\partial z}\sigma(z)=\sigma(z)(1-\sigma(z))$:
	$\frac{\partial}{\partial z}\sigma(z)=\frac{\partial}{\partial z}\frac{1}{1+e^{-z}}$
	$=\frac{\partial}{\partial z}(1+e^{-z})^{-1}$
	$=-(1+e^{-z})^{-2}\frac{\partial}{\partial z}(1+e^{-z})$
	$=-(\frac{1}{(1+e^{-z})^2})(-e^{-z})$
	$=(\frac{e^{-z}}{1+e^{-z}})(\frac{1}{1+e^{-z}})$
	$=(1-\frac{1}{1+e^{-z}})(\frac{1}{1+e^{-z}})$

Substitute in $\sigma(z)$:
	$=\sigma(z)(1-\sigma(z))$

Now let $z=wx$. We have: 
	$∇_wL(w)=\frac{\partial L}{\partial z}\frac{\partial z}{\partial w}$, and $\frac{\partial z}{\partial w}=\frac{\partial}{\partial w}wx=x$:
	$\frac{\partial L}{\partial z}=-(y\frac{1}{\sigma(z)}\frac{\partial \sigma(z)}{\partial z}) + (1-y)(\frac{1}{1-\sigma(z)})\frac{\partial \sigma(z)}{\partial z}$

Simplify:
	$= -(\frac{y\sigma(z)(1-\sigma(z))}{\sigma(z)})+(1-y)(\frac{\sigma(z)(1-\sigma(z))}{1-\sigma(z)})$
	$=-y(1-\sigma(z))+\sigma(z)(1-y)$
	$=\sigma(z)(1-y)-y(1-\sigma(z))$
	$=\sigma(z)-y\sigma(z)-y+y\sigma(z)$
	$=\sigma(z)-y$

Now find $∇_wL(w)$ given $\frac{\partial z}{\partial w}=x$, and $\frac{\partial L}{\partial z}=\sigma(z)-y$:
	$∇_wL(w)=\frac{\partial L}{\partial z}\frac{\partial z}{\partial w}$
	$=(\sigma(z)-y)x$

Substituting $z=wx$ back in:
	$=(\sigma(wx)-y)x$

Why can’t we (easily) compute the exact solution in closed form? What should we do instead?

We are unable to easily find the exact closed form solution as we introduce a nonlinearity with our activation function. Rather, we can probably think about an iterative process in which we calculate the gradient of the loss function, take a small step in the negative direction of said gradient by adjusting our weights, and repeat until we are satisfied somewhere close to the minimum.