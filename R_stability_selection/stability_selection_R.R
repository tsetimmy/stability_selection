# http://members.cbio.mines-paristech.fr/~jvert/svn/tutorials/practical/reginference/stabilityselection.R
"stabilityselection" <- function(x,y,nbootstrap=100,nsteps=20,alpha=0.2,plotme=FALSE, non_zero_indices=0)
{
	# Stability selection in the spirit of Meinshausen&Buhlman
	# JP Vert, 14/9/2010
	
	# x is the n*p design matrix, y the n*1 variable to predict
	# x should be normalized to unit variance per column before calling this function (if you want to)
	# the result is a score (length p) for each feature, the probability that each feature is selected during the first nsteps steps of the Lasso path when half of the samples are used and the features are reweigthed by a random weight uniformaly sampled in [alpha,1]. This probability is estimated by nbootstrap bootstrap samples
	require(lars)
	dimx <- dim(x)
	n <- dimx[1]
	p <- dimx[2]
	halfsize <- as.integer(n/2)
	freq <- matrix(0,nsteps+1,p)
	
	for (i in seq(nbootstrap)) {
		
		# Randomly reweight each variable
		xs <- t(t(x)*runif(p,alpha,1))
		
		# Ramdomly split the sample in two sets
		perm <- sample(dimx[1])
		i1 <- perm[1:halfsize]
		i2 <- perm[(halfsize+1):n]
		
		# run the randomized lasso on each sample and check which variables are selected
		r <- lars(xs[i1,],y[i1],max.steps=nsteps,normalize=FALSE, use.Gram=FALSE)
        print(r$lambda)
		freq <- freq + abs(sign(coef.lars(r)))
		r <- lars(xs[i2,],y[i2],max.steps=nsteps,normalize=FALSE, use.Gram=FALSE)
        print(r$lambda)
		freq <- freq + abs(sign(coef.lars(r)))
        print('--------------------------------')
		}
		
	# normalize frequence in [0,1]
	freq <- freq/(2*nbootstrap)
	
	if (plotme) {
        print(dim(freq))
        print(non_zero_indices)
        print(dim(freq[,non_zero_indices]))
		matplot(freq,type='l',xlab="LARS iteration",ylab="Frequency")
		matplot(freq[,non_zero_indices],type='l',xlab="LARS iteration",ylab="Frequency")
        q()
	}
	
	# the final stability score is the maximum frequency over the steps
	result <- apply(freq,2,max)
}



library("base")

n <- 100
p <- 1000

s <- 3
non_zero_indices <- sample(p)[1:s]
beta <- rep(0, p)
beta[non_zero_indices] <- runif(s)
print(beta[non_zero_indices])

snr <- .5
sigma2_noise <- 1 / snr
noise_vec <- rnorm(n, mean=0, sd=sqrt(sigma2_noise/n))

X <- matrix(rnorm(n*p, mean=0, sd=1), n, p)

normalizer <- sqrt(colSums(X^2))
X_scaled <- t(t(X) / normalizer)


y <- X_scaled %*% beta + noise_vec





tmp <- stabilityselection(X_scaled, y, plotme=TRUE, non_zero_indices=non_zero_indices)
#print(tmp)



