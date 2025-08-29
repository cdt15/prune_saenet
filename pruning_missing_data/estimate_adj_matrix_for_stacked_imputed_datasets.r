if (!require(miselect)) {
    quit("no", 2)
}

library(miselect)

cv.saenet_modified <- function(x, y, pf, adWeight, weights, family = c("gaussian", "binomial"),
                      alpha = 1, nlambda = 100, lambda.min.ratio =
                        ifelse(isTRUE(all.equal(adWeight, rep(1, p))), 1e-3, 1e-6),
                      lambda = NULL, nfolds = 5, foldid = NULL, maxit = 1000,
                      eps = 1e-5)
{
  call <- match.call()
  
  if (!is.list(x))
    stop("'x' should be a list of numeric matrices.")
  if (any(sapply(x, function(.x) !is.matrix(.x) || !is.numeric(.x))))
    stop("Every 'x' should be a numeric matrix.")
  
  dim <- dim(x[[1]])
  n <- dim[1]
  p <- dim[2]
  m <- length(x)
  
  if (!is.numeric(nfolds) || length(nfolds) > 1)
    stop("'nfolds' should a be single number.")
  
  if (!is.null(foldid))
    if (!is.numeric(foldid) || length(foldid) != length(y[[1]]))
      stop("'nfolds' should a be single number.")
  
  fit <- saenet(x, y, pf, adWeight, weights, family, alpha, nlambda,
                lambda.min.ratio, lambda, maxit, eps)
  
  X <- do.call("rbind", x)
  Y <- do.call("c", y)
  
  weights <- rep(weights / m , m)
  
  if (!is.null(foldid)) {
    if (!is.numeric(foldid) || !is.vector(foldid) || length(foldid) != n)
      stop("'foldid' must be length n numeric vector.")
    nfolds <- max(foldid)
  } else {
    r     <- n %% nfolds
    q     <- (n - r) / nfolds
    if(r == 0) {
      foldid = rep(seq(nfolds), q)
    } else {
      foldid = c(rep(seq(nfolds), q), seq(r))
    }
    foldid <- sample(foldid, n)
    foldid <- rep(foldid, m)
  }
  if (nfolds < 3)
    stop("'nfolds' must be bigger than 3.")
  
  lambda  <- fit$lambda
  nlambda <- length(lambda)
  X.scaled <- scale(X, scale = apply(X, 2, function(.X) stats::sd(.X) * sqrt(m)))
  
  cvm  <- array(0, c(nlambda, length(alpha), nfolds))
  cvse <- matrix(nlambda, length(alpha))
  for (j in seq(nfolds)) {
    Y.train <- Y[foldid != j]
    X.train <- subset_scaled_matrix(X.scaled, foldid != j)
    w.train <- weights[foldid != j]
    
    X.test  <- X[foldid == j, , drop = F]
    Y.test  <- Y[foldid == j]
    w.test <- weights[foldid == j]
    
    cv.fit <- switch(match.arg(family),
                     gaussian = fit.saenet.gaussian(X.train, Y.train, n, p, m, w.train,
                                                    nlambda, lambda, alpha, pf, adWeight,
                                                    maxit, eps),
                     binomial = fit.saenet.binomial(X.train, Y.train, n, p, m, w.train,
                                                    nlambda, lambda, alpha, pf, adWeight,
                                                    maxit, eps)
    )
    
    cvm[,, j] <- cv.saenet.err(cv.fit, X.test, Y.test, w.test, m)
  }
  
  cvse <- apply(cvm, c(1, 2), stats::sd) / sqrt(nfolds)
  cvm  <- apply(cvm, c(1, 2), mean)
  
  min.id = which(cvm == min(cvm), arr.ind = TRUE)
  se = cvse[min.id[1,1], min.id[1,2]] # modified
  range = min(cvm) + se
  
  all.id = which(cvm < range, arr.ind = TRUE)
  lambda.seq = lambda[all.id[, 1]]
  alpha.seq = alpha[all.id[, 2]]
  L1 = lambda.seq * alpha.seq
  L1.max.id = which(L1 == max(L1))
  lambda.1se.id = all.id[L1.max.id, 1]
  alpha.1se.id = all.id[L1.max.id, 2]
  lambda.1se = lambda[lambda.1se.id]
  alpha.1se = alpha[alpha.1se.id]
  i.min    <- which.min(apply(cvm, 1, min))
  j.min    <- which.min(apply(cvm, 2, min))
  
  lambda.min <- fit$lambda[i.min]
  alpha.min <- fit$alpha[j.min]
  
  structure(list(call = call, lambda = fit$lambda, alpha = alpha, cvm = cvm,
                 cvse = cvse, saenet.fit = fit, 
                 lambda.min = lambda.min,
                 alpha.min = alpha.min, 
                 lambda.1se = lambda.1se, alpha.1se =
                 alpha.1se, df = fit$df), class = "cv.saenet")
}

environment(cv.saenet_modified) <- asNamespace("miselect")

# command args
temp_dir <- NULL
seed <- NULL

args <- commandArgs(trailingOnly=TRUE)
for (arg in args) {
    s <- strsplit(arg, "=")[[1]]
    if (length(s) < 2) next

    if (s[1] == "--temp_dir") {
        temp_dir <- paste(s[2:length(s)], collapse="=")
    } else if (s[1] == "--rs") {
        seed <- strtoi(s[2])
    }
}

if (!is.null(seed))
    set.seed(seed)

# args
X_list <- c()
causal_order <- NULL
weights <- NULL
ad_weight_type <- NULL
prior_knowledge <- NULL
is_discrete <- NULL

# X_list
path <- file.path(temp_dir, "X_names.csv")
X_names <- read.csv(path, sep=",", header=FALSE)
X_names <- lapply(X_names, function(x) {paste(temp_dir, "/", x, sep="")})[[1]]
X_list <- lapply(X_names, read.csv, sep=",", header=FALSE)

# causal_order
path <- file.path(temp_dir, "causal_order.csv")
causal_order <- read.csv(path, sep=',', header=FALSE)
causal_order <- causal_order + 1

# ad_weight_type (1se or min)
path <- file.path(temp_dir, "ad_weight_type.csv")
ad_weight_type <- read.csv(path, sep=',', header=FALSE)[1, 1]

# weights
path <- file.path(temp_dir, "weights.csv")
weights <- read.csv(path, sep=",", header=FALSE)
weights <- as.vector(weights)[[1]]

# prior_knowledge
path <- file.path(temp_dir, "prior_knowledge.csv")
if (file.exists(path))
    prior_knowledge <- read.csv(path, sep=',', header=FALSE)

# is_discrete
path <- file.path(temp_dir, "is_discrete.csv")
if (file.exists(path)) {
    is_discrete <- read.csv(path, sep=",", header=FALSE)
    is_discrete <- as.vector(is_discrete)
}

# params
n_imputation <- length(X_list)
n_sample <- dim(X_list[[1]])[[1]]
n_feature <- dim(X_list[[1]])[[2]]

# estimated adjacency matrix
B <- data.frame(matrix(0, nrow=n_feature, ncol=n_feature))

for (i in 2:dim(causal_order)[1]) {
    predictors <- causal_order[1:i - 1, 1]
    target <- causal_order[i, 1]

    # apply prior_knowledge
    if (!is.null(prior_knowledge)) {
        drop_targets <- c()

        for (j in 1:length(predictors))
            if (prior_knowledge[target, predictors[j]] == 0)
                drop_targets <- append(-1 * j, drop_targets)

        if (length(drop_targets) > 0)
            predictors <- predictors[drop_targets]
    }
    
    if (length(predictors) == 0) next

    family <- "gaussian"
    if (!is.null(is_discrete) && is_discrete[[target]])
        family <- "binomial"

    # make dataset
    x <- lapply(X_list, function(x_) { as.matrix(x_[, predictors]) })
    y <- lapply(X_list, function(x_) { as.vector(x_[, target]) })

    # options
    pf <- rep(1, length(predictors))

    # run to search adaptive weights
    adWeight_cv <- rep(1, length(predictors))
    
    fit <- saenet(x, y, pf, adWeight_cv, weights, family=family)
    CV <- cv.saenet_modified(x, y, pf, adWeight_cv, weights, lambda=fit$lambda, family=family)
    coef_cv <- fit$coef[, 1, ]

    lambda_list_cv <- fit$lambda
    
    if (ad_weight_type == "min") {
        key <- "lambda.min"
    } else if (ad_weight_type == "1se") {
        key <- "lambda.1se"
    }
    lambda_cv <- CV[[key]]
    
    for (j in 1:length(lambda_list_cv)) {
        if (abs(lambda_list_cv[[j]] / lambda_cv - 1.0) < 1.0e-6)
            n_lambda_cv = j
    }
    
    # run to search beta
    beta_hat_cv <- coef_cv[n_lambda_cv, 2:dim(coef_cv)[[2]]]
    abs_beta_hat_cv <- abs(beta_hat_cv) + 1.0 / (n_sample * n_imputation)
    nu <- log(n_feature) / log(n_sample * n_imputation)
    gamma <- ceiling(2 * nu / (1 - nu) + 1)
    adWeight_cv <- abs_beta_hat_cv ** (-gamma)
    
    fit <- saenet(x, y, pf, adWeight_cv, weights, family=family)
    CV <- cv.saenet_modified(x, y, pf, adWeight_cv, weights, lambda=fit$lambda, family=family)
    
    lambda_list_cv <- fit$lambda
    lambda_cv <- CV[[key]]
    for (j in 1:length(lambda_list_cv)) {
        if (abs(lambda_list_cv[[j]] / lambda_cv - 1.0) < 1.0e-6)
            n_lambda_cv = j
    }
    beta_hat_cv <- coef_cv[n_lambda_cv, 2:dim(coef_cv)[[2]]]
    
    B[target, predictors] <- beta_hat_cv
}

path <- file.path(temp_dir, "result_adj_mat.csv")
write.csv(B, path, row.names=F)

quit("no", 0)
