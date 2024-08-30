if (!require(miselect)) {
    quit("no", 2)
}

library(miselect)

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

    # make dataset
    x <- lapply(X_list, function(x_) { as.matrix(x_[, predictors]) })
    y <- lapply(X_list, function(x_) { as.vector(x_[, target]) })

    # options
    pf <- rep(1, length(predictors))

    # run to search adaptive weights
    adWeight_cv <- rep(1, length(predictors))
    
    fit <- saenet(x, y, pf, adWeight_cv, weights)
    CV <- cv.saenet(x, y, pf, adWeight_cv, weights, lambda=fit$lambda)
    coef_cv <- fit$coef[, 1, ]

    lambda_list_cv <- fit$lambda
    
    if (ad_weight_type == "min") {
        key <- "lambda.min"
    } else if (ad_weight_type == "1se") {
        key <- "lambda.1se"
    }
    lambda_cv <- CV[[key]]
    
    for (i in 1:length(lambda_list_cv)) {
        if (abs(lambda_list_cv[[i]] / lambda_cv - 1.0) < 1.0e-6)
            n_lambda_cv = i
    }
    
    # run to search beta
    beta_hat_cv <- coef_cv[n_lambda_cv, 2:dim(coef_cv)[[2]]]
    abs_beta_hat_cv <- abs(beta_hat_cv) + 1.0 / (n_sample * n_imputation)
    nu <- log(n_feature) / log(n_sample * n_imputation)
    gamma <- ceiling(2 * nu / (1 - nu) + 1)
    adWeight_cv <- abs_beta_hat_cv ** (-gamma)
    
    fit <- saenet(x, y, pf, adWeight_cv, weights)
    CV <- cv.saenet(x, y, pf, adWeight_cv, weights, lambda=fit$lambda)
    
    lambda_list_cv <- fit$lambda
    lambda_cv <- CV[[key]]
    for (i in 1:length(lambda_list_cv)) {
        if (abs(lambda_list_cv[[i]] / lambda_cv - 1.0) < 1.0e-6)
            n_lambda_cv = i
    }
    beta_hat_cv <- coef_cv[n_lambda_cv, 2:dim(coef_cv)[[2]]]
    
    B[target, predictors] <- beta_hat_cv
}

path <- file.path(temp_dir, "result_adj_mat.csv")
write.csv(B, path, row.names=F)

quit("no", 0)
